from zipfile import ZipFile 
!pip install torch torchvision feather-format kornia pyarrow --upgrade   > /dev/null

!pip install git+https://github.com/fastai/fastai_dev                    > /dev/null
from fastai2.basics           import *

from fastai2.vision.all       import *

from fastai2.medical.imaging  import *



np.set_printoptions(linewidth=120)

matplotlib.rcParams['image.cmap'] = 'bone'

torch.set_num_threads(1)

set_num_threads(1)



path = Path('../input/rsna-intracranial-hemorrhage-detection/')

path_trn = path/'stage_1_train_images'

path_tst = path/'stage_1_test_images'

path_dest = Path()

path_dest.mkdir(exist_ok=True)



path_inp = Path('../input')
path_df = path_inp/'creating-a-metadata-dataframe'

df_lbls = pd.read_feather(path_df/'labels.fth')

df_tst = pd.read_feather(path_df/'df_tst.fth')

df_trn = pd.read_feather(path_df/'df_trn.fth').dropna(subset=['img_pct_window'])

comb = df_trn.join(df_lbls.set_index('ID'), 'SOPInstanceUID')
repr_flds = ['BitsStored','PixelRepresentation']

df1 = comb.query('(BitsStored==12) & (PixelRepresentation==0)')

df2 = comb.query('(BitsStored==12) & (PixelRepresentation==1)')

df3 = comb.query('BitsStored==16')

dfs = L(df1,df2,df3)
def df2dcm(df): return L(Path(o).dcmread() for o in df.fname.values)
df_iffy = df1[df1.RescaleIntercept>-100]

dcms = df2dcm(df_iffy)



_,axs = subplots(2,4, imsize=3)

for i,ax in enumerate(axs.flat): dcms[i].show(ax=ax)
dcm = dcms[2]

d = dcm.pixel_array

plt.hist(d.flatten());
d1 = df2dcm(df1.iloc[[0]])[0].pixel_array

plt.hist(d1.flatten());
scipy.stats.mode(d.flatten()).mode[0]
d += 1000



px_mode = scipy.stats.mode(d.flatten()).mode[0]

d[d>=px_mode] = d[d>=px_mode] - px_mode

dcm.PixelData = d.tobytes()

dcm.RescaleIntercept = -1000
plt.hist(dcm.pixel_array.flatten());
_,axs = subplots(1,2)

dcm.show(ax=axs[0]);   dcm.show(dicom_windows.brain, ax=axs[1])
def fix_pxrepr(dcm):

    if dcm.PixelRepresentation != 0 or dcm.RescaleIntercept<-100: return

    x = dcm.pixel_array + 1000

    px_mode = 4096

    x[x>=px_mode] = x[x>=px_mode] - px_mode

    dcm.PixelData = x.tobytes()

    dcm.RescaleIntercept = -1000
dcms = df2dcm(df_iffy)

dcms.map(fix_pxrepr)



_,axs = subplots(2,4, imsize=3)

for i,ax in enumerate(axs.flat): dcms[i].show(ax=ax)
df_iffy.img_pct_window[:10].values
plt.hist(comb.img_pct_window,40);
comb = comb.assign(pct_cut = pd.cut(comb.img_pct_window, [0,0.02,0.05,0.1,0.2,0.3,1]))

comb.pivot_table(values='any', index='pct_cut', aggfunc=['sum','count']).T
comb.drop(comb.query('img_pct_window<0.02').index, inplace=True)
df_lbl = comb.query('any==True')

n_lbl = len(df_lbl)

n_lbl
df_nonlbl = comb.query('any==False').sample(n_lbl//2)

len(df_nonlbl)
comb = pd.concat([df_lbl,df_nonlbl])

len(comb)
dcm = Path(dcms[3].filename).dcmread()

fix_pxrepr(dcm)
px = dcm.windowed(*dicom_windows.brain)

show_image(px);
blurred = gauss_blur2d(px, 100)

show_image(blurred);
show_image(blurred>0.3);
_,axs = subplots(1,4, imsize=3)

for i,ax in enumerate(axs.flat):

    dcms[i].show(dicom_windows.brain, ax=ax)

    show_image(dcms[i].mask_from_blur(dicom_windows.brain), cmap=plt.cm.Reds, alpha=0.6, ax=ax)
def pad_square(x):

    r,c = x.shape

    d = (c-r)/2

    pl,pr,pt,pb = 0,0,0,0

    if d>0: pt,pd = int(math.floor( d)),int(math.ceil( d))        

    else:   pl,pr = int(math.floor(-d)),int(math.ceil(-d))

    return np.pad(x, ((pt,pb),(pl,pr)), 'minimum')



def crop_mask(x):

    mask = x.mask_from_blur(dicom_windows.brain)

    bb = mask2bbox(mask)

    if bb is None: return

    lo,hi = bb

    cropped = x.pixel_array[lo[0]:hi[0],lo[1]:hi[1]]

    x.pixel_array = pad_square(cropped)
_,axs = subplots(1,2)

dcm.show(ax=axs[0])

crop_mask(dcm)

dcm.show(ax=axs[1]);
htypes = 'any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural'



def get_samples(df):

    recs = [df.query(f'{c}==1').sample() for c in htypes]

    recs.append(df.query('any==0').sample())

    return pd.concat(recs).fname.values



sample_fns = concat(*dfs.map(get_samples))

sample_dcms = tuple(Path(o).dcmread().scaled_px for o in sample_fns)

samples = torch.stack(sample_dcms)

bins = samples.freqhist_bins()
(path_dest/'bins.pkl').save(bins)
def dcm_tfm(fn): 

    fn = Path(fn)

    try:

        x = fn.dcmread()

        fix_pxrepr(x)

        #print('read', fn)

    except Exception as e:

        print(fn,e)

        raise SkipItemException

    if x.Rows != 512 or x.Columns != 512: x.zoom_to((512,512))

    return x.scaled_px
fns = list(comb.fname.values)

path_dest = path_dest/'train_png'

# NB: Use bs=512 for GPUs with <16GB RAM

#bs=1024

bs=512



dsrc = DataSource(fns, [[dcm_tfm],[os.path.basename]])

# it worked with only 1 worker for the moment

dl = TfmdDL(dsrc, bs=bs, num_workers=1)


def save_img(x, buf, bins, compress_level, format="TIFF"):

    im = Image.fromarray(x.to_uint16(bins))

    if compress_level: im.save(buf, compress_level=compress_level, format=format)

    else: im.save(buf, compress_type=Image.RLE, format=format)

def dest_fname_dest(fname, dest, suffix): return str(dest/Path(fname).with_suffix(suffix))

    

def save_cropped_img(o, dest):

    def dest_fname(fname): return dest_fname_dest(fname, dest, suffix='.tiff')

    fname,px,zip_file = o

    buf = io.BytesIO()

    save_img(px, buf, bins=bins, compress_level=9, format='TIFF')

    zip_file.writestr(dest_fname_dest(fname, dest), buf.getvalue())

def process_batch(pxs, fnames, dest, n_workers=4):

    zip_files = []

    for worker in range(n_workers):

        zip_files.append(ZipFile(dest/'result_dataset_{}.zip'.format(worker), 'w'))

    pxs = pxs.cuda()

    masks = pxs.mask_from_blur(dicom_windows.brain)

    bbs = mask2bbox(masks)

    gs = crop_resize(pxs, bbs, 256).cpu().squeeze()

    #g = partial(save_cropped_img, dest=dest)

    parallel(save_cropped_img, zip(fnames, gs, zip_files), n_workers=n_workers, progress=False, dest=dest)

    

def process_batch_no_parallel(pxs, fnames, dest, zip_file):

    pxs = pxs.cuda()

    masks = pxs.mask_from_blur(dicom_windows.brain)

    bbs = mask2bbox(masks)

    gs = crop_resize(pxs, bbs, 256).cpu().squeeze()

    for fname, px in zip(fnames, gs):

        buf = io.BytesIO()

        save_img(px, buf, bins=bins, compress_level=9, format='TIFF')

        zip_file.writestr(dest_fname_dest(fname, dest, suffix='.tiff'), buf.getvalue())

#!rm -f core

dest = path_dest

dest.mkdir(exist_ok=True)

# , force_zip64=True

with ZipFile(dest/'result_dataset.zip', 'w' ) as zip_file:

    for b in progress_bar(dl): process_batch_no_parallel(*b, dest=dest, zip_file=zip_file)

dest = path_dest

dest.mkdir(exist_ok=True)

#ZipFile(dest/'result_dataset.zip', 'w')

for path in  ['.', str(dest)]:

    for fname in os.listdir(path):

        full_fname = os.path.join(path, fname)

        print(full_fname, os.stat(full_fname).st_size)
# Uncomment this to test and time a single batch



# %time process_batch(*dl.one_batch(), n_workers=4)
# Uncomment this to view some processed images



# for i,(ax,fn) in enumerate(zip(subplots(2,4)[1].flat,fns)):

#     pngfn = dest/Path(fn).with_suffix('.png').name

#     a = pngfn.png16read()

#     show_image(a,ax=ax)