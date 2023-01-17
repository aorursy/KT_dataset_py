# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





!pip install torch torchvision feather-format kornia pyarrow --upgrade   > /dev/null

!pip install git+https://github.com/fastai/fastai2                 > /dev/null
from fastai2.torch_basics import *

from fastai2.test import *

from fastai2.layers import *

from fastai2.data.all import *

from fastai2.optimizer import *

from fastai2.learner import *

from fastai2.metrics import *

from fastai2.vision.all import *

from fastai2.vision.learner import *

from fastai2.vision.models import *

from fastai2.callback.all import *

from fastai2.basics           import *

from fastai2.vision.all       import *

from fastai2.medical.imaging  import *

np.set_printoptions(linewidth=120)

matplotlib.rcParams['image.cmap'] = 'bone'

set_seed(42)

set_num_threads(1)


path = Path('../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/')

path_train = path/'stage_2_train'

path_test = path/'stage_2_test'





path_dest = Path()

path_dest.mkdir(exist_ok=True)



path_inp = Path('../input')

path_df = path_inp/'rsna-stage2-meta'

path_df.ls()
df_lbls = pd.read_feather(path_df/'labels_stage_2.fth')

df_tst = pd.read_feather(path_df/'fns_test_2.fth')

df_trn = pd.read_feather(path_df/'fns_train_2.fth').dropna(subset=['img_pct_window'])

comb = df_trn.join(df_lbls.set_index('ID'), 'SOPInstanceUID')
repr_flds = ['BitsStored','PixelRepresentation']

df1 = comb.query('(BitsStored==12) & (PixelRepresentation==0)')

df2 = comb.query('(BitsStored==12) & (PixelRepresentation==1)')

df3 = comb.query('BitsStored==16')

dfs = L(df1,df2,df3)
dfs[0]
dfs[1]
dfs[2]
def df2dcm(df): 

    return L(Path(o).dcmread() for o in df.fname.values)
df_iffy = df1[df1.RescaleIntercept>-1000]

dcms = df2dcm(df_iffy)



_,axs = subplots(4,4, imsize=3)

for i,ax in enumerate(axs.flat): dcms[i].show(ax=ax)
dcm = dcms[2]

d = dcm.pixel_array

plt.hist(d.flatten());
len(dcms)
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

dcm.show(ax=axs[0]);   

dcm.show(dicom_windows.brain, ax=axs[1])

def fix_pxrepr(dcm):

    if dcm.PixelRepresentation != 0 or dcm.RescaleIntercept<-1000: return

    x = dcm.pixel_array + 1000

    px_mode = 4096

    x[x>=px_mode] = x[x>=px_mode] - px_mode

    dcm.PixelData = x.tobytes()

    dcm.RescaleIntercept = -1000
dcms = df2dcm(df_iffy)

dcms.map(fix_pxrepr)



_,axs = subplots(2,5, imsize=3)

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

comb.head()
comb_ = comb.reset_index()

comb_.index
comb_ = comb_.drop(['pct_cut'],axis =1)
comb_.to_feather('comb_stage_2.fth')

from IPython.display import FileLink, FileLinks

FileLink('comb_stage_2.fth')
dcm = Path(dcms[3].filename).dcmread()

fix_pxrepr(dcm)

px = dcm.windowed(*dicom_windows.brain)

show_image(px);
blurred = gauss_blur2d(px, 100)

show_image(blurred);
show_image(blurred>0.3)
dcm.show(dicom_windows.brain)

show_image(dcm.mask_from_blur(dicom_windows.brain), cmap=plt.cm.Reds, alpha=0.6)
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
(path_dest/'bins_2.pkl').save(bins)
from IPython.display import FileLink, FileLinks

FileLink('bins_2.pkl')