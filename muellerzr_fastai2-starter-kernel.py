!pip install fastai2 --quiet
from fastai2.vision.all import *



path = Path('../input/global-wheat-detection')
path.ls()
df = pd.read_csv(path/'train.csv')
df.head()
df['bbox'].isna().sum()
imgs = get_image_files(path/'train')
len(imgs) == df['image_id'].nunique()
len(imgs) - df['image_id'].nunique()
im_df = df['image_id'].unique()
im_df = [fn + '.jpg' for fn in im_df]
im_df[:5]
fns = [Path(str(path/'train') + f'/{fn}') for fn in im_df]
fns[0]
fns[0].name[:-4]
def get_items(noop): return fns
df['label'] = 'wheat'
df_np = df.to_numpy()
coco_source = untar_data(URLs.COCO_TINY)

images, lbl_bbox = get_annotations(coco_source/'train.json')

img2bbox = dict(zip(images, lbl_bbox))
fn = images[0]; fn
img2bbox[fn][0][0]
def get_tmp_bbox(fn):

    "Grab bounding boxes from `DataFrame`"

    rows = np.where((df_np[:, 0] == fn.name[:-4]))

    bboxs = df_np[rows][:,3]

    bboxs = [b.replace('[', '').replace(']', '') for b in bboxs]

    return np.array([np.fromstring(b, sep=',') for b in bboxs])
def get_tmp_lbl(fn):

    "Grab label from `DataFrame`"

    rows = np.where((df_np[:, 0] == fn.name[:-4]))

    return df_np[rows][:,5]
fnames = df['image_id'].unique(); fnames[:3]
bboxs = get_tmp_bbox(fns[0])

lbls = get_tmp_lbl(fns[0])

arr = np.array([fns[0].name[:-4], bboxs, lbls])
arr
for fname in fns[1:]:

    bbox = get_tmp_bbox(fname)

    lbl = get_tmp_lbl(fname)

    arr2 = np.array([fname.name[:-4], bbox, lbl])

    arr = np.vstack((arr, arr2))
arr[:,1][0][0][0] + arr[:,1][0][0][2]
arr[:,1][0][1]
for i, im in enumerate(arr[:,1]):

    for j, box in enumerate(im):

        arr[:,1][i][j][2] = box[0]+box[2]

        arr[:,1][i][j][3] = box[1]+box[3]
arr[0][1][0]
np.save('data.npy', arr)
def get_bbox(fn):

    "Gets bounding box from `fn`"

    idx = np.where((arr[:,0] == fn.name[:-4]))

    return arr[idx][0][1]
def get_lbl(fn):

    "Get's label from `fn`"

    idx = np.where((arr[:,0] == fn.name[:-4]))

    return arr[idx][0][2]
%%timeit

_ = get_bbox(fns[0])
%%timeit

_ = get_lbl(fns[0])
wheat = DataBlock(blocks=(ImageBlock, BBoxBlock, BBoxLblBlock),

                 get_items=get_items,

                 splitter=RandomSplitter(),

                 get_y=[get_bbox, get_lbl],

                 item_tfms=Resize(256, method=ResizeMethod.Pad),

                 n_inp=1)
dls = wheat.dataloaders(path,bs=32)
dls.show_batch(max_n=1, figsize=(12,12))
batch = dls.one_batch()
batch[1].shape