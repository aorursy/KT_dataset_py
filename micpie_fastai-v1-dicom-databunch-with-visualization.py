import pandas as pd

import pydicom

from fastai.vision import *

from fastai.data_block import _maybe_squeeze

from collections import defaultdict
#!wget "https://gist.githubusercontent.com/FedeMiorelli/640bbc66b2038a14802729e609abfe89/raw/34a26667e1528c9e4465cbc0be30d10cbe8d4a40/turbo_colormap_mpl.py"
#import turbo_colormap_mpl
# show fastai version

__version__
df = pd.read_csv('../input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv')

df['fn'] = df.ID.apply(lambda x: '_'.join(x.split('_')[:2]) + '.dcm')

df.columns = ['ID', 'probability', 'fn']

df['label'] = df.ID.apply(lambda x: x.split('_')[-1])

df.drop_duplicates('ID', inplace=True)

pivot = df.pivot(index='fn', columns='label', values='probability')

pivot.reset_index(inplace=True)

d = defaultdict(list)



for fn in df.fn.unique(): d[fn]



for tup in df.itertuples():

    if tup.probability: d[tup.fn].append(tup.label)

        

ks, vs = [], []



for k, v in d.items():

    ks.append(k), vs.append(' '.join(v))

    

df_train = pd.DataFrame(data={'fn': ks, 'labels': vs})
df_train.labels.fillna('', inplace=True)
# safety check of df shape

assert df_train.shape == (674258, 2)
# remove damaged dicom file from df

df_train = df_train[df_train['fn'] != 'ID_6431af929.dcm']
# safety check of df shape

assert df_train.shape == (674258-1, 2)
# change here the colormap for the entire notebook

#cmap='turbo'

cmap='jet'
class DicomImage(Image):

    "DicomImage to support applying transforms to image data in `px`."

    def __init__(self, px:Tensor):

        self._px = px

        self._logit_px=None

        self._flow=None

        self._affine_mat=None

        self.sample_kwargs = {}

    

    def _repr_image_format(self, format_str):

        with BytesIO() as str_buffer:

            plt.imsave(str_buffer, image2np(self.px[0].unsqueeze(0)), # We show only one channel!

                       format=format_str, cmap=cmap)

            return str_buffer.getvalue()

        

    def clone(self):

        "Mimic the behavior of torch.clone for `Image` objects."

        return self.__class__(self.px.clone())



    @property

    def data(self)->TensorImage:

        "Return this images pixels as a tensor."

        return self.px

    

    def show(self, ax:plt.Axes=None, figsize:tuple=(3,3), title:Optional[str]=None, hide_axis:bool=True,

              cmap:str=cmap, y:Any=None, **kwargs):

        "Show image on `ax` with `title`, overlaid with optional `y`"

        ax = show_dicom_image(self, ax=ax, hide_axis=hide_axis, cmap=cmap, figsize=figsize)

        if y is not None: y.show(ax=ax, **kwargs)

        if title is not None: ax.set_title(title)
def open_dicom_image(fn:PathOrStr, cls:type=DicomImage, 

                     after_open:Callable=None, expand=True,

                     clamp_min=-1024, clamp_max=1024)->Image:

    "Return `Image` object created from image in file `fn`."

    

    ds = pydicom.dcmread(fn) # open dicom image as dicom dataset

    img = ds.pixel_array # get pixel data as np array

    

    # Convert to Hounsfield units (HU) and rescale and set intercept.

    # In this setup we only take a look at the values between -1024 and 1024.

    # Values below will be set to -1024, values above to 1024

    resc_img = img * ds.RescaleSlope + ds.RescaleIntercept

    resc_img[resc_img < -1024] = clamp_min # Clamp to minimum value   

    resc_img[resc_img > 1024] = clamp_max # Clamp to maximum value

    resc_img = (resc_img - clamp_min) / (clamp_max - clamp_min) # rescale to range from 0 to 1

    

    if after_open: resc_img = after_open(resc_img)

    

    resc_img = torch.from_numpy(resc_img.astype(np.float32, copy=False))

    

    x = resc_img.view(1,*resc_img.shape)

    

    if expand: x = x.expand(3,*x.shape[-2:])

    # x.shape[-2:] is needed because not everything is of size 512x512!

    # expand is memory efficient: https://stackoverflow.com/questions/44593141/stacking-copies-of-an-array-a-torch-tensor-efficiently

                

    return cls(x)
def show_dicom_image(img:Image, ax:plt.Axes=None, figsize:tuple=(3,3), hide_axis:bool=True, cmap:str=cmap,

                alpha:float=None, **kwargs)->plt.Axes:

    "Display `DicomImage` in the notebook."

    if ax is None: fig,ax = plt.subplots(figsize=figsize)

    ax.imshow(image2np(img.data.data[0].unsqueeze(0)), # We show only one channel!

              cmap=cmap, alpha=alpha, **kwargs)

    if hide_axis: ax.axis('off')

    return ax
class DicomImageList(ImageList):

    def __init__(self, *args, after_open:Callable=None, **kwargs):

        super().__init__(*args, **kwargs)

        self.after_open = after_open

        self.c,self.sizes = 1,{}

        

    def open(self, fn):

        "Open image in `fn`, subclass and overwrite for custom behavior."

        return open_dicom_image(fn, after_open=self.after_open)

    

    # based on https://github.com/radekosmulski/rsna-intracranial/blob/master/03_train_basic_model.ipynb

    def label_from_df(self, cols:IntsOrStrs=1, label_cls:Callable=None, **kwargs):

        "Label `self.items` from the values in `cols` in `self.inner_df`."

        self.inner_df.labels.fillna('', inplace=True)

        labels = self.inner_df.iloc[:,df_names_to_idx(cols, self.inner_df)]

        assert labels.isna().sum().sum() == 0, f"You have NaN values in column(s) {cols} of your dataframe, please fix it."

        if is_listy(cols) and len(cols) > 1 and (label_cls is None or label_cls == MultiCategoryList):

            new_kwargs,label_cls = dict(one_hot=True, classes= cols),MultiCategoryList

            kwargs = {**new_kwargs, **kwargs}

        return self._label_from_list(_maybe_squeeze(labels), label_cls=label_cls, **kwargs)

    

    def show_xys(self, xs, ys, imgsize:int=4, figsize:Optional[Tuple[int,int]]=None, **kwargs):

        "Show the `xs` (inputs) and `ys` (targets) on a figure of `figsize`."

        rows = int(np.ceil(math.sqrt(len(xs))))

        axs = subplots(rows, rows, imgsize=imgsize, figsize=figsize)

        for x,y,ax in zip(xs, ys, axs.flatten()): x.show(ax=ax, y=y, **kwargs)

        for ax in axs.flatten()[len(xs):]: ax.axis('off')

        plt.tight_layout()

        

    def reconstruct(self, t): return DicomImage(t)
fn = '../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ID_52c9913b1.dcm'
img = open_dicom_image(fn)
img.data.shape
img
# see if resize works properly

img.resize(128)
# check min & max

img.data.min(), img.data.max()
# set image and batch size

sz, bs = 256, 32
data = (DicomImageList.from_df(df_train,

                               path='../input/rsna-intracranial-hemorrhage-detection',

                               folder='stage_1_train_images')

        .split_none()

        .label_from_df(cols=-1, label_delim=' ')

        .transform(size=sz)

        .databunch(bs=bs))
# these dummy statistics are only calculated on three random batches

stats_dicom = [torch.tensor(0.2192), torch.tensor(0.2775)]
data.normalize(stats_dicom)
# verify that we have our 6 classes

assert data.c == 6
data.show_batch()
data.show_batch()
data.show_batch()