
%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *
# 1) manually download the model from:
#   * https://download.pytorch.org/models/resnet34-333f7ec4.pth" to /tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth
# 2) use "+ Add Data" on the kaggle's data panel on the right to make the model available
# 3) make the temp folder (that pytorch wants to use) if it doesn't exist already
Path('/tmp/.cache/torch/checkpoints').mkdir(parents=True, exist_ok=True)
# 4) copy the model from input to the temp folder
!cp /kaggle/input/pytorch-models/resnet34-333f7ec4.pth /tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth
# 5) then make sure we set path='/kaggle/working' when creating the learner - otherwise it'll use the dir of the model (which is read-only)
output_path = '/kaggle/working'
def get_mask_fn(x): return x.parents[0] / (x.stem + '_mask' + x.suffix)
# Designating the path to the data
path = Path('/kaggle/input/lgg-mri-segmentation/kaggle_3m')
# Designating the path to two images we are going to inspect.
image_temp_file = path/'TCGA_CS_6666_20011109/TCGA_CS_6666_20011109_19.tif'
mask_temp_file = get_mask_fn(image_temp_file)
# Show an example of an image with a tumor
image = open_image(image_temp_file)
image.show(figsize = (5,5))
#Show the mask for that image
mask = open_image(mask_temp_file, div = True)
mask.show(figsize = (5,5))
mask_size = np.array(mask.shape)
image_size = np.array(mask.shape[1:])
mask_size
# Validation set must come from patients that the model has not seen before. 
# Thus, I am specifying certain patients for the validation set.
validataion_folders = [
        'TCGA_HT_7694_19950404', 'TCGA_DU_5874_19950510', 'TCGA_DU_7013_19860523',
        'TCGA_HT_8113_19930809', 'TCGA_DU_6399_19830416', 'TCGA_HT_7684_19950816',
        'TCGA_CS_5395_19981004', 'TCGA_FG_6688_20020215', 'TCGA_DU_8165_19970205',
        'TCGA_DU_7019_19940908', 'TCGA_HT_7855_19951020', 'TCGA_DU_A5TT_19980318',
        'TCGA_DU_7300_19910814', 'TCGA_DU_5871_19941206', 'TCGA_DU_5855_19951217']

codes = ['t','f']
# This comes from ilikemath's "Ultrasound Nerve Segmentation with fastai" notebook cited below.
class SegmentationLabelListWithDiv(SegmentationLabelList):
    def open(self, fn): return open_mask(fn, div=True)
class SegmentationItemListWithDiv(SegmentationItemList):
    _label_cls = SegmentationLabelListWithDiv
src = (SegmentationItemListWithDiv
       # get each the data from the folders in this path
       .from_folder(path)
        # the inputs of the data are all files whose name does not end with '_mask.tif' (we are extracting all the raw images)
       .filter_by_func(lambda x: not x.name.endswith('_mask.tif'))
       # designate the validation set as the folders with names listed in validation_folders
       .split_by_valid_func(lambda x: x.parts[-2] in validataion_folders)
       # designate the labels of each raw image as their corresponding masks
       .label_from_func(get_mask_fn, classes = codes))
# We are working with MRI images which are usually similar in orientation, so the only transform I am applying is a flip
# horizontally across the y-axis with a probability of .5 since I figure the brain is pretty symmetrical.
tfms = get_transforms(max_rotate = 0, p_affine = 0, p_lighting = 0)

data = (src.transform(get_transforms(), size = image_size)
# the batch size is generally dependent of GPU capacity, but 16 is usually a decent number for full-sized images on Kaggle's GPU
        .databunch(bs=16)
        .normalize(imagenet_stats))
data.show_batch(figsize = (7,8))
# wd stands for weight decay- a measure to reduce overfitting
wd = 1e-2
learn = unet_learner(data, models.resnet34, metrics = dice, wd = wd, path = output_path)
learn.lr_find()
learn.recorder.plot()
learn.save('pre-trained')
lr = 1e-3
learn.fit_one_cycle(10, lr)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.load('stage-1')
# pct_start means we are applying the cyclic learning rate on only 80% of the epochs. In other words, the feature of 
# fit_one_cycle to increase the learning rate then decrease the learning rate is applied on the epoch with a probability
# of .8.
learn.fit_one_cycle(7, slice(lr/800, lr/8), pct_start = .8)
learn.save('stage-2')
# let's inspect the results now
learn.show_results(rows = 3, figsize = (8,8))
