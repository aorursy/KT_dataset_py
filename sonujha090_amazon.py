!pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html -q
!pip install --upgrade kornia -q
!pip install allennlp==1.1.0.rc4 -q
!pip install --upgrade fastai -q
import torch
print(torch.__version__)
print(torch.cuda.is_available())

import fastai
print(fastai.__version__)

from fastai.vision.all import *
path = Path('../input/planets-dataset/planet/planet/')
Path.BASE_PATH = path
path.ls()
train = pd.read_csv(path/'train_classes.csv')
train.head()
(path/'train-jpg').ls()
get_x = ColReader(0, pref=f'{path}/train-jpg/', suff='.jpg')
get_y = ColReader(1, label_delim=' ')

planet = DataBlock(
    blocks=(ImageBlock, MultiCategoryBlock),
    get_x = get_x, 
    get_y = get_y,
    splitter=RandomSplitter(),
    item_tfms=Resize(224))
dls = planet.dataloaders(train)
dls.show_batch()
xb, yb = dls.one_batch()
xb.shape, yb.shape
get_x = lambda x: path/'train-jpg'/f'{x[0]}.jpg'
get_y = lambda x: x[1].split(' ')

planet = DataBlock(
    blocks=(ImageBlock, MultiCategoryBlock),
    get_x = get_x,
    get_y = get_y,
    splitter=RandomSplitter(),
    item_tfms=Resize(224))
dls = planet.dataloaders(train)
dls.show_batch()
xb, yb = dls.one_batch()
xb.shape, yb.shape
def planet_item(x): return (f'{path}/train-jpg/'+x.image_name+'.jpg', x.tags.str.split())
planet = DataBlock.from_columns(
    blocks=(ImageBlock, MultiCategoryBlock),
    get_items=planet_item,
    splitter=RandomSplitter(),
    item_tfms=Resize(228))
dls = planet.dataloaders(train)
dls.show_batch()
xb, yb = dls.one_batch()
xb.shape, yb.shape
learn = cnn_learner(dls, resnet34, pretrained=True, metrics=[accuracy_multi])
class BCEWithLogitsLossFlat(BaseLoss):
    "Same as `nn.CrossEntropyLoss`, but flattens input and target."
    def __init__(self, *args, axis=-1, floatify=True, thresh=0.5, **kwargs):
        super().__init__(nn.BCEWithLogitsLoss, *args, axis=axis, floatify=floatify, is_2d=False, **kwargs)
        self.thresh = thresh

    def decodes(self, x):    return x>self.thresh
    def activation(self, x): return torch.sigmoid(x)
learn.loss_func = BCEWithLogitsLossFlat()
learn.lr_find()
lr = 1e-2
learn = learn.to_fp16()
learn.fit_one_cycle(5, slice(lr))
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.fit_one_cycle(5, slice(1e-6, lr/5))