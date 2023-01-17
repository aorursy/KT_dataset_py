!pip install fastai --upgrade -q
from fastai.vision.all import *

from fastai.callback.cutmix import *
path = untar_data(URLs.PETS)
pets = DataBlock(blocks = (ImageBlock, CategoryBlock),
                 get_items=get_image_files, 
                 splitter=RandomSplitter(seed=42),
                 get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'),
                 item_tfms=Resize(460),
                 batch_tfms=aug_transforms(size=224, min_scale=0.75))
dls = pets.dataloaders(path/"images")
# import torchvision.models as models

# wrn = models.wide_resnet50_2(pretrained=True)
# learn = cnn_learner(dls, resnet34, metrics=error_rate, cbs=MixUp)
# learn.fine_tune(1)
# learn = cnn_learner(dls, resnet34, metrics=error_rate, cbs=CutMix)
# learn.fine_tune(1)
# learn = cnn_learner(dls, resnet18, metrics=error_rate)
# learn.fine_tune(1)
# learn = cnn_learner(dls, resnet34, metrics=error_rate)
# learn.fine_tune(1)
# learn = cnn_learner(dls, resnet50, metrics=error_rate)
# learn.fine_tune(1)
# learn = cnn_learner(dls, resnet101, metrics=error_rate)
# learn.fine_tune(1)
# learn = cnn_learner(dls, resnet152, metrics=error_rate)
# learn.fine_tune(1)
# learn = cnn_learner(dls, resnet50, metrics=error_rate)
# learn.fine_tune(15, freeze_epochs=10)
# learn = cnn_learner(dls, resnet50, metrics=error_rate, cbs=CutMix)
# learn.fine_tune(15, freeze_epochs=10)
# learn = cnn_learner(dls, resnet50, metrics=error_rate, cbs=CutMix)
# learn.fine_tune(15, freeze_epochs=10)
learn = cnn_learner(dls, resnet50, metrics=error_rate, cbs=CutMix)
learn.lr_find()
learn.fit_one_cycle(5,lr_max=3e-3)
learn.unfreeze()
learn.lr_find()
learn.fit_one_cycle(20,lr_max=slice(1e-6,1e-4))
learn = cnn_learner(dls, resnet50, metrics=error_rate, cbs=CutMix)
learn.fit_one_cycle(10,lr_max=3e-3)
learn.unfreeze()
learn.lr_find()
learn.fit_one_cycle(20,lr_max=slice(1e-6,1e-4))
