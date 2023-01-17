## To setup the fastai2 environment



!pip install fastai2
from fastai2.basics import *

from fastai2.vision.all import *

from fastai2.callback.all import *
path = Path("../input/chest-xray-pneumonia/chest_xray")

path.ls()
pneumonia = DataBlock(

    blocks = (ImageBlock, CategoryBlock),

    get_items = get_image_files,

    splitter = GrandparentSplitter(train_name="train", valid_name="test"),

    get_y = parent_label,

    item_tfms = Resize(400),

    batch_tfms = [*aug_transforms(size=224), Normalize.from_stats(*imagenet_stats)]

)
dls = pneumonia.dataloaders(source=path, bs=64)
dls.show_batch()
learner = cnn_learner(dls, resnet50, metrics=accuracy).to_fp16()
learner.fit_one_cycle(4)
learner.show_results()
interp = ClassificationInterpretation.from_learner(learner)
interp.plot_confusion_matrix()
interp.plot_top_losses(k=9, figsize=(15,10))
learner.save('stage-1')

learner.unfreeze()

learner.lr_find()
learner.fit_one_cycle(4, slice(1e-6, 3e-4))
interp = ClassificationInterpretation.from_learner(learner)
interp.plot_confusion_matrix()