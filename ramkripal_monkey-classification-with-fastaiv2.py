# install the fastai2

!pip install fastai2
from fastai2.basics import *

from fastai2.vision.all import *

from fastai2.callback.all import *
path = Path("../input/10-monkey-species/")

path.ls()
monkey = monkey = DataBlock(

    blocks = (ImageBlock, CategoryBlock),

    get_items = get_image_files,

    splitter=GrandparentSplitter(train_name='training', valid_name='validation'),

    get_y = parent_label,

    item_tfms = Resize(300),

    batch_tfms = aug_transforms(size=224), 

                  

)
dbunch = monkey.dataloaders(source=path, bs=64)
dbunch.show_batch()
learner = cnn_learner(dbunch, resnet34, metrics=accuracy)
learner.fit_one_cycle(6)
interp = ClassificationInterpretation.from_learner(learner)
interp.plot_confusion_matrix(figsize=(8,8))
learner.unfreeze()
learner.lr_find()
learner.fit_one_cycle(4, slice(1e-5, 3e-4))
interp = ClassificationInterpretation.from_learner(learner)
interp.plot_confusion_matrix(figsize=(8,8))
learner.show_results(max_n=16, figsize=(15,9))

plt.tight_layout()
interp.plot_top_losses(k=4, figsize=(12, 8))