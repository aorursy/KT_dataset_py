from fastai2.vision.all import *
from utils import *
import matplotlib.pyplot as plt
img_dir = '/content/Simpson/simpsons_dataset'
img_dir = '/content/Simpson/simpsons_dataset'
path=Path(img_dir)
img = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    #get_items=get_image_files,#using_attr(get_image_files(recurse=True)),
    get_items=partial(get_image_files,recurse=true),
    #splitter=RandomSubsetSplitter(0.2, 0.1),#RandomSplitter(valid_pct=0.2, seed=42),
    splitter=RandomSplitter(seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))
#img.summary(path)
img = img.dataloaders(path)
img.valid.show_batch(max_n=4, nrows=2)
import warnings
warnings.filterwarnings('ignore')
learn = cnn_learner(img, resnet34, metrics=accuracy, opt_func=Adam, model_dir="/tmp/model/")
learn.fine_tune(4)
#learn.summary
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(12,12), dpi=80)
learn.unfreeze()
learn.lr_find()
learn.fit_one_cycle(6, lr_max=1e-5)
from fastai2.callback.fp16 import *
learn = cnn_learner(img, resnet50, metrics=accuracy).to_fp16()
learn.fine_tune(10, freeze_epochs=3)