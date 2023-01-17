!pip install fastai --upgrade -q
from fastai.vision.all import *
from fastai.callback.fp16 import *
from fastai.vision.widgets import *
pets = DataBlock(blocks = (ImageBlock, CategoryBlock),
                get_items = get_image_files,
                splitter = RandomSplitter(seed = 42),
                get_y = parent_label,
                item_tfms= Resize(460),
                batch_tfms = aug_transforms(size = 224, min_scale = 0.75))
dls = pets.dataloaders("../input/animal-faces/afhq",bs = 32)
dls.show_batch(nrows = 2, ncols = 3)
learn = cnn_learner(dls, resnet50, metrics = error_rate).to_fp16()
learn.fine_tune(6, freeze_epochs = 3)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize = (12,12), dpi = 60)
interp.most_confused(min_val = 2)
cleaner = ImageClassifierCleaner(learn)
cleaner
