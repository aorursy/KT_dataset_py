!pip install fastai --upgrade
from fastai.vision.all import *
path = Path('../input/cell-images-for-detecting-malaria/cell_images')
img = get_image_files(path)
len(img)
data = DataBlock(blocks=(ImageBlock, CategoryBlock),
                get_items = get_image_files,
                get_y = parent_label,
                splitter = RandomSplitter(),
                item_tfms = Resize(224),
                batch_tfms = Normalize.from_stats(*imagenet_stats))
dls = data.dataloaders(path)
dls.show_batch(max_n=6)
learn = cnn_learner(dls, resnet50, metrics=accuracy)
learn.fit_one_cycle(4)
learn.show_results(max_n=6)

interp = Interpretation.from_learner(learn)








