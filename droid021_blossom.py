%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate
bs = 64



!ls '../input/flower_data/flower_data/'
path = Path('../input/flower_data/flower_data/')
path.ls()
img_size = 224

data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(), 

                                  valid='valid', size=img_size, bs = bs) .normalize(imagenet_stats)
import json



with open('../input/cat_to_name.json', 'r') as f:

    cat_to_name = json.load(f)
class_names = data.classes
for i in range(0,len(class_names)):

    class_names[i] = cat_to_name.get(class_names[i])

class_names[20]
data.classes
data.show_batch(rows=3, figsize=(8,7))
learn = cnn_learner(data, models.resnet34, metrics=accuracy, model_dir='/kaggle/working')
learn.model
learn.fit_one_cycle(5)
learn.save('stage-1')
## Results



interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused(min_val=2)
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5)
learn.save('stage-2');
learn.load('stage-2');
interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=2)
interp.plot_confusion_matrix(figsize=(12,12),cmap='viridis', dpi=60)
learn.recorder.plot_losses()
learn.export('/kaggle/working/export.pkl')
newpath = '../input/hackathon-blossom-flower-classification/test set/'



test = ImageDataBunch.from_folder(newpath, ds_tfms=get_transforms(), 

                                  valid='valid', size=img_size, bs = bs) .normalize(imagenet_stats)

len(test)
learn = load_learner('/kaggle/working', test=test)



preds, _ = learn.get_preds(ds_type=DatasetType.Test)
thresh = 0.2

labelled_preds = [' '.join([learn.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]
labelled_preds[:5]
fnames = [f.name[:-4] for f in learn.data.test_ds.items]
df = pd.DataFrame({'image_name':fnames, 'tags':labelled_preds}, columns=['image_name', 'tags'])

df.head(10)