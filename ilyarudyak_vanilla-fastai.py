from fastai.vision import *

import fastai

import pandas as pd
fastai.__version__
Path('../input').ls(), 
path = Path('../input/dog-breed-identification')

path_train = path / 'train'

path_test = path / 'test/test'

path_model = Path('/tmp/model/')
path_train, path_test
path.ls()
len(path_train.ls()), len(path_test.ls())
list(Path('../input/dog-breed-identification').glob('**'))
bs, size = 64, 224
data = ImageDataBunch.from_csv(path=path,

                               folder='train',

                               csv_labels='labels.csv',

                               ds_tfms=get_transforms(),

                               suffix='.jpg',

                               test='test/test',

                               size=size,

                               bs=bs,

                               num_workers=0).normalize(imagenet_stats)
len(data.train_ds), len(data.valid_ds), len(data.test_ds.items)
8178 + 2044
learn = cnn_learner(data, 

                    base_arch=models.resnet50, 

                    metrics=accuracy, 

                    model_dir=path_model)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(8)
learn.save('stage-1-50-224')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-4))
learn.load('stage-1-50-224');
preds, _ = learn.get_preds(DatasetType.Test)
data.test_ds.items[0]
path_test
def get_numeric_part(file_path, path_test):

    low = len(str(path_test)) + 1

    return str(file_path)[low:-4]
get_numeric_part(data.test_ds.items[0], path_test)
df = pd.DataFrame(preds.numpy(), columns=data.classes)

filenames = [get_numeric_part(fp, path_test) for fp in data.test_ds.items]

df.insert(0, "id", filenames)

df_sorted = df.sort_values(by='id')
df_sorted.head()
df_sorted.iloc[0, 1:].sum()
df_sorted.to_csv('submission.csv', index=False)
ls