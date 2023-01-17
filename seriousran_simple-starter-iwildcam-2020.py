%reload_ext autoreload

%autoreload 2

%matplotlib inline
import pandas as pd

import numpy as np

import json

import PIL.Image, PIL.ImageFile



PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True



from fastai import *

from fastai.vision import *

from fastai.utils.mem import *
path = Path('/kaggle/input/iwildcam-2020-fgvc7')



debug =1

if debug:

    train_pct=0.04

else:

    train_pct=0.5

bs=32
!ls /kaggle/input/iwildcam-2020-fgvc7
with open(path/'iwildcam2020_train_annotations.json') as f:

    train_data = json.load(f)

    

with open(path/'iwildcam2020_test_information.json') as f:

    test_data = json.load(f)
train_data.keys()
print( '#train_data')

print()

for key in train_data.keys():

    print( 'length of', key, ':', len(train_data[key]) )

    if key != 'info':

        print( 'example:', train_data[key][0])

    else:

        print(train_data[key])

    print()
print( '#test_data')

print()

for key in test_data.keys():

    print( 'length of', key, ':', len(test_data[key]) )

    if key != 'info':

        print( 'example:', test_data[key][0])

    else:

        print(test_data[key])

    print()
df_train = pd.DataFrame.from_records(train_data['annotations'])

df_train
df_train = pd.DataFrame.from_records(train_data['annotations'])

df_train
#df_image[df_image['id'] == '896c1198-21bc-11ea-a13a-137349068a90']

#df_image[df_image['id'] == '8792549a-21bc-11ea-a13a-137349068a90']

#df_image[df_image['id'] == '87022118-21bc-11ea-a13a-137349068a90']



#df_image[df_image['seq_id'] == '98a295ba-21bc-11ea-a13a-137349068a90']

#df_image[df_image['location'] == 537]['id'].values
df_image = pd.DataFrame.from_records(train_data['images'])



indices = []

#indices.append( df_train[ df_train['image_id'] == '896c1198-21bc-11ea-a13a-137349068a90' ].index )

#indices.append( df_train[ df_train['image_id'] == '8792549a-21bc-11ea-a13a-137349068a90' ].index )

for _id in df_image[df_image['location'] == 537]['id'].values:

    indices.append( df_train[ df_train['image_id'] == _id ].index )



for the_index in indices:

    df_train = df_train.drop(df_train.index[the_index])
df_train[df_train['count']>1]
df_test = pd.DataFrame.from_records(test_data['images'])

df_test
df_test['frame_num'].value_counts()
df_test = df_test.rename(columns={"id": "image_id"})
train, test = [ImageList.from_df(df, path=path, cols='image_id', folder=folder, suffix='.jpg') 

               for df, folder in zip([df_train, df_test], ['train', 'test'])]

data = (train.split_by_rand_pct(0.2, seed=2020)

        .label_from_df(cols='category_id')

        .add_test(test)

        .transform(get_transforms(), size=32)

        .databunch(path=Path('.'), bs=bs).normalize())
if debug:

    src= train.split_subsets(train_size=train_pct, valid_size= train_pct*2)

#     test=test[:1000]

else:

    src= train.split_subsets(train_size=train_pct, valid_size=0.2, seed=2)

#     src= train.split_by_rand_pct(0.2, seed=2)



print(src)

    

def get_data(size, bs, padding_mode='reflection'):

    return (src.label_from_df(cols='category_id')

           .add_test(test)

           .transform(tfms, size=size, padding_mode=padding_mode)

           .databunch(bs=bs).normalize(imagenet_stats))    
tfms = get_transforms(max_rotate=20, max_zoom=1.3, max_lighting=0.4, max_warp=0.4,

                      p_affine=1., p_lighting=1.)



data = get_data(224, bs, 'zeros')
def _plot(i,j,ax):

    x,y = data.train_ds[3]

    x.show(ax, y=y)



plot_multi(_plot, 3, 3, figsize=(12,12))
!ls /kaggle/working/
gc.collect()

wd=1e-2

#learn = cnn_learner(data, models.densenet121, metrics=error_rate, bn_final=True, wd=wd )

learn = cnn_learner(data, models.resnet34, metrics=error_rate, bn_final=True, wd=wd )

learn.model_dir= '/kaggle/working/'
data = get_data(352,bs)

learn.data = data

learn.fit_one_cycle(6, max_lr=slice(1e-6,1e-4))

learn.save('352')
learn.unfreeze()
lr = 1e-3

learn.fit_one_cycle(4, slice(lr/100, lr))
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
# %%time

# interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
test_preds = learn.get_preds(DatasetType.Test)

df_test['Category'] = test_preds[0].argmax(dim=1)
df_test.head()
df_test = df_test.rename(columns={"image_id": "Id"})

df_test = df_test.drop(['seq_num_frames', 'location', 'datetime', 'frame_num', 'seq_id', 'width', 'height', 'file_name'], axis=1)
submission = pd.read_csv('/kaggle/input/iwildcam-2020-fgvc7/sample_submission.csv')

submission = submission.drop(['Category'], axis=1)

submission = submission.merge(df_test, on='Id')

submission.to_csv('submission.csv', index=False)