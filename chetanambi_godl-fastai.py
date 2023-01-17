from fastai.vision import *

from fastai.callbacks import *
from pathlib import Path
bs = 64
path = Path("../input/gameofdl/")

train_path = path/'train.csv'

test_path = path/'test.csv'

train_path, test_path
np.random.seed(42)
data = ImageDataBunch.from_csv(path=path, 

                               folder='train', 

                               csv_labels='train.csv', 

                               test='test', 

                               ds_tfms=get_transforms(), 

                               size=224, 

                               bs=bs).normalize(imagenet_stats)

data.path = pathlib.Path('.')
get_transforms()
#data.show_batch()
data.show_batch(rows=3, figsize=(7,6))
print(data.classes)

len(data.classes),data.c
learn = cnn_learner(data, models.resnet50, metrics=[error_rate, accuracy])
learn.fit_one_cycle(4)
learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused(min_val=10)
learn.unfreeze()
learn.fit_one_cycle(1)
learn.load('stage-1');
learn.lr_find()
learn.recorder.plot()
learn.load('stage-1');

learn.unfreeze()

learn.fit_one_cycle(2)
learn50 = cnn_learner(data, models.resnet50, metrics=error_rate)
learn50.lr_find()

learn50.recorder.plot()
learn50.fit_one_cycle(8)
learn50.save('stage-1-50')
learn50.load('stage-1-50')

learn50.unfreeze()

learn50.fit_one_cycle(5, max_lr=slice(1e-4,1e-2))
learn50.save('stage-2-50')
sizes = [32, 64, 128, 224]
def get_data(sz, bs):

    data = ImageDataBunch.from_csv(path=path, 

                                   folder='train', 

                                   csv_labels='train.csv', 

                                   test='test', 

                                   ds_tfms=get_transforms(), 

                                   size=sz, 

                                   bs=bs).normalize(imagenet_stats)

    data.path = pathlib.Path('.')

    return data
learn50 = cnn_learner(get_data(8, int(2048/8)), 

                      models.resnet50, 

                      metrics=error_rate)

#learn50.save('res50_0')

learn50.save('res50_8')



learn50 = cnn_learner(get_data(16, int(2048/16)), 

                      models.resnet50, 

                      metrics=error_rate).load('res50_8')

learn50.save('res50_16')



learn50 = cnn_learner(get_data(24, int(2048/24)), 

                      models.resnet50, 

                      metrics=error_rate).load('res50_16')

learn50.save('res50_24')



learn50 = cnn_learner(get_data(32, int(2048/32)), 

                      models.resnet50, 

                      metrics=error_rate).load('res50_24')

learn50.save('res50_32')



learn50 = cnn_learner(get_data(64, int(2048/64)), 

                      models.resnet50, 

                      metrics=error_rate).load('res50_32')

learn50.save('res50_64')



learn50 = cnn_learner(get_data(128, int(2048/128)), 

                      models.resnet50, 

                      metrics=error_rate).load('res50_64')

learn50.save('res50_128')



learn50 = cnn_learner(get_data(224, int(2048/224)), 

                      models.resnet50, 

                      metrics=error_rate).load('res50_128')

learn50.save('res50_224')
def train_model(sz, i):

    learn50 = cnn_learner(get_data(sz, int(2048/sz)), 

                          models.resnet50, 

                          metrics=[error_rate, accuracy]).load('res50_'+str(sz-8))

    learn50.fit_one_cycle(6*i)

    learn50.lr_find()

    learn50.recorder.plot()

    learn50.unfreeze()

    learn50.fit_one_cycle(2*i)

    learn50.save('res50_'+str(sz))
train_model(8, 1)
train_model(16, 2)
train_model(24, 3)
train_model(32, 4)
sz = 64; i = 5

learn50 = cnn_learner(get_data(sz, int(2048/sz)), 

                      models.resnet50, 

                      metrics=[error_rate, accuracy]).load('res50_32')

learn50.fit_one_cycle(6*i)

learn50.lr_find()

learn50.recorder.plot()

learn50.unfreeze()

learn50.fit_one_cycle(2*i)

learn50.save('res50_'+str(sz))
sz = 128; i = 6

learn50 = cnn_learner(get_data(sz, bs), 

                      models.resnet50, 

                      metrics=[error_rate, accuracy], 

                      callbacks=[SaveModelCallback(learn50, every='improvement', monitor='accuracy', name='best_128')]).load('res50_64')

learn50.fit_one_cycle(6*i)
learn50.unfreeze()

learn50.fit_one_cycle(2*i)

learn50.save('res50_'+str(sz))
sz = 224; i = 7

learn50 = cnn_learner(get_data(sz, bs),  

                      models.resnet50, 

                      metrics=[error_rate, accuracy], 

                      callbacks=[SaveModelCallback(learn50, every='improvement', monitor='accuracy', name='best_224')]).load('stage-2-50')

                      #callbacks=[SaveModelCallback(learn50, every='improvement', monitor='accuracy', name='best_224')]).load('res50_128')  



learn50.fit_one_cycle(6*i)
sz = 224; i = 7

learn50.load('best_224')

learn50.fit_one_cycle(2*i)

learn50.save('res50_'+str(sz))
learn50 = cnn_learner(get_data(224, 64), 

                      models.resnet50, 

                      metrics=[error_rate, accuracy], 

                      callbacks=[SaveModelCallback(learn50, every='improvement', monitor='accuracy', name='best_224_mixup')]).load('res50_224').mixup()
learn50.fit(8)
learn50.save('mixup_8')

learn50.fit(5)

learn50.save('mixup_5')
#learn50.unfreeze()

#learn50.fit_one_cycle(5)
import numpy as np

import pandas as pd
learn50.load('mixup_5')

log_preds, test_labels = learn50.get_preds(ds_type=DatasetType.Test)
preds = np.argmax(log_preds, 1)

preds_classes = [data.classes[i] for i in preds]
a = np.array(preds)

data.test_ds.x[0]
test_df = pd.DataFrame({ 'image': os.listdir('../input/gameofdl/test/test_images/'), 'category': preds_classes})

test_df.head()
#test_df.sort_values(by='image').reset_index(drop=True)
# import the modules we'll need

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link(df, title = "Download CSV file", filename = "submission.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



test_df['category'] = pd.DataFrame(data=preds_classes)



# create a link to download the dataframe

create_download_link(test_df)