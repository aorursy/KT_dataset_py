from fastai import *

from fastai.vision import *

from fastai.callbacks.hooks import *



import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import auc,roc_curve



import os

print(os.listdir("../input"))

% matplotlib inline
labels_path='../input/train_labels.csv'

labels = pd.read_csv(labels_path)

labels.head()
print(f'Number of labels {len(labels)}')

sns.countplot(x='label',data=labels)
path=Path('../input/')
tfms=get_transforms(flip_vert=True, max_warp=0, max_rotate=0, max_lighting=0.05)
data=ImageDataBunch.from_csv(path, 

                             csv_labels='train_labels.csv', 

                             folder='train', 

                             suffix='.tif',

                             num_workers=2,

                             ds_tfms=tfms,

                             bs=64,

                             size=72,

                             test=path/'test').normalize()

data.show_batch(rows=3,figsize=(8,10))
learner= create_cnn(data, models.resnet50, metrics=[accuracy], model_dir="/tmp/models/")
learner.lr_find()

learner.recorder.plot()
learner.fit_one_cycle(6, max_lr=(1e-4, 1e-3, 1e-2), wd=(1e-4, 1e-4, 1e-1))
learner.unfreeze()
learner.lr_find()

learner.recorder.plot()
learner.fit_one_cycle(14, slice(1e-5,1e-4,1e-3))
learner.save('stage-1') 
learner.recorder.plot_losses()
learner.recorder.plot_metrics()
conf= ClassificationInterpretation.from_learner(learner)

conf.plot_confusion_matrix(figsize=(10,8))
pred_data= ImageDataBunch.from_csv(path, 

                             csv_labels='train_labels.csv', 

                             folder='train', 

                             suffix='.tif',

                             num_workers=2,

                             ds_tfms=get_transforms(flip_vert=True, max_warp=0),

                             bs=64,

                             test=path/'test').normalize(imagenet_stats)
classes=[0,1]

pred_data.single_from_classes(path, classes)
predictor=create_cnn(pred_data, models.resnet50, metrics=[accuracy], model_dir="/tmp/models/").load('stage-1')
x,y=pred_data.valid_ds[1]

x.show()

pred_data.valid_ds.y[1]
pred_class,pred_idx,outputs = predictor.predict(x)

pred_class
def heatMap(x,y,data, learner, size=(0,96,96,0)):

    """HeatMap"""

    

    # Evaluation mode

    m=learner.model.eval()

    

    # Denormalize the image

    xb,_ = data.one_item(x)

    xb_im = Image(data.denorm(xb)[0])

    xb = xb.cuda()

    

    # hook the activations

    with hook_output(m[0]) as hook_a: 

        with hook_output(m[0], grad=True) as hook_g:

            preds = m(xb)

            preds[0,int(y)].backward()



    # Activations    

    acts=hook_a.stored[0].cpu()

    

    # Avg of the activations

    avg_acts=acts.mean(0)

    

    # Show HeatMap

    _,ax = plt.subplots()

    xb_im.show(ax)

    ax.imshow(avg_acts, alpha=0.6, extent=size,

              interpolation='bilinear', cmap='magma')

    
heatMap(x,y,pred_data,learner)
# Predictions of the validation data

preds_val, y_val=learner.get_preds()
#  ROC curve

fpr, tpr, thresholds = roc_curve(y_val.numpy(), preds_val.numpy()[:,1], pos_label=1)



#  ROC area

pred_score = auc(fpr, tpr)

print(f'ROC area is {pred_score}')
plt.figure()

plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.2f)' % pred_score)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlim([-0.01, 1.0])

plt.ylim([0.0, 1.01])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic')

plt.legend(loc="lower right")
# Predictions on the Test data

preds_test,y_test = learner.TTA(ds_type=DatasetType.Test)

# preds_test, y_test=learner.get_preds(ds_type=DatasetType.Test)
sub=pd.read_csv(f'{path}/sample_submission.csv').set_index('id')

sub.head()
names=np.vectorize(lambda img_name: str(img_name).split('/')[-1][:-4]) 

file_names= names(data.test_ds.items).astype(str)
sub.loc[file_names,'label']=preds_test.numpy()[:,1]

sub.to_csv(f'submission_{pred_score}.csv')
sub.head()