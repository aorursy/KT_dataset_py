import warnings

warnings.filterwarnings("ignore")



get_ipython().run_line_magic('reload_ext', 'autoreload')

get_ipython().run_line_magic('autoreload', '2')

get_ipython().run_line_magic('matplotlib', 'inline')



import os

from fastai import *

from fastai.vision import *
path = Path('../input/parkinsons-drawings/')



bs = 64

size = 224

num_workers = 0



tfms = get_transforms()                               #Do standard data augmentation

data = (ImageList.from_folder(path)               #Get data from path

        .split_by_rand_pct()                        #Randomly separate 20% of data for validation set

        .label_from_folder()                          #Label based on dir names

        .transform(tfms, size=size)                   #Pass in data augmentation

        .databunch(bs=bs, num_workers=num_workers)    #Create ImageDataBunch

        .normalize(imagenet_stats))                   #Normalize using imagenet stats





data.show_batch(rows=3, figsize=(7,6))



data.classes
learn = create_cnn(data, models.resnet50, metrics=accuracy, model_dir='/kaggle/working')

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(10, max_lr=slice(1e-05, 1e-02))

learn.save('stage-2')
learn.export('/kaggle/working/export.pkl')
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(8,8), dpi=60)
preds,y, loss = learn.get_preds(with_loss=True)

# get accuracy

acc = accuracy(preds, y)

print('The accuracy is {0} %.'.format(acc))
from sklearn.metrics import roc_curve, auc

# probs from log preds

probs = np.exp(preds[:,1])

# Compute ROC curve

fpr, tpr, thresholds = roc_curve(y, probs, pos_label=1)



# Compute ROC area

roc_auc = auc(fpr, tpr)

print('ROC area is {0}'.format(roc_auc))

plt.figure()

plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlim([-0.01, 1.0])

plt.ylim([0.0, 1.01])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")