import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from fastai.vision import *
path = '/kaggle/input/'

dest = path
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
data.classes
data.show_batch(rows=3, figsize=(7,8))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
learn = cnn_learner(data, models.resnet34, metrics=[accuracy, error_rate])
learn.model_dir='/kaggle/working/'
learn.fit_one_cycle(4)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
callbacks = [callbacks.SaveModelCallback(learn, every='improvement', mode='min', monitor='error_rate', name='/kaggle/working/best_model')]
learn.fit_one_cycle(30, max_lr=slice(5e-5,5e-4), callbacks=callbacks)
learn.recorder.plot_losses()
learn.save('stage-2')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(20,20))
#from fastai.vision import *
#path = '/kaggle/input/'

#dest = path
#np.random.seed(42)

#data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

#        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
#data.show_batch(rows=3, figsize=(7,8))
#learn = cnn_learner(data, models.resnet34, metrics=[accuracy, error_rate])
#import os

#for dirname, _, filenames in os.walk('/kaggle/input/'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))
#learn.load('/kaggle/input/model-temp/stage-2')
from sklearn.metrics import roc_curve, auc
preds,y,loss = learn.get_preds(with_loss=True)

# get accuracy

acc = accuracy(preds, y)

print('The accuracy is {0} %.'.format(acc))
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(title='Confusion matrix', figsize=(20,20))
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
data.classes
learn.export('/kaggle/working/birdclassifier_new3.pkl')