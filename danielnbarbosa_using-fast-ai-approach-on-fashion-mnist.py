# automatic reloading and inline plotting
%reload_ext autoreload
%autoreload 2
%matplotlib inline
# import libraries
from fastai.imports import *
from fastai.torch_imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# make sure CUDA is available and enabled
print(torch.cuda.is_available(), torch.backends.cudnn.enabled)
# setup paths
# model needs in be stored in /tmp or else kaggle complains about creating too many files
PATH = "../input"
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"
# all the necessary data is in the CSV files
!ls -l ../input
# create panda dataframes from CSVs
train_df = pd.read_csv(f'{PATH}/fashion-mnist_train.csv')
test_df = pd.read_csv(f'{PATH}/fashion-mnist_test.csv')
# examine dataframe structure
train_df.head()
# test data is also labeled
test_df.head()
print('train:', train_df.shape)
print('test: ', test_df.shape)
# split out cross-validation dataframe
valid_df = train_df.sample(frac=0.10, random_state=42)
valid_df.shape
# drop cross-validation data from train dataframe
train_df = train_df.drop(valid_df.index)
train_df.shape
# create inputs
train_X = train_df.drop('label', axis=1)
valid_X = valid_df.drop('label', axis=1)
test_X = test_df.drop('label', axis=1)
# create labels 
train_Y = train_df['label']
valid_Y = valid_df['label']
test_Y = test_df['label']
# sanity check
print('train:', train_X.shape, train_Y.shape)
print('valid:', valid_X.shape, valid_Y.shape)
print('test: ', test_X.shape, test_Y.shape)
# convert to ndarray
train_X = np.array(train_X)
valid_X = np.array(valid_X)
test_X = np.array(test_X)
train_Y = np.array(train_Y)
valid_Y = np.array(valid_Y)
test_Y = np.array(test_Y)
# reshape into single channel ndarray
train_X = train_X.reshape(-1, 28, 28)
valid_X = valid_X.reshape(-1, 28, 28)
test_X = test_X.reshape(-1, 28, 28)
print('train:', train_X.shape)
print('valid:', valid_X.shape)
print('test:', test_X.shape)
# reshape into 3 channel ndarray (as expected by model)
train_X = np.stack((train_X, ) * 3, axis=-1)
valid_X = np.stack((valid_X, ) * 3, axis=-1)
test_X = np.stack((test_X, ) * 3, axis=-1)
print('train:', train_X.shape)
print('valid:', valid_X.shape)
print('test:', test_X.shape)
# sanity check
print('train:', train_X.shape, train_Y.shape)
print('valid:', valid_X.shape, valid_Y.shape)
print('test:', test_X.shape, test_Y.shape)
# create label dictionary
label_dict = {
    0: 't-shirt/top',
    1: 'trouser',
    2: 'pullover',
    3: 'dress',
    4: 'coat',
    5: 'sandal',
    6: 'shirt',
    7: 'sneaker',
    8: 'bag',
    9: 'ankle boot'    
}
# visualize training images with labels
images = [train_X[i] for i in range(36)]
labels = [label_dict.get(train_Y[i]) for i in range(36)]
plots(images, figsize=(12,12), rows=6, titles=labels)
# define architecture, image size, batch size
# resnet34 is one of the smaller, modern, pre-trained models, so it will run fairly quickly
# for better accuracy try a bigger model
arch = resnet34
sz = 28
bs = 64
def get_data(sz):
    tfms = tfms_from_model(arch, sz)
    return ImageClassifierData.from_arrays(path=PATH, 
                                       trn=(train_X, train_Y),
                                       val=(valid_X, valid_Y),
                                       bs=bs,
                                       tfms=tfms,
                                       classes=train_Y,
                                       test=test_X)
data = get_data(sz)
# create learner with precompute enabled
learn = ConvLearner.pretrained(arch, data, precompute=True, tmp_name=TMP_PATH, models_name=MODEL_PATH)
# summarize model structure
# the fully connected layers at the end are the only ones that are trained when the model is frozen
learn.summary
# find optimal learning rate
lrf = learn.lr_find()
learn.sched.plot()
lr = 0.01
# first train only the randomly initialized layers added to the end of the model
learn.fit(lr, 4)
learn.sched.plot_loss()
# disable precompute and unfreeze layers
learn.precompute=False
learn.unfreeze()
# define differential learning rates
lrs = np.array([lr/9, lr/3, lr])
# retrain full model
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
learn.sched.plot_loss()
# save full model
learn.save("28_all")
# get accuracy for validation set
log_preds, y = learn.TTA(n_aug=1)
probs = np.mean(np.exp(log_preds), 0)
accuracy_np(probs, y)
# print classification report
preds = np.argmax(probs, axis=1)
target_names = [label_dict.get(i) for i in range(10)]
print(classification_report(y, preds, target_names=target_names))
# plot confusion matrix
cm = confusion_matrix(y, preds)
classes = np.unique(train_Y)
plot_confusion_matrix(cm, classes)
# count incorrect predictions
idxs = np.where(preds != y)[0]
len(idxs)
# visualize incorrect predictions
# title is (prediction, label)
images = [test_X[i] for i in idxs[0:40]]
titles = [(preds[i], test_Y[i]) for i in idxs[0:40]]
plots(images, rows=4, titles=titles)
# get accuracy for test set
log_preds_test, y_test = learn.TTA(n_aug=1, is_test=True)
probs_test = np.mean(np.exp(log_preds_test), 0)
accuracy_np(probs_test, test_Y)
# print classification report
preds_test = np.argmax(probs_test, axis=1)
target_names = [label_dict.get(i) for i in range(10)]
print(classification_report(test_Y, preds_test, target_names=target_names))
# plot confusion matrix
preds_test = np.argmax(probs_test, axis=1)
cm_test = confusion_matrix(test_Y, preds_test)
plot_confusion_matrix(cm_test, classes)
# count incorrect predictions
idxs = np.where(preds_test != test_Y)[0]
len(idxs)
