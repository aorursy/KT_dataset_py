# verify dataset is there
!ls ../input
%reload_ext autoreload
%autoreload 2
%matplotlib inline
# load fastai libraries
from fastai.imports import *
from fastai.torch_imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
# load additional libraries
import h5py
# verify GPU
print(torch.cuda.is_available(), torch.backends.cudnn.enabled)
def load_dataset(path_to_train, path_to_test):
    train_dataset = h5py.File(path_to_train)
    train_x = np.array(train_dataset['train_set_x'][:])
    train_y = np.array(train_dataset['train_set_y'][:])

    test_dataset = h5py.File(path_to_test)
    test_x = np.array(test_dataset['test_set_x'][:])
    test_y = np.array(test_dataset['test_set_y'][:])

    # y reshaped
    train_y = train_y.reshape((1, train_x.shape[0]))
    test_y = test_y.reshape((1, test_y.shape[0]))

    return train_x, train_y, test_x, test_y
PATH = "../input"
X_train, Y_train, X_test, Y_test = load_dataset(f"{PATH}/train_happy.h5", f"{PATH}/test_happy.h5")
# swap dimensions (Andrew Ng likes them flipped around)
Y_train = Y_train.T.squeeze()
Y_test = Y_test.T.squeeze()
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
# visualize a training example
plt.imshow(X_train[0])
# and it's label
Y_train[0]
# setup architecture
arch = resnet34
sz = 64
bs = 30
def get_data(sz):
    tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
    return ImageClassifierData.from_arrays(path='tmp', 
                                       trn=(X_train, Y_train),
                                       val=(X_test, Y_test),
                                       bs=bs,
                                       classes=Y_train,
                                       tfms=tfms)
data = get_data(sz)
# run learner with precompute enabled
#learn = ConvLearner.pretrained(arch, data, precompute=True)
# find optimal learning rate
#lrf = learn.lr_find()
#learn.sched.plot()
lr = 0.01
# increase batch size
# could increase learning rate as well, but current one works just as well
# learning rate finder doesn't work with larger batch size due to lack of training examples
#bs=30
#data = get_data(sz)
# train
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)
learn.sched.plot_loss()
# add test time augmentation
log_preds,y = learn.TTA()
probs = np.mean(np.exp(log_preds),0)
accuracy_np(probs, y)
preds = np.argmax(probs, axis=1)
probs = probs[:,1]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, preds)
classes = np.unique(Y_train)
plot_confusion_matrix(cm, classes)
def load_img_id(idx):
    #print(idx)
    img = X_test[idx].reshape(64,64,3)
    return img

def plot_val_with_title(idxs, title):
    print(idxs)
    imgs = [load_img_id(x) for x in idxs]
    title_probs = [(preds[x], y[x]) for x in idxs]
    print(title)
    return plots(imgs, rows=4, titles=title_probs, figsize=(16,8)) if len(imgs)>0 else print('Not Found.')
# count incorrect predictions
incorrect_digits = np.where(preds != y)[0]
len(incorrect_digits)
# visualize incorrect predictions
plot_val_with_title(incorrect_digits, "Incorrect digits (prediction, label)")