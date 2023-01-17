!pip install -q fastai -t ./ftai
#!mv fastai ftai
from ftai.fastai.imports import *
from ftai.fastai.transforms import *
from ftai.fastai.conv_learner import *
from ftai.fastai.models import *
from ftai.fastai.dataset import *
from ftai.fastai.sgdr import *
from ftai.fastai.plots import *
!pip install -q torchvision -t ./ftai
# If you haven't downloaded weights.tgz yet, download the file
#     http://forums.fast.ai/t/error-when-trying-to-use-resnext50/7555
#     http://forums.fast.ai/t/lesson-2-in-class-discussion/7452/222
#!wget -P ftai/fastai/ http://files.fast.ai/models/weights.tgz
#!tar xvfz ftai/fastai/weights.tgz -C ftai/fastai/
!wget -q https://s3.amazonaws.com/resnext/imagenet_models/resnext_50_32x4d.t7
!mkdir ftai/fastai/weights
!cp resnext* ftai/fastai/weights
!rm resnext*
!mkdir data
!cp -r ../input/train-jpg data/
!cp ../input/train_v2.csv data/
!cp ../input/test_v2_file_mapping.csv data/
!mkdir data/test-jpg-v2
!cp -r ../input/test-jpg-v2/file_8220.jpg data/test-jpg-v2
PATH = "data/"
train = pd.read_csv(f'{PATH}train_v2.csv')
test = pd.read_csv(f'{PATH}test_v2_file_mapping.csv')
len(test), len(train)
train.head()
test.tail()
val_idxs = get_cv_idxs(len(list(open(f'{PATH}train_v2.csv')))-1)
len(val_idxs)
import cv2
img = cv2.imread(PATH+"train-jpg/"+train.iloc[100,0]+".jpg")
plt.imshow(img)
train.iloc[100,0][6:]
from fastai.imports import *
from fastai.transforms import *
from fastai.dataset import *
from sklearn.metrics import fbeta_score
import warnings

def f2(preds, targs, start=0.17, end=0.24, step=0.01):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return max([fbeta_score(targs, (preds>th), 2, average='samples')
                    for th in np.arange(start,end,step)])

def opt_th(preds, targs, start=0.17, end=0.24, step=0.01):
    ths = np.arange(start,end,step)
    idx = np.argmax([fbeta_score(targs, (preds>th), 2, average='samples')
                for th in ths])
    return ths[idx]

def get_data(path, tfms,bs,  n, cv_idx):
    val_idxs = get_cv_idxs(n, cv_idx)
    return ImageClassifierData.from_csv(path, 'train-jpg', f'{path}train_v2.csv', bs, tfms,
                                 suffix='.jpg', val_idxs=val_idxs, test_name='test-jpg-v2')

def get_data_zoom(f_model, path, sz, bs, n, cv_idx):
    tfms = tfms_from_model(f_model, sz, aug_tfms=transforms_top_down, max_zoom=1.05)
    return get_data(path, tfms, bs, n, cv_idx)

def get_data_pad(f_model, path, sz, bs, n, cv_idx):
    transforms_pt = [RandomRotateZoom(9, 0.18, 0.1), RandomLighting(0.05, 0.1), RandomDihedral()]
    tfms = tfms_from_model(f_model, sz, aug_tfms=transforms_pt, pad=sz//12)
    return get_data(path, tfms, bs, n, cv_idx)
sz = 256
f_model = resnet34
bs = 64
n=len(list(open(f'{PATH}train_v2.csv')))-1
data=get_data_pad(f_model, PATH, 256, 64, n, 0)
learn = ConvLearner.pretrained(f_model, data, metrics=[f2])
lrf = learn.lr_find()
learn.sched.plot()
lr = 0.1
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)
lrs = np.array([lr/9, lr/3, lr])
learn.unfreeze()
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
#learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
learn.sched.plot_loss()
lrs = [lr/13, lr/9, lr/5]
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
learn.sched.plot_loss()
# Save Results
#path = "data/models/amazonSpaceResnet34_224-3.h5"
#save_file_to_drive(path, path, "x-hdf","")
# You can save this file in any folder by going to drive.
log_preds, y = learn.TTA(is_test=True)
img = cv2.imread("data/"+data.test_dl.dataset.fnames[0])
plt.imshow(img)
def get_labels(a): return [data.classes[o] for o in a.nonzero()[0]]
get_labels(log_preds[0][0][:]>0.2)
#new_preds = log_preds>0.20
#res = pd.DataFrame(index=np.arange(61191), columns=["image_name", "tags"] )
#" ".join(get_labels(log_preds[0,1000,:]))
#for i in range(61191):
#  name = data.test_dl.dataset.fnames[i][9:-4]
#  res.iloc[i, :] = np.array([name, " ".join(get_labels(new_preds[0,i,:]))])
#res.tail()
#SUBM = f'{PATH}/subm/'
#os.makedirs(SUBM, exist_ok=True)
#res.to_csv(f'{SUBM}subm.csv', index=False)
# Submit Predictions
#!kaggle competitions submit -f data/subm/subm.csv -m "On Resnet34" planet-understanding-the-amazon-from-space
!rm -r ftai
!rm -r data