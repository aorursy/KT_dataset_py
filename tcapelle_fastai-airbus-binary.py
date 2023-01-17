%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai.conv_learner import *

torch.cuda.is_available()
torch.backends.cudnn.enabled
INPUT_PATH = '../input/airbus-ship-detection/'
TRAIN = os.path.join(INPUT_PATH, 'train')
TEST = os.path.join(INPUT_PATH, 'test')
!mkdir tmp
!mkdir model
TMP = '/kaggle/working/tmp'
MODEL = '/kaggle/working/model'
!ls
masks = pd.read_csv(os.path.join(INPUT_PATH, 'train_ship_segmentations.csv'))
masks.head()
def is_boat(s):
  s = str(s)
  if len(s)>0 and ('nan' not in str.lower(s)):
    return 1
  else: return 0
masks['EncodedPixels']=masks['EncodedPixels'].apply(is_boat)
masks.drop_duplicates(inplace=True)
masks.head()
masks.hist()
masks.to_csv('boat_count.csv',index=False)
def get_data(sz):
    tfms = tfms_from_model(f_model, sz, aug_tfms=transforms_top_down, max_zoom=1.05)
    return ImageClassifierData.from_csv(INPUT_PATH, 'train', label_csv, tfms=tfms,
                    suffix='', val_idxs=val_idxs, test_name='test')
      
bs=64; 
f_model = resnet34
n = len(list(open('boat_count.csv')))-1
print(n)
label_csv = 'boat_count.csv'
val_idxs = get_cv_idxs(n)
sz = 64
data = get_data(sz)
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
data = data.resize(int(sz*1.3), '/tmp/')
x,y = next(iter(data.val_dl))
plt.imshow(data.val_ds.denorm(to_np(x))[0]*1.4);
!cp ../input/resnet34/resnet34.pth /tmp/.torch/models/resnet34-333f7ec4.pth
learner = ConvLearner.pretrained(f_model, data, tmp_name=TMP,models_name=MODEL)
lrf=learner.lr_find()
learner.sched.plot()
lr = (1E-2)/2
learner.fit(lr, 3, cycle_len=1, cycle_mult=2)
learn.save(f'{sz}')
lrs = np.array([lr/9,lr/3,lr])
learner.unfreeze()
learner.fit(lrs, 3, cycle_len=1, cycle_mult=2)
learn.save(f'{sz}-lrs ')














