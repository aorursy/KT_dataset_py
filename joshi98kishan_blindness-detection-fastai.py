import os

print(os.listdir("/tmp/.cache"))
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate
path_data = Path('../input/aptos2019-blindness-detection')

path_data.ls()
path_train = path_data/'train_images'

path_test = path_data/'test_images'
df_labels = pd.read_csv(path_data/'train.csv')

df_labels.shape
data = ImageDataBunch.from_csv(path_data, 'train_images',

                               csv_labels = 'train.csv', suffix = '.png', test = 'test_images', 

                               ds_tfms = get_transforms(), size = 224, bs = 32).normalize(imagenet_stats)

data
data.show_batch(3, figsize = (8, 8))
tmp_dir = os.path.expanduser(os.path.join('/', 'tmp/.cache/torch/checkpoints'))

if not os.path.exists(tmp_dir):

    os.makedirs(tmp_dir)
!cp ../input/resnet34/resnet34.pth /tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth
print(os.listdir("/tmp/.cache/torch/checkpoints"))
kappa = KappaScore()

kappa.weights = 'quadratic'
learn = cnn_learner(data, models.resnet34, metrics = [accuracy, kappa], path='/kaggle/working/', model_dir = '/kaggle/working/')
learn.fit_one_cycle(4)
learn.save('s1-e4-res34')
Path('/kaggle/working').ls()
preds, pred_label = learn.get_preds(ds_type = DatasetType.Test)
y = preds.argmax(dim = 1)

y
df_submi = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

df_submi.head()
df_submi['diagnosis'] = y

# df_submi.head()
df_submi.to_csv('submission.csv', index_label = False, index = False)