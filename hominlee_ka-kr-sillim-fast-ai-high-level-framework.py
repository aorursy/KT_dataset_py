import fastai

from fastai.vision import *

fastai.__version__
path = Path('../input/2019-3rd-ml-month-with-kakr/')
# Making pretrained weights work without needing to find the default filename

from torch.utils import model_zoo

Path('models').mkdir(exist_ok=True)

!cp '../input/densenet201/densenet201.pth' 'models/'

def load_url(*args, **kwargs):

    model_dir = Path('models')

    filename  = 'densenet201.pth'

    if not (model_dir/filename).is_file(): raise FileNotFoundError

    return torch.load(model_dir/filename)

model_zoo.load_url = load_url
# Load train dataframe

train_df = pd.read_csv(path/'train.csv')

train_df.head()
# Load sample submission

test_df = pd.read_csv(path/'sample_submission.csv')

test_df.head()
tfms = get_transforms(do_flip=True, flip_vert=False, max_rotate=0.10, max_zoom=1.5, max_lighting=0.2)
train, test = [ImageList.from_df(df, path=path, cols='img_file', folder=folder) 

               for df, folder in zip([train_df, test_df], ['train', 'test'])]

data = (train.split_by_rand_pct(0.2, seed=42)

        .label_from_df(cols='class')

        .add_test(test)

        .transform(tfms, size=224)

        .databunch(path=Path('.'), bs=64).normalize(imagenet_stats))
data.show_batch() # data에서 하나의 배치를 확인하게 할 수 있게 합니다.
data.classes # data가 갖는 class들을 볼 수 있게 합니다.
learn = cnn_learner(data, base_arch=models.densenet201, metrics=accuracy)
# Find a good learning rate

learn.lr_find()

learn.recorder.plot()
lr = 7e-02

learn.fit_one_cycle(3, slice(lr))
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
lr = 7e-4

learn.fit_one_cycle(21, slice(lr/10, lr))
# Test predictions

test_preds = learn.TTA(ds_type=DatasetType.Test)

test_df['class'] = np.argmax(test_preds[0], axis=1) + 1

test_df.head()
test_df.to_csv('submission.csv', index=False) 