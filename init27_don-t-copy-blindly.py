%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate
!ls ../input/train
path = Path("../input/")

tfms = get_transforms()

data = ImageDataBunch.from_folder(path,valid_pct=0.1,test="test", ds_tfms=tfms, size=224)
learn = cnn_learner(data, models.resnet18, metrics=error_rate)
learn.fit_one_cycle(1)
sub=pd.read_csv('../input/sample_submission.csv').set_index('id')

sub.head()
preds, _ = learn.get_preds(ds_type=DatasetType.Test)

thresh = 0.2

labelled_preds = [' '.join([learn.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]



labelled_preds[:5]
fnames = [f.name[:-4] for f in learn.data.test_ds.items]
df = pd.DataFrame({'id':fnames, 'predicted_class':labelled_preds}, columns=['id', 'predicted_class'])

df['id'] = df['id'].astype(str) + '.jpg'

df.to_csv('/kaggle/working/submission.csv', index=False)