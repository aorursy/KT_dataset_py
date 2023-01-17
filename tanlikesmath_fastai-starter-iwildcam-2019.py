%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

import pandas as pd
print('Make sure cuda is installed:', torch.cuda.is_available())

print('Make sure cudnn is enabled:', torch.backends.cudnn.enabled)
path = Path('../input')
# Load train dataframe

train_df = pd.read_csv(path/'train.csv')

train_df = pd.concat([train_df['id'],train_df['category_id']],axis=1,keys=['id','category_id'])

train_df.head()
# Load sample submission

test_df = pd.read_csv(path/'test.csv')

test_df = pd.DataFrame(test_df['id'])

test_df['predicted'] = 0

test_df.head()
train, test = [ImageList.from_df(df, path=path, cols='id', folder=folder, suffix='.jpg') 

               for df, folder in zip([train_df, test_df], ['train_images', 'test_images'])]

data = (train.split_by_rand_pct(0.2, seed=123)

        .label_from_df(cols='category_id')

        .add_test(test)

        .transform(get_transforms(), size=32)

        .databunch(path=Path('.'), bs=64).normalize())
data.show_batch()
learn = cnn_learner(data, base_arch=models.densenet121, metrics=[FBeta(),accuracy], wd=1e-5).mixup()
learn.lr_find()

learn.recorder.plot(suggestion=True)
lr = 2e-2

learn.fit_one_cycle(2, slice(lr))
learn.save('stage-1-sz32')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
lr = 1e-3

learn.fit_one_cycle(4, slice(lr/100, lr))
learn.save('stage-2-sz32')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
test_preds = learn.TTA(DatasetType.Test)

test_df['predicted'] = test_preds[0].argmax(dim=1)
test_df.to_csv('submission.csv', index=False)