import numpy as np

import pandas as pd

from fastai.vision import *
path = Path('../input')
path.ls()
get_image_files(path/'train-jpg')[:5]
df = pd.read_csv(path/'train_v2.csv')

df.head()
np.random.seed(42)

size = 224

bs = 64

num_workers = 0  # set this to 2 to prevent kernel from crashing
tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
src = (ImageItemList.from_csv(path, 'train_v2.csv', folder='train-jpg', suffix='.jpg')

       .random_split_by_pct()

       .label_from_df(sep=' ')

       .add_test_folder('test-jpg-v2'))
data = (src.transform(tfms, size=size)

        .databunch(bs=bs, num_workers=num_workers)

        .normalize(imagenet_stats))
print(len(data.train_ds))

print(len(data.valid_ds))

print(len(data.test_ds))
data.classes
data.show_batch(rows=3, figsize=(7,6))
arch = models.resnet50

acc = partial(accuracy_thresh, thresh=0.2)

f_score = partial(fbeta, thresh=0.2)
learn = create_cnn(data, arch, metrics=[acc, f_score], model_dir='/tmp/models')
learn.lr_find()

learn.recorder.plot()
lr = 1e-2
learn.fit_one_cycle(4, slice(lr))
learn.save('stage-1')
learn.recorder.plot_losses()
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))
learn.save('stage-2')
learn.recorder.plot_losses()
preds, y = learn.get_preds(ds_type=DatasetType.Test)
preds[:5]
thresh = 0.2

labelled_preds = [' '.join([learn.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]
labelled_preds[:5]
submission = pd.DataFrame({'image_name':os.listdir('../input/test-jpg-v2'), 'tags':labelled_preds})
submission['image_name'] = submission['image_name'].map(lambda x: x.split('.')[0])
submission = submission.sort_values('image_name')
submission[:5]
submission.to_csv('submission.csv', index=False)