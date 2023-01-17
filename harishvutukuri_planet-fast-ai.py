import time

start_time = time.time()



%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai import *
from pathlib import Path

path = Path('../input/')
df = pd.read_csv(path/'train_v2.csv')

df.head()
tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
np.random.seed(42)

src = (ImageItemList.from_csv(path, 'train_v2.csv', folder='train-jpg', suffix='.jpg')

       .random_split_by_pct(0.2)

       .label_from_df(label_delim=' '))
data = (src

        .transform(tfms, size=128)

        .databunch()

        .normalize(imagenet_stats)

       )
data.show_batch(rows = 3)
arch = models.resnet50
acc_02 = partial(accuracy_thresh, thresh=0.2)

f_score = partial(fbeta, thresh=0.2)

learn = create_cnn(data, arch, metrics=[acc_02, f_score], path='../working/')
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(3, slice(2.29E-02,0.01))
learn.save('stage-1-rn50')
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(3, slice(7.59E-05))
learn.save('stage-2-rn50')

learn.export()
test = ImageItemList.from_folder(path/'test-jpg-v2').add(ImageImageList.from_folder(path/'test-jpg-additional'))

len(test)
learn = load_learner('../working/', test=test)

preds, _ = learn.get_preds(ds_type=DatasetType.Test)
thresh = 0.2

labelled_preds = [' '.join([learn.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]
submission = pd.read_csv(path/'sample_submission_v2.csv')

submission['tags'] = labelled_preds

submission.to_csv('fastai_resnet50.csv')
print('Kernel Runtime: {0} minutes '.format((time.time() - start_time)/60.0))