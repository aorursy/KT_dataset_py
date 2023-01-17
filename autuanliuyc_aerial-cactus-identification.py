%matplotlib inline

%reload_ext autoreload

%autoreload 2

# 多行输出

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all" 
from fastai.vision import *

from pathlib import Path
root = Path("../input")

root

root.as_posix()
train_df = pd.read_csv(root/"train.csv")

test_df = pd.read_csv(root/"sample_submission.csv")
train_df.head()

test_df.head()
test_set = ImageList.from_df(test_df, path=root/'test', cols='id', folder='test')
test_set
tsfm = get_transforms(do_flip=True, flip_vert=True, max_rotate=10.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75)
SZ=128

BS=64
np.random.seed(42)

data = (ImageList.from_df(train_df, path=root/'train', cols='id', folder='train')

       .split_by_rand_pct(0.01)

       .label_from_df()

       .transform(tsfm, size=SZ)

        .add_test(test_set)

       .databunch(path='./', bs=BS, device= torch.device('cuda:0'))

       .normalize(imagenet_stats)

      )
data
data.show_batch(rows=3, figsize=(6,6))
# arch = models.densenet161

arch = models.densenet169

# arch = models.densenet121

# arch = resnet50 # 也可以达到 1
learn = cnn_learner(data, arch, metrics=[error_rate, accuracy])
learn.lr_find()

learn.recorder.plot(suggestion=True)
lr = 1e-03

learn.fit_one_cycle(5, slice(lr))
learn.recorder.plot_losses()
# learn.unfreeze()
# learn.lr_find(1e-10,10)

# learn.recorder.plot(skip_end=15)
# lr1 = 5e-6

# learn.fit_one_cycle(2, slice(lr1/2.6**3, lr1))
# data1 = (ImageList.from_df(train_df, path=root/'train', cols='id', folder='train')

#        .split_by_rand_pct(0.01)

#        .label_from_df()

#        .transform(tsfm, size=224)

#         .add_test(test_set)

#        .databunch(path='./', bs=64, device= torch.device('cuda:0'))

#        .normalize(imagenet_stats)

#       )
# learn.data = data1
# learn.freeze()
# learn.lr_find()

# learn.recorder.plot()
# lr2 = 1e-3

# learn.fit_one_cycle(3, lr2)
# learn.unfreeze()
# learn.lr_find()

# learn.recorder.plot()
# lr3 = 1e-5

# learn.fit_one_cycle(2, slice(lr3/2.6**3,lr3))
learn.recorder.plot_losses()
preds,_ = learn.get_preds(ds_type=DatasetType.Test)
test_df.has_cactus = preds.numpy()[:, 0]
test_df.to_csv('submission.csv', index=False)