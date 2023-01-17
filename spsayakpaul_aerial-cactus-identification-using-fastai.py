import numpy as np

import pandas as pd 



import matplotlib.pyplot as plt

plt.style.use('ggplot')



import torch

from fastai.vision import *

from fastai.metrics import *



np.random.seed(7)

torch.cuda.manual_seed_all(7)
import os

print(os.listdir("../input"))
train_dir="../input/train/train"

test_dir="../input/test/test"

train = pd.read_csv('../input/train.csv')

sub_file = pd.read_csv("../input/sample_submission.csv")

data_folder = Path("../input")
train.head()
sub_file.head()
# transformations for data augmentation

trfm = get_transforms(do_flip=True, flip_vert=True, max_rotate=10.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75)
test_img = ImageList.from_df(sub_file, path=data_folder/'test', folder='test')



databunch = (ImageList.from_df(train, path=data_folder/'train', folder='train')

        .split_by_rand_pct(0.01)

        .label_from_df()

        .add_test(test_img)

        .transform(trfm, size=48)

        .databunch(path='.', bs=64, device= torch.device('cuda:0'))

        .normalize(imagenet_stats)

       )
databunch.show_batch(rows=3, figsize=(8,8))
databunch.classes
databunch.label_list
learn = cnn_learner(databunch, models.resnet34, metrics=[error_rate, accuracy])

learn.fit_one_cycle(5)
learn.recorder.plot_losses()
learn.lr_find()

learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(5, max_lr=slice(1e-03))
learn.recorder.plot_losses()
learn.show_results(rows=3)
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(databunch.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(12,10), heatmap=False)
interp.plot_confusion_matrix()
predictions1=learn.get_preds(DatasetType.Test)

predictions2=learn.get_preds(DatasetType.Test)

predictions3=learn.get_preds(DatasetType.Test)

predictions4=learn.get_preds(DatasetType.Test)

predictions5=learn.get_preds(DatasetType.Test)

predictions6=learn.get_preds(DatasetType.Test)

predictions7=learn.get_preds(DatasetType.Test)

predictions8=learn.get_preds(DatasetType.Test)



comb_output=[predictions1[0],predictions2[0],predictions3[0],predictions4[0],

            predictions5[0],predictions6[0],predictions7[0],predictions8[0]]



comb_output=torch.sum(torch.stack(comb_output),dim=0)
sub_file.has_cactus = comb_output.numpy()[:, 0]

sub_file.to_csv('submission.csv', index=False)