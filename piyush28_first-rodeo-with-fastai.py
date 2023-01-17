import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from fastai import vision

from fastai import metrics



import os

print(os.listdir("../input"))
train_imgs_path = '../input/train/train'

test_imgs_path = '../input/test/test'

labels_path = '../input/train.csv'

in_path = '../input/'
df = pd.read_csv(labels_path)

df['id'] = 'train/train/' + df['id']

df.head()
df.has_cactus.hist(grid=False, figsize=(5, 4), bins=np.arange(3)-0.3, width=0.6)

plt.xticks([0, 1])

plt.show()
data = vision.ImageDataBunch.from_df(in_path, df, ds_tfms=vision.get_transforms(), size=224)

data = data.normalize(vision.imagenet_stats)
data.show_batch(rows=3, figsize=(10, 8))
learn = vision.cnn_learner(data, vision.models.resnet34, metrics=metrics.accuracy)

learn.fit(2)
interp = vision.ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(5,5), dpi=60)
submission_df = pd.read_csv('../input/sample_submission.csv')

files = submission_df['id'].values

img_paths = ('../input/test/test/' + submission_df['id']).values
img_paths[:10]
from tqdm import tqdm

preds = []



for p in tqdm(img_paths):

    pred = learn.predict(vision.open_image(p))[-1].numpy()

    preds.append(pred)
submission_df['has_cactus'] = np.array(preds)[:, 1]

submission_df.head()
np.sum(np.array(preds)[:, 1] > 0.5), submission_df.shape
submission_df.to_csv('submission.csv', index=False)