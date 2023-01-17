from fastai import *

from fastai.vision import *

import pandas as pd
labels = pd.read_csv('../input/train.csv')

def get_labels(name):

    return labels[labels['id'] == name.name]['has_cactus'].values[0]
data = ImageList.from_folder('../input/train/train/').split_by_rand_pct(0.01).label_from_func(get_labels).transform(get_transforms(), size=224).add_test(ItemList.from_folder('../input/test/test/')).databunch(bs=64).normalize(imagenet_stats)
learner = cnn_learner(data=data, base_arch=models.densenet201, model_dir='../../../../models/', metrics=accuracy)
learner.fit_one_cycle(5, slice(5e-2))
preds = learner.get_preds(DatasetType.Test)
predictions = preds[0].argmax(dim=1)

names = [item.name for item in data.test_ds.x.items]

pd.DataFrame({'id': names, 'has_cactus': predictions}).to_csv('submission.csv', index=False)