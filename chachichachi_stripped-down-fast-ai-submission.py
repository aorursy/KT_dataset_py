from fastai import *

from fastai.vision import *

from torchvision.models import * 



import os

import pathlib

import matplotlib.pyplot as plt
np.random.seed(123)



path = Path("../input")



tfms = get_transforms(do_flip = True,flip_vert = True)
sz = 36

data = ImageDataBunch.from_csv(path, folder = 'train', csv_labels = "train_labels.csv",

                               test = 'test',suffix=".tif", size = sz, ds_tfms = tfms)

data.path = pathlib.Path('.')

stats = data.batch_stats()

data.normalize(stats)
from sklearn.metrics import roc_auc_score



def auc_score(y_pred,y_true,tens=True):

    score = roc_auc_score(y_true,torch.sigmoid(y_pred)[:,1])

    if tens:

        score = tensor(score)

    return score
learn = create_cnn(data,resnet18, pretrained = True,metrics = [auc_score ,accuracy])

learn.fit(epochs=3)
submissions = pd.read_csv('../input/sample_submission.csv')

id_list = list(submissions.id)
preds,y = learn.TTA(ds_type=DatasetType.Test)

pred_list = list(preds[:,1])
print(learn.data.test_ds.items)
pred_dict = dict((key, value.item()) for (key, value) in zip(learn.data.test_ds.items,pred_list))

pred_ordered = [pred_dict[Path('../input/test/' + id + '.tif')] for id in id_list]
import time # Adding int(time.time() at end of file name so it's unique)

submissions = pd.DataFrame({'id':id_list,'label':pred_ordered})

submissions.to_csv("submission_weightdecay_{}.csv".format(int(time.time())),index = False)