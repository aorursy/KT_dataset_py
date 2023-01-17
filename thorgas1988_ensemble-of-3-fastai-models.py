import numpy as np

import pandas as pd

import os

import fastai

from fastai.vision import *

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

import pydicom

import torchvision.models as models

from tqdm import tqdm
class RegMetrics(Callback):

  "Stores predictions and targets to perform calculations on epoch end."

  def on_epoch_begin(self, **kwargs):

    self.targs, self.preds = Tensor([]), Tensor([])



  def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs):

    assert last_output[:, 1].numel() == last_target.numel(), "Expected same numbers of elements in pred {} & targ {}".format(last_output.shape, last_target.shape)

    self.preds = torch.cat((self.preds, partial(F.softmax, dim=-1)(last_output)[:, 1].cpu()))

    self.targs = torch.cat((self.targs, last_target.cpu().float()))



# Define some custom metrics 

class AUCROC(RegMetrics):

  """ Compute the area under the receiver operating characteristic curve. """

  def on_epoch_begin(self, **kwargs):

    super().on_epoch_begin()

    

  def on_epoch_end(self, **kwargs):

    self.metric = roc_auc_score(self.targs, self.preds)
df_train = pd.read_csv('../input/train.csv')

sample_set = df_train.sample(6)

print('Train.csv samples:')

print(sample_set)



print('\nNumber of training samples:{0}'.format(len(os.listdir('../input/train/train'))))

print('Number of test samples:{0}'.format(len(os.listdir('../input/test/test'))))
fig, ax = plt.subplots(2, 3)

index = 0

for row in ax:

    for col in row:

        img = open_image('../input/train/train/'+str(sample_set.iloc[index]["id"]))

        img.show(col, title='Cactus:'+str(sample_set.iloc[index]["has_cactus"]))

        index += 1
train_path = '../input/train/train'



number_epochs=5



def train(arch):

    tfms = get_transforms(do_flip=True, flip_vert=False, max_rotate=10.,

                            max_zoom=1.1, max_lighting=0.2, max_warp=0.2, 

                            p_affine=1.0, p_lighting=0.0)

    

    #setup data source

    data = ImageDataBunch.from_df(path=train_path, df=df_train, label_col=1, bs=16, size=32, ds_tfms=tfms)



    #define learner

    learn = cnn_learner(data, arch, metrics=[accuracy, AUCROC()], model_dir='../../../models')



    #train

    learn.fit_one_cycle(number_epochs, 3e-3)

    

    return learn

resnet50_learner = train(models.resnet50)

densenet121_learner = train(models.densenet121)

vgg_learner = train(models.vgg19_bn)
result_csv = 'submission.csv'

test_path = '../input/test/test/'



def ensemble_predition(test_img):

    img = open_image(test_path + test_img)

    

    resnet50_predicition = resnet50_learner.predict(img)

    densenet121_predicition = densenet121_learner.predict(img)

    vgg_predicition = vgg_learner.predict(img)

    

    #ensemble average

    sum_pred = resnet50_predicition[2] + densenet121_predicition[2] + vgg_predicition[2]

    prediction = sum_pred / 3

    

    #prediction results

    predicted_label = torch.argmax(prediction).item()

    

    return predicted_label



#to give np array the correct style

submission_data = np.array([['dummy', 0]])



#progress bar

with tqdm(total=len(os.listdir(test_path))) as pbar:       

    #test all test images

    for img in os.listdir(test_path):

        label = ensemble_predition(img)

        new_np_array = np.array([[img, label]])

        submission_data = np.concatenate((submission_data, new_np_array), axis=0)

        pbar.update(1)



#remove dummy

submission_data = np.delete(submission_data, 0, 0)



#save final submission

result_df = pd.DataFrame(submission_data, columns=['id','has_cactus'])

result_df.to_csv(result_csv, index=False)


