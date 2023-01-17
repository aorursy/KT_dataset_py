#This kernel applies the fastAI libraries to the well-known MNIST dataset.  

#There are a gazillion other descriptions of what MNIST is so I will not spend time giving an explanation here
#standard imports

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input")) #look for the files we'll be using

#we'll be using fastAI for this project

from fastai import *

from fastai.vision import *
#load the training data

df_train = pd.read_csv('../input/train.csv')

df_train['fn'] = df_train.index

df_train.head()
#df_train.head() just shows 0's for the pixel values  so let's find a spot where they're not zero

df_train.iloc[0,185:193]
class PixelImageItemList(ImageList):

    def open(self,fn):

        regex = re.compile(r'\d+')

        fn = re.findall(regex,fn)

        df = self.inner_df[self.inner_df.fn.values == int(fn[0])]

        df_fn = df[df.fn.values == int(fn[0])]

        img_pixel = df_fn.drop(labels=['label','fn'],axis=1).values

        img_pixel = img_pixel.reshape(28,28)

        img_pixel = np.stack((img_pixel,)*3,axis=-1)

        return vision.Image(pil2tensor(img_pixel,np.float32).div_(255))
#reshape each 784 line of ints into 28x28xRGB images using PixelImageItemList.  

#Random_split will take 20% of the data to use for validation

#label contains the *value* that the image is supposed to represent

src = (PixelImageItemList.from_df(df_train,'.',cols='fn')

      .split_by_rand_pct()

      .label_from_df(cols='label'))
#make sure our src looks like it's of the right shape and that we have both training and validation sets.

src
#Add transforms to the data since people will write numbers larger vs. smaller and at slight differnt angles of rotation

#do_flip=false means that people *don't* write their numbers backwards!

tfms = get_transforms(max_rotate=15., max_zoom=1.2, max_lighting=0, max_warp=0, do_flip=False)

data = ImageDataBunch.create_from_ll(src, ds_tfms=tfms)
#hey, now we have images that look like real numbers!

data.show_batch(rows=3, figsize=(6,6))
#standard resnet model.  Initially we'll leave the model frozen and just train the last couple of layers

learn = cnn_learner(data,models.resnet50,metrics=accuracy)
#find the best learning rate (look for a nice steep downards slope)

learn.lr_find()

learn.recorder.plot()
#Now let's do some learning!

#Weight decay helps avoid overfitting by taking into account the sum of the activation parameters within the 

#loss function, and therefore helps minimize the overall model complexity

learn.fit_one_cycle(6,slice(1e-2), wd=.1)
learn.save('50-stage-1')
#learn.load('50-stage-1')
#OK, now we'll unfreeze and train the whole model

learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(9,slice(2e-3,2e-5), wd=.1)
#99.6% accuracy is a pretty good result so let's save results and move on.

learn.save('50-stage-2')
#Out of interest, let's see where our model messed up...

interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9,figsize=(6,6))
#OK, we're done building our model.  Let's run agains the Kaggle test set and see how we do.

#Load in the test data

df_test = pd.read_csv('../input/test.csv')

df_test['label'] = 0

df_test['fn'] = df_test.index

df_test.head()
#Run our predictions

learn.data.add_test(PixelImageItemList.from_df(df_test, path='.', cols='fn'))

test_pred, test_y, test_loss = learn.get_preds(ds_type=DatasetType.Test, with_loss=True)

test_result = torch.argmax(test_pred,dim=1)

result = test_result.numpy()
#Check that the predictions is of the right length and width (e.g. number of posible values)

test_pred.shape
#Values look OK (e.g. range from 0-0)

result
#create a CSV file to submit

final = pd.Series(result,name='Label')

submission = pd.concat([pd.Series(range(1,28001),name='ImageId'),final],axis=1)

submission.to_csv('submission.csv',index=False)
#Cross my fingers and submit1

!head submission.csv