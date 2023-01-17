#################################################################

#  Importing the libraries 

#################################################################

from fastai import * 

from fastai.vision import * 

from fastai.callbacks import *



from sklearn.metrics import roc_auc_score,f1_score

import cv2 as cv

import numpy as np

import pandas as pd 

import os



import matplotlib.pyplot as plt

%matplotlib inline 
#################################################################

#  adjusting the path and displaying the files 

#################################################################



print(os.listdir(("../input/aptos2019-blindness-detection")))

kaggle_path = '../input/aptos2019-blindness-detection'

#################################################################

#  Reading the csv files  

#################################################################

labels = pd.read_csv(kaggle_path+'/train.csv')

test_labels = pd.read_csv(kaggle_path+'/sample_submission.csv')

test = ImageList.from_df(test_labels, path = kaggle_path+'/test_images', suffix = '.png')
#################################################################

#  For Debugging and understanding the data 

#################################################################

labels.head()
#################################################################

#  Transforming parameters for augmentation 

#################################################################

tfms = get_transforms(

    do_flip=True,

    flip_vert=False,

    max_warp=0.15,

    max_rotate=360.,

    max_zoom=1.1,

    max_lighting=0.1,

    p_lighting=0.5

)
#################################################################

#  For Debugging and understanding the data 

#################################################################

labels['diagnosis'].value_counts().plot(kind = 'bar', title='Distribution of diagnosis categories')

plt.show()

labels.head()
#################################################################

#  For Debugging and understanding the data 

#################################################################

img = open_image(kaggle_path+'/train_images/002c21358ce6.png')

img.show(figsize = (5,5))

print(img.shape)
#################################################################

#  loading arranging and splitting the data  

#################################################################

src = (ImageList.from_df(labels, path = kaggle_path+'/train_images', suffix = '.png')

       .split_by_rand_pct(0.2) # slpliting 20 -80

       .label_from_df(cols = 'diagnosis')

       .add_test(test))
#################################################################

#  adjusting data, image size , batch sizes and normalization  

#################################################################

data = (

    src.transform(

        tfms,

        size = 224, 

        resize_method=ResizeMethod.SQUISH,

        padding_mode='zeros'

    )

    .databunch(bs=32)

    .normalize(imagenet_stats))

#################################################################

#  For Debugging and understanding the data 

#################################################################

print(data.classes)

data.show_batch(rows=3, figsize=(10,6), hide_axis=False)
#################################################################

#  Kappa score matrix  

#################################################################

kappa = KappaScore()

kappa.weights = "quadratic"
#################################################################

#  making CNN resnet 34  

#################################################################

learn = cnn_learner(

    data, 

    models.resnet34, 

    metrics = [accuracy, kappa], 

    model_dir = Path(kaggle_path+'/working'),

    path = Path(".")

)
#################################################################

#  For learning  

#################################################################

learn.fit_one_cycle(1)
#################################################################

#  For readjusting the weights if needed

#################################################################

# MODEL_PATH = str(arch).split()[1]

learn.model_dir='/kaggle/working/'



# learner.save(MODEL_PATH + '_stage1')

learn.save('model-34')



# learn.unfreeze()

# learn.lr_find()

# learn.recorder.plot(suggestion=True)
######################################################################

#  Finding the best results base upon kappa and saving the best model

######################################################################

learn.model_dir='/kaggle/working/'

lr = slice(1e-4,1e-3)

learn.fit_one_cycle(5,lr, callbacks=[SaveModelCallback(learn, every='imrpovement', monitor='kappa_score', name='bestmodel')])
#################################################################

#  For submitting the result

#################################################################



learn.load('bestmodel')

preds, _ = learn.get_preds(ds_type=DatasetType.Test)



preds = np.array(preds.argmax(1)).astype(int).tolist()

submission = pd.read_csv(kaggle_path+'/sample_submission.csv')

submission.head()
#################################################################

#  Saving our results on the submission file 

#################################################################

submission['diagnosis'] = preds

submission.head()
#################################################################

#  Generating the output file 

#################################################################

# submission.to_csv('submission.csv', index = False)
#################################################################

#  For debugging 

#################################################################

interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()