import os, sys, random

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
df_MetaData = pd.read_csv('../input/train-set-metadata-for-dfdc/metadata')

df_MetaData.head()
sample_submission = pd.read_csv("../input/sampledeepfakesubmissioncsv/sample_submission.csv")

sample_submission.head()
def label_convert(label):

    if label=="REAL":

        return 0

    else:

        return 1



test_dir = "/kaggle/input/deepfake-detection-challenge/test_videos/"

test_videos = sorted([x for x in os.listdir(test_dir) if x[-4:] == ".mp4"])



df_TestData = pd.DataFrame(columns=['label', 'prediction'])

for file_id in tqdm(test_videos):

    df_TestData.loc[file_id] = [label_convert(list(df_MetaData[df_MetaData.filename==file_id].label)[0]),list(sample_submission[sample_submission.filename==file_id].label) [0]

]

    

df_TestData.head()
df_Real =df_TestData[df_TestData.label==0]

df_Fake =df_TestData[df_TestData.label==1]



data = list(df_Real.prediction)

count = np.histogram(data)[0]

plt.hist(data,50)



data = list(df_Fake.prediction)

count = np.histogram(data)[0]

plt.hist(data,50)



plt.legend(['REAL', 'FAKE'])



plt.axis([0,1,0,140])

plt.show()
from sklearn.metrics import log_loss

LOG_LOSS = log_loss(list(df_TestData.label),list(df_TestData.prediction))

print("Log loss in the test folder is: " + str(LOG_LOSS))

from sklearn.metrics import confusion_matrix

df_CM = df_TestData.copy()

df_CM.loc[df_CM.prediction>0.5,'prediction']=1

df_CM.loc[df_CM.prediction<=0.5,'prediction']=0



CONFUSION_MATRIX = confusion_matrix(list(df_TestData.label),list(df_CM.prediction))

print("Confusion Matrix is:\n" + str(CONFUSION_MATRIX))