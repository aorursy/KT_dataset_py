import numpy as np 
import pandas as pd 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train_V2.csv') #loading in the training set
print(train.head()) #examining the first few rows of the training set
training_average_placement = train['winPlacePerc'].mean() #this calculates the average value in the "winPlacePerc" columns
print(training_average_placement)
submission = pd.read_csv('../input/sample_submission_V2.csv')
submission['winPlacePerc'] = training_average_placement
print(submission.head())
submission.to_csv("Everyone_Averaged.csv", index=False) #no bad characters in the csv, or you will get an error!