# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 

import pandas as pd 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_file_path = '../input/pubg-finish-placement-prediction/train_V2.csv'

train = pd.read_csv(train_file_path) #loading in the training set

print(train.head()) #examining the first few rows of the training set
training_average_placement = train['winPlacePerc'].mean() #this calculates the average value in the "winPlacePerc" columns

print(training_average_placement)
submission_file_path = '../input/pubg-finish-placement-prediction/sample_submission_V2.csv'

submission = pd.read_csv(submission_file_path)

submission['winPlacePerc'] = training_average_placement

print(submission.head())
submission.to_csv("submission.csv", index=False) #no bad characters in the csv, or you will get an error!