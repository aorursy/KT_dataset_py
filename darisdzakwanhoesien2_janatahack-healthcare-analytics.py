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
test = pd.read_csv("../input/janatahackhealthcareanalytics/test_l0Auv8Q.csv")

submission = pd.read_csv("../input/janatahackhealthcareanalytics/sample_submmission.csv")

detail = pd.read_csv("../input/janatahackhealthcareanalytics/Train_2/Train/Health_Camp_Detail.csv")

profile = pd.read_csv("../input/janatahackhealthcareanalytics/Train_2/Train/Patient_Profile.csv")



first_train = pd.read_csv("../input/janatahackhealthcareanalytics/Train_2/Train/First_Health_Camp_Attended.csv")

second_train = pd.read_csv("../input/janatahackhealthcareanalytics/Train_2/Train/Second_Health_Camp_Attended.csv")

third_train = pd.read_csv("../input/janatahackhealthcareanalytics/Train_2/Train/Third_Health_Camp_Attended.csv")

train = pd.read_csv("../input/janatahackhealthcareanalytics/Train_2/Train/Train.csv")
print('Test',test.shape) 

print('Train',train.shape) 



print('Submission',submission.shape) 

print('Detail',detail.shape) 

print('Profile',profile.shape) 



print('Train 1st',first_train.shape) 

print('Train 2nd',second_train.shape) 

print('Train 3rd',third_train.shape) 
print('Test',test.columns) 

print('Train',train.columns) 



print('Submission',submission.columns) 

print('Detail',detail.columns) 

print('Profile',profile.columns) 



print('Train 1st',first_train.columns) 

print('Train 2nd',second_train.columns) 

print('Train 3rd',third_train.columns) 
train
test
submission 
detail
profile
first_train
second_train
third_train
train.columns
first_train.columns
train2 = train.merge(first_train.drop(['Donation','Unnamed: 4'],axis=1), on=['Patient_ID', 'Health_Camp_ID'])

train2.columns = ['Patient_ID', 'Health_Camp_ID', 'Registration_Date', 'Var1', 'Var2', 'Var3', 'Var4', 'Var5', 'Health Score']

train2
train1 = train.merge(second_train, on=['Patient_ID', 'Health_Camp_ID'])

train1
train_dataset = pd.concat([train1,train2])

train_dataset
x_train = train_dataset.iloc[:len(train_dataset)*9//10].drop(['Patient_ID','Health_Camp_ID','Registration_Date','Health Score'], axis=1)

x_val = train_dataset.iloc[len(train_dataset)*9//10:].drop(['Patient_ID','Health_Camp_ID','Registration_Date','Health Score'], axis=1)



y_train = train_dataset.iloc[:len(train_dataset)*9//10]['Health Score']

y_val = train_dataset.iloc[len(train_dataset)*9//10:]['Health Score']
import time

from xgboost import XGBRegressor

ts = time.time()



model = XGBRegressor(

    max_depth=10,

    n_estimators=1000,

    min_child_weight=0.5, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.1,

#     tree_method='gpu_hist',

    seed=42)



model.fit(

    x_train, 

    y_train, 

    eval_metric="rmse", 

    eval_set=[(x_train, y_train), (x_val, y_val)], 

    verbose=True, 

    early_stopping_rounds = 20)



time.time() - ts
test_dataset = test.merge(submission, on=['Patient_ID', 'Health_Camp_ID'])

test_dataset
str.isupper('U')
x_test = test_dataset.drop(['Patient_ID','Health_Camp_ID','Registration_Date','Outcome'], axis=1)



Y_pred = model.predict(x_val).clip(0, 20)

Y_test = model.predict(x_test).clip(0, 20)
Y_test
submission['Outcome'] = Y_test

submission.to_csv('submission.csv',index=False)

submission
from IPython.display import display, Image

display(Image(filename='../input/results/Janatahack - Healthcare Analytics.PNG'))