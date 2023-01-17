# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train_2v.csv')

df_test  = pd.read_csv('../input/test_2v.csv')

print("train set : ", df_train.shape)

print("test set : ", df_test.shape)

df_train.head(2)
print("############  gender")

print(df_train.groupby('gender').size())



print("############  hypertension")

print(df_train.groupby('hypertension').size())



print("############  heart_disease")

print(df_train.groupby('heart_disease').size())



print("############  ever_married")

print(df_train.groupby('ever_married').size())



print("############  work_type")

print(df_train.groupby('work_type').size())



print("############  work_type")

print(df_train.groupby('Residence_type').size())



print("############  smoking")

print(df_train.groupby('smoking_status').size())



print("############  stroke")

print(df_train.groupby('stroke').size())
import matplotlib.pyplot as plt

########## there are missing gender 



print("############  age")

print(df_train.age.describe())

plt.hist(df_train.age, bins = 20)

plt.show()

########  some patients are too young ... ? Maybe two models, one for adult and one for child



print("############ glucose")

print(df_train.avg_glucose_level.describe())

plt.hist(df_train.avg_glucose_level, bins = 20)

plt.show()

########  < 55 is really low, > 200 is weired, maybe not fasting ?



print("############ bmi")

print(df_train.bmi.describe())  

plt.hist(df_train.bmi, bins = 20)

plt.show()

#######  < 10 or > 50 is almost impossible .... should replace/filter those wrong data & there are missings 