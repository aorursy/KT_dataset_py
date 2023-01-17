# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

import seaborn as sb

import math

from sklearn.model_selection import train_test_split

%matplotlib inline
data=pd.read_csv('../input/HR_comma_sep.csv')

data.head()
print(data.describe())



print ('<---------------------------------->'+ '\n')

data.info()
#studying categories



print (data['sales'].unique())

print (data['salary'].unique())

#Any missing values?



data.isnull().sum()
fig, (axis1,axis2) = plt.subplots(1,2, figsize=(8,4))

sb.factorplot(x='time_spend_company', y='average_montly_hours', hue='left', data=data, kind='bar', ax=axis1)

sb.violinplot(x='number_project', y='average_montly_hours', hue='left', data=data, split=True, ax=axis2)
plt.figure(figsize=(8,4))



plt.subplot(1,2,1)

sb.countplot(x='time_spend_company', hue='left',data=data, palette="BrBG")

plt.subplot(1,2,2)

sb.countplot(x='number_project', hue='left',data=data, palette='BrBG')



sb.factorplot(x='time_spend_company', y='left', data=data, size=3, aspect=2)
f, (axis1,axis2)=plt.subplots(1,2, figsize=(8,4))

sb.countplot(x='salary', hue='left', data=data, ax=axis1)

#plt.subplot(1,2,2)

sb.factorplot(x='salary', y='promotion_last_5years', hue='left', data=data, ax=axis2, )
#Converting the continous variables into categorial for better estimation in graphs

data['satisfaction_level_cat']=0

 

def tocategory(level):

    temp=math.ceil(level*10)

    level=temp/10

    return level



data['satisfaction_level_cat']=data[['satisfaction_level']].apply(tocategory, axis=1)

data['satisfaction_level_cat'].head()



data['last_evaluation_cat']=data[['last_evaluation']].apply(tocategory, axis=1)

data['last_evaluation_cat'].head()
f, (axis1,axis2)=plt.subplots(1,2, figsize=(8,4))

sb.countplot(x="satisfaction_level_cat", hue="left", data=data,ax= axis1);

sb.countplot(x="last_evaluation_cat", hue="left", data=data,ax= axis2);

plt.subplot(111)

sb.boxplot(x="satisfaction_level_cat", y="last_evaluation_cat", hue="left", data=data, palette="PRGn")

#Splitting in training and testing datasets

model_data=data.drop(labels=['satisfaction_level_cat', 'last_evaluation_cat', 'salary', 'sales'], axis=1)

train = model_data.sample(frac=0.8, random_state=1)

print (train.shape)

test = model_data.loc[~model_data.index.isin(train.index)]



'''

#convert string variables into int

def harmonize(data):

    data.loc[data['salary']=='low']=0

    data.loc[data['salary']=='medium']=1

    data.loc[data['salary']=='low']=2

    

    data.loc[data['sales']== s=findname()'''
# Import the random forest model.

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score



print (train.columns)



predictors=['satisfaction_level', 'last_evaluation', 'number_project',

       'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years']



model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=1)

model.fit(train[predictors], train["left"])



predictions = model.predict(test[predictors])

accuracy_score(predictions, test['left'])