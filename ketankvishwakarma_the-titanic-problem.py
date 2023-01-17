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
import pandas as pd
import sklearn as sklearn
import numpy as np

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

import matplotlib.pyplot as matplot
import seaborn as seaborn
seaborn.set_style( 'white' )
%matplotlib inline
train_data_path = '../input/train.csv'
test_data_path = '../input/test.csv'
train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(train_data_path)
# train_df[['Name','Ticket']]
# drop unsignificant column 
train_df.drop(train_df[['Name','Ticket']],axis=1,inplace=True)
# drop nan values 
for col in train_df.columns.values:
   train_df.dropna(subset=[col], how='all', inplace = True)

# Pclass vs Servived 
# 1. dataframe of two columns 
pcclassServived  = train_df[['Pclass','Survived']]

# 2. group by PC Class 
pc_group = pcclassServived.groupby('Pclass',as_index=False)

# 3. lets see what the group has 
pc_group.get_group(1)
# 4. take mean and sort 
print(pc_group.mean().sort_values(by='Survived',ascending=False))
def visualize(column):
    tempDF = train_df[[column,'Survived']]
    trmpX = tempDF.groupby(column,as_index=False).mean().sort_values(by='Survived',ascending=False)
    trmpX['Died'] = 1-trmpX['Survived']
    xtick = []
    xIndex = []
    for x in range(len(trmpX[column])):
        xIndex.append(x)
        xtick.append(trmpX[column][x])
        matplot.xticks(xIndex,xtick)
    matplot.xlabel(column)
    matplot.ylabel('% Servived')   
    matplot.bar(xIndex,height=0,linewidth=0)
    matplot.bar(xIndex,trmpX['Survived'] *100,label='Survived',color='g')
    matplot.bar(xIndex,trmpX['Died'] * 100,label='Died',color='red')
    matplot.legend(loc='upper right')
validCols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare', 'Embarked']
# group by describtion
def groupData(column):
     tempDF = train_df[[column,'Survived']]
     return tempDF.groupby(column,as_index=False).mean().sort_values(by='Survived',ascending=False)   
## Plot all values 
matplot.figure(figsize=(8, 10))
for cols in validCols:
    matplot.subplot(7,1,validCols.index(cols)+1)
    matplot.subplots_adjust(left=0, bottom=1,right=1, top=3,wspace=0, hspace=0.3)
    visualize(cols)
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)
survived = train_df['Survived']
columns_to_trianwith = ['Sex','Age','SibSp','Parch','Fare']
training_data = train_df[columns_to_trianwith]
print(len(training_data))
new_train_data , new_validate_data = training_data[0:83], training_data[83:len(training_data)] 
print(len(new_train_data))
print(len(new_validate_data))
sex_map = {'female':1, 'male':2}
training_data = training_data.applymap(lambda s: sex_map.get(s) if s in sex_map else s)
from sklearn.naive_bayes import GaussianNB
naive_bayes_classifier = GaussianNB()
model = naive_bayes_classifier.fit(training_data,survived)
test = test_df[columns_to_trianwith]
test = test.applymap(lambda s: sex_map.get(s) if s in sex_map else s)
test.dropna(inplace=True)
pridiction = model.predict(test)
actual = test_df['Survived']
from sklearn.metrics import accuracy_score
accuracy_score(actual,pridiction,normalize=True)
