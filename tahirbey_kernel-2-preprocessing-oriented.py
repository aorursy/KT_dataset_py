# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
#plt.style.use('seaborn-whitegrid')
import seaborn as sns
from collections import Counter

import warnings
warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data.columns
train_data.head()
train_data.describe() #To have more datails on the overview of the dataset. Using the output of descibe() method, we can understand general tendency
# of the variables. I believe a first impression on the dataset is important, it guides us through our path.
train_data.info()
def bar_plot(variable):
    
    #* input : 'sex'
    #output : barplot and value count
    
    var = train_data[variable] #get feature
    varValue = var.value_counts() #count number of categorical variable
    
    #visualization
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index,varValue.index.values)
    plt.ylabel('Frequency')
    plt.title(variable)
    plt.show()
    print(' {}: /n {}' .format(variable,varValue))
category1 = ['Survived','Sex','Pclass','Embarked','SibSp','Parch']
for c in category1:
    bar_plot(c)
# Variables that are categoric but visualization is problematic
category2 = ['Cabin','Name','Ticket']
for c in category2:
    print( train_data[c].value_counts())
def plot_hist(variable):
    plt.plot(figsize = (9,3))
    plt.hist(train_data[variable],bins = 50)
    plt.xlabel(variable)
    plt.ylabel('Frequency')
    plt.title('{} distribution with hist' .format(variable))
    plt.show()
numericVar = ['Fare','Age','PassengerId']
for n in numericVar:
    plot_hist(n)

#Pclass-Survived
train_data[['Pclass','Survived']].groupby(['Pclass'],as_index = False).mean().sort_values(by='Survived',ascending = False)
#Sex-Survived
train_data[['Sex','Survived']].groupby(['Sex'],as_index = False).mean().sort_values(by='Survived',ascending = False)
#SibSp-Survived
train_data[['SibSp','Survived']].groupby(['SibSp'],as_index = False).mean().sort_values(by='Survived',ascending = False)
#Parch-Survived
train_data[['Parch','Survived']].groupby(['Parch'],as_index = False).mean().sort_values(by='Survived',ascending = False)
def detect_outliers(data,features):
    outlier_indices = []
    
    for c in features:
        
        #1st Quartile
        Q1 = np.percentile(data[c],25)
        #3rd Quartile
        Q3 = np.percentile(data[c],75)
        #IQR
        IQR = Q3 - Q1
        #Outlier step
        outlier_step = IQR * 1.5
        #detect outliers and indices
        outlier_list_col = data[(data[c] < Q1 - outlier_step) | (data[c] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i,v in outlier_indices.items() if v > 2)
    
    return multiple_outliers
train_data.loc[detect_outliers(train_data,['Age','SibSp','Parch','Fare'])]
#drop outliers
train_data = train_data.drop(detect_outliers(train_data,['Age','SibSp','Parch','Fare']),axis = 0).reset_index(drop = True)
train_data_len = len(train_data) #tutmak istedim
train_data = pd.concat([train_data,test_data],axis = 0).reset_index(drop = True)
train_data.columns[train_data.isnull().any()]
train_data.isnull().sum()
train_data[train_data['Embarked'].isnull()]
train_data.boxplot(column = 'Fare', by = 'Embarked')
plt.show()
train_data['Embarked'] = train_data['Embarked'].fillna('C')
train_data[train_data['Embarked'].isnull()] #Let us check if are there any missing values?
train_data[train_data['Fare'].isnull()]
#np.mean(train_data[train_data['Pclass'] ==3]['Fare'])
train_data['Fare'] = train_data['Fare'].fillna(np.mean(train_data[train_data['Pclass'] ==3]['Fare']))
train_data[train_data['Fare'].isnull()]