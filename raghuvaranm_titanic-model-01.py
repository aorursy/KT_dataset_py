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
#Importing the traing_dataset 

train = pd.read_csv("../input/train.csv")



#Importign the Testing dataset

test = pd.read_csv("../input/test.csv")
#reading the top 5 lines in the data 



train.head()
#reading the last 5 lines in the train dataset

train.tail()
#we can check the dimensions of the dataset



train.shape

print("There are {} rows & {} columns in the train dataset.".format(train.shape[0], train.shape[1]))
train.dtypes #datatypes in the dataset
train.describe() # we can use **describe** function to get the description of the numeric data
import seaborn as  sns

import plotly 

import plotly.offline as pyoff

import plotly.figure_factory as ff

from plotly.offline import init_notebook_mode, iplot, plot # saving file (plot), plotting (iplot)\n",

import plotly.graph_objs as go

import squarify #for tree  maps\n",

%matplotlib inline
import plotly_express as px 

#to check the sum of missing values 

print("The no.of missing values in the traing datset are {}\n".format(train.isnull().sum()))
#we can use the Imputaion methods or fillna methods over here to fill missing values 

from sklearn.preprocessing import Imputer



train['Age'].fillna(method= 'pad', inplace=True)



print("The no.of missing values in Age column after filling missing values {}\n" .format(train['Age'].isnull().sum()))
train.describe()
train.dtypes
train.head()
# Functions that returns the title from a name. All the name in the dataset has the format "Surname, Title. Name"

def get_title(name):

    if '.' in name:

        return name.split(',')[1].split('.')[0].strip()

    else:

        return 'Unknown'



# A list with the all the different titles

titles = sorted(set([x for x in train.Name.map(lambda x: get_title(x))]))

print('Different titles found on the dataset:')

print(len(titles), ':', titles)

print()



# Normalize the titles, returning 'Mr', 'Master', 'Miss' or 'Mrs'

def replace_titles(x):

    title = x['Title']

    if title in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir']:

        return 'Mr'

    elif title in ['the Countess', 'Mme', 'Lady']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title



# Lets create a new column for the titles

train['Title'] = train['Name'].map(lambda x: get_title(x))

# train.Title.value_counts()

# train.Title.value_counts().plot(kind='bar')



# And replace the titles, so the are normalized to 'Mr', 'Miss' and 'Mrs'

train['Title'] = train.apply(replace_titles, axis=1)



# Check that the number of Mr, Mrs and Miss are the same that 'male' and 'female'

print('Title column values. Males and females are the same that for the "Sex" column:')

print(train.Title.value_counts())



# Plot the result

train.Title.value_counts().plot(kind='bar')
train.describe()
cat_cols  = ['Pclass','Sex','SibSp','Parch','Embarked','Title']
def int_to_object(data, col_names):

    for i in col_names:

        data[i].astype(str)

        

    return(data.dtypes)
int_to_object(data=train, col_names= cat_cols) #passing the function created with the columns to convert.



train['Age'].astype('int64')
final_data = pd.get_dummies(data= train, columns=cat_cols, drop_first=True)

final_data.shape
drop_cols = ['PassengerId','Cabin','Name','Ticket']



final = final_data.drop(columns=drop_cols, axis=1, inplace=True)
final_data.describe()
final_data['Age'] = pd.qcut(final_data['Age'], 5)

final_data['Fare'] = pd.cut(final_data['Fare'],4)
final_data.shape
final_data.head()
#splitting into train and test 



x= final_data.loc[:, 1:]