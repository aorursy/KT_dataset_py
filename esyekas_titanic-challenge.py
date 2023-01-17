# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

#load packages
import sys #access to system parameters https://docs.python.org/3/library/sys.html
print("Python version: {}". format(sys.version))

import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
print("pandas version: {}". format(pd.__version__))

import matplotlib #collection of functions for scientific and publication-ready visualization
print("matplotlib version: {}". format(matplotlib.__version__))

import numpy as np #foundational package for scientific computing
print("NumPy version: {}". format(np.__version__))

import scipy as sp #collection of functions for scientific computing and advance mathematics
print("SciPy version: {}". format(sp.__version__)) 

import IPython
from IPython import display #pretty printing of dataframes in Jupyter notebook
print("IPython version: {}". format(IPython.__version__)) 


# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Configure visualisations
%matplotlib inline
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6


#misc libraries
import random
import time


#ignore warnings
import warnings
warnings.filterwarnings('ignore')
print('-'*25)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
files = os.listdir('../input')
for file in files:
    print(file)

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]

full = train_df.append( test_df , ignore_index = True )
titanic = full[ :891 ]

print ('Datasets:' , 'full:' , full.shape , 'titanic:' , titanic.shape)
print(train_df.columns.values)
# preview the data
train_df.head()


titanic.head()
train_df.tail()
train_df.info()
print('_'*40)
test_df.info()
train_df.describe()
# Review survived rate using `percentiles=[.61, .62]` knowing our problem description mentions 38% survival rate.
# Review Parch distribution using `percentiles=[.75, .8]`
# SibSp distribution `[.68, .69]`
# Age and Fare `[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]`
#Names are unique across the dataset (count=unique=891)
#Sex variable as two possible values with 65% male (top=male, freq=577/count=891).
#Cabin values have several dupicates across samples. Alternatively several passengers shared a cabin.
#Embarked takes three possible values. S port used by most passengers (top=S)
#Ticket feature has high ratio (22%) of duplicate values (unique=681).
train_df.describe(include=['O'])
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Transform Sex into binary values 0 and 1
sex = pd.Series( np.where( full.Sex == 'male' , 1 , 0 ) , name = 'Sex' )
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
titanic.head()
sex_pivot = train_df.pivot_table(index = "Sex", values = "Survived")
sex_pivot.plot.bar()
plt.show()
class_pivot = train_df.pivot_table(index="Pclass",values="Survived")
class_pivot.plot.bar()
plt.show()
#the proportion of people in first class that survived is much higher
#TO DO Include the number of people on each group that survived
siblins_plot = train_df.pivot_table(index="SibSp",values="Survived")
siblins_plot.plot.bar()
plt.show()
#people with 1 sibling or spouse had a higher survival rate than the rest.
#Bucketing the population into segments
#This allows to make a story 
def process_age(df,cut_points,label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

cut_points = [-1,0,5,12,18,35,60,100]
label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]

train_df = process_age(train_df,cut_points,label_names)
test_df = process_age(test_df,cut_points,label_names)
train_df.head()
age_category_plot = train_df.pivot_table(index="Age_categories",values='Survived')
age_category_plot.plot.bar()
plt.show()
#Infants had the highest survival rate
train_df["Pclass"].value_counts()
def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis = 1)
    return df

for column in ["Pclass","Sex","Age_categories"]:
    train_df = create_dummies(train_df,column)
    test_df = create_dummies(test_df,column)    
    
train_df.head()
list(train_df)
#Getting only engineered columns
model_columns = [ 'Pclass_1',
 'Pclass_2',
 'Pclass_3',
 'Sex_female',
 'Sex_male',
 'Age_categories_Missing',
 'Age_categories_Infant',
 'Age_categories_Child',
 'Age_categories_Teenager',
 'Age_categories_Young Adult',
 'Age_categories_Adult',
 'Age_categories_Senior',
 'Pclass_1',
 'Pclass_2',
 'Pclass_3',
 'Sex_female',
 'Sex_male',
 'Age_categories_Missing',
 'Age_categories_Infant',
 'Age_categories_Child',
 'Age_categories_Teenager',
 'Age_categories_Young Adult',
 'Age_categories_Adult',
 'Age_categories_Senior']

lr = LogisticRegression()
lr.fit(train_df[model_columns],train_df["Survived"])
#Dividing now the train datafrain between train and test set (Since the given test df doesn't have the target column)
all_X = train_df[model_columns]
all_y = train_df['Survived']
train_X, test_X, train_y, test_y = train_test_split(all_X,all_y, test_size = 0.20, random_state = 0)
test_X.describe()
#training the data in the training set and predicting in the test set
lr = LogisticRegression()
lr.fit(train_X,train_y)
predictions = lr.predict(test_X)
#measuring accuracy of predictions
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_y,predictions)
print(accuracy)
#Performs Cross Validation on different training and test sets
from sklearn.model_selection import cross_val_score
lr = LogisticRegression()
scores = cross_val_score(lr,all_X,all_y,cv = 10)
scores.sort()
accuracy = scores.mean()
print(scores)
print(accuracy)
lr.fit(all_X,all_y)
holdout_predictions = lr.predict(test_df[model_columns])
print('Hello')
submission_df = {"PassengerId": test_df["PassengerId"],
                              "Survived":holdout_predictions}
submission = pd.DataFrame(submission_df)
submission.head()
submission.to_csv("submission.csv",index=False)
