#Warnings
import warnings
warnings.filterwarnings('ignore', category = DeprecationWarning)
warnings.filterwarnings('ignore', category = FutureWarning)

#Importing modules
import numpy as np
import pandas as pd
import re
import xgboost as xgb
import math

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
%matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont

import seaborn as sns
import missingno as mn
from scipy import stats


#Print availabel data
import os
print(os.listdir("../input"))
#Set default plot
plt.style.use('bmh')
sns.set_style({'axis.grid' :False})

from IPython.display import Markdown
def bold(string):
    display(Markdown(string))

#Import train data
train = pd.read_csv('../input/train.csv')
bold('**Preview of train data**:')
display(train.head(2))

#Import test data
test = pd.read_csv('../input/test.csv')
bold('**Preview of test data**:')
display(test.head(2))

#test passenger IDs
PassengerId = test['PassengerId']

#What is 'Gender Submission'?
Gen_Sub = pd.read_csv('../input/gender_submission.csv')
bold('**Preview Gender Submission data**')
display(Gen_Sub.head(2))
#Merge sets together: head, shape, variable names
merged = pd.concat([train, test], sort = False)

bold('**Merged head:**')
display(merged.head(2))

bold('**Merged shape:**')
display(merged.shape)

bold('**Variable Names:**')
display(merged.columns)
age_slice = merged["Age"]
fare_slice = merged["Fare"]
pclass_slice = merged["Pclass"]
# Print descriptions of columsn
for col_name in train.columns:
    print(col_name)
    #print(merged[col_name].describe())
    #print("\n")
#Convert text fields to boolean where appropriate, and drop the text field that is replaced.
#Define columns to concat & drop.
train_bool = pd.concat([train, pd.get_dummies(train.Sex).rename(columns = "{}_binary".format)], axis=1)
train_bool = pd.concat([train_bool, pd.get_dummies(train_bool.Embarked).rename(columns = "{}_binary".format)], axis=1)
# print(train_bool.head(5))

print(list(train_bool.columns.values))

#Numerical Columns
cols_list = [1,2,5,9,13]
train_numerical = train_bool.iloc[:,cols_list]
print(type(train_numerical))
print(train_numerical.shape)

#Drop NaN Ages
train_numerical = train_numerical[np.isfinite(train_numerical['Age'])]
print(train_numerical.head())

graphs=sns.pairplot(train_numerical)
#Box plots for numeric variables vs. Survival
iter_cols = list(train_numerical.columns.values)

for item in iter_cols:
    if item != 'Survived':
        plt.figure()
        sns.boxplot(train_numerical['Survived'], train_numerical[item])
# search for cross correlation
corr = train_numerical.corr()
plt.matshow(train_numerical.corr())
corr.style.background_gradient()
plt.figure()
sns.heatmap(train_numerical.corr())

#Survived vs died
plt.hist(train_numerical['Survived'])
plt.show()

#Survival odds: overall
total_survived = train_numerical['Survived'].sum()
total_count = train_numerical['Survived'].count()
odds = total_survived/total_count
print("Overall odds of survival = " + str(odds))

#Male survival odds
male = train_numerical["male_binary"]==1
male_train = train_numerical[male]
male_survived = male_train['Survived'].sum()
male_total_count = male_train['Survived'].count()
male_odds = male_survived/male_total_count
print("Male odds of survival = " + str(male_odds))

#Female survival odds
female = train_numerical["male_binary"]==0
female_train = train_numerical[female]
female_survived = female_train['Survived'].sum()
female_total_count = female_train['Survived'].count()
female_odds = female_survived/female_total_count
print("Female odds of survival = "+ str(female_odds))

#Pivot Tables
sex_pivot = train.pivot_table(index="Sex", values="Survived")
sex_pivot.plot.bar()
plt.show()

Pclass_pivot = train.pivot_table(index="Pclass", values ="Survived")
Pclass_pivot.plot.bar()
plt.show()

class_sex_pivot = train.pivot_table(index=["Pclass", "Sex"], values="Survived")
class_sex_pivot.plot.bar()
plt.show()
                                           
#Age histograms
survived = train[train["Survived"] ==1]
died = train[train["Survived"] ==0]
survived["Age"].plot.hist(alpha=0.5,color='red',bins=50)
died["Age"].plot.hist(alpha=0.5,color='blue',bins=50)
plt.legend(["Survived","Died"])
plt.show()


#bins
#Age into bins
def process_age(df, cut_points, label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"], cut_points, labels = label_names)
    return df

cut_points = [-1,0,5,12,18,35,60,100]
label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]

train=process_age(train, cut_points, label_names)
test=process_age(test, cut_points, label_names)

age_pivot = train.pivot_table(index="Age_categories", values="Survived")
age_pivot.plot.bar()
plt.show()

#family size into bins

#Group Parch & SibSp into family
for row in train:
    train['family']=train['Parch']+train['SibSp']
    
for row in test:
    test['family']=test['Parch']+test['SibSp']
    
#print(train['family'].head())
print(train['family'].describe())
print(test['family'].describe())

#into bins
def process_famsize(df, cut_points, label_names):
    df["family_cat"] = pd.cut(df["family"], cut_points2, labels = label_names2)
    return df

cut_points2 = [-0.1,0.1,1.1,2.1,100]
label_names2 = ["Single","Double","Triple", "MoreFam"]

train=process_famsize(train, cut_points2, label_names2)
test=process_famsize(test, cut_points2, label_names2)

family_pivot = train.pivot_table(index="family_cat", values="Survived")
family_pivot.plot.bar()
plt.show()

#drop age & SibSp & Parch
to_drop = ['Parch', 'SibSp', 'family','Age']
train = train.drop(to_drop, axis=1)
test = test.drop(to_drop, axis=1)

list(train)
list(test)


bold('**Preview of train data**:')
display(train.head(2))
#Get dummmies for all relevant columns
def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df

for column in ["Pclass", "Sex", "Age_categories", "family_cat"]:
    train = create_dummies(train, column)
    test = create_dummies(test, column)

print(list(train.columns.values))
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

#this is the Kaggle test data
holdout = test
from sklearn.model_selection import train_test_split
columns_mk1 = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'Age_categories_Missing', 'Age_categories_Infant', 'Age_categories_Child', 'Age_categories_Teenager', 'Age_categories_Young Adult', 'Age_categories_Adult', 'Age_categories_Senior', 'family_cat_Single', 'family_cat_Double', 'family_cat_Triple', 'family_cat_MoreFam']
lr.fit(train[columns_mk1], train['Survived'])

all_X = train[columns_mk1]
all_y = train['Survived']

train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.2, random_state = 0)

lr.fit(train_X, train_y)
predictions = lr.predict(test_X)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_y, predictions)

print(accuracy)
#feature engineering:
#deweight family_size

def weight_columns(col, weight):
    for row in column:
        train[col]=weight*train[col]
        test[col]=weight*test[col]
    return

cols_deweight = ['family_cat_Single', 'family_cat_Double', 'family_cat_Triple', 'family_cat_MoreFam']
weighting = 0.5
weight_columns(cols_deweight,weighting)


#deweighted family size
lr.fit(train[columns_mk1], train['Survived'])

all_X = train[columns_mk1]
all_y = train['Survived']

train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.2, random_state = 0)

lr.fit(train_X, train_y)
predictions = lr.predict(test_X)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_y, predictions)

print(accuracy)

#feature engineering - deweight child teenager young_adult adult

#cols_deweight_2 = ['Age_categories_Child', 'Age_categories_Teenager', 'Age_categories_Young Adult', 'Age_categories_Adult']
#weighting2 = 0.5
#weight_columns(cols_deweight_2,weighting2)
#deweighted family size + some age bins
#lr.fit(train[columns_mk1], train['Survived'])

#all_X = train[columns_mk1]
#all_y = train['Survived']

#train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.2, random_state = 0)

#lr.fit(train_X, train_y)
#predictions = lr.predict(test_X)

#from sklearn.metrics import accuracy_score
#accuracy = accuracy_score(test_y, predictions)

#print(accuracy)
Survived = lr.predict(holdout[columns_mk1])
PassengerID = holdout['PassengerId']
#submission = final_test[['PassengerId','Survived']]

#submission.to_csv("submission.csv", index=False)
submission2 = pd.DataFrame([PassengerID, Survived])
submission2 = submission2.transpose()
submission2.rename(columns={'Unnamed 0':'Survived'}, 
                 inplace=True)
display(submission2.head(10))
submission2.to_csv("submission2.csv", index=False)
submission2.tail()
