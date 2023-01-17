# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from IPython.display import display, HTML
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import math

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
import os

#grab files
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
#place a column to remember its orgin before combining datasets
train_df['source']='train'
test_df['source']='test'
#combine the training and testing datasets
combine = pd.concat([train_df,test_df],ignore_index=True)
combine.head(10)
print ('Training: '+ str(train_df.shape)) 
print ('Testing: ' + str(test_df.shape))
print ('Combined: '+str(combine.shape))
null_columns=combine.columns[combine.isnull().any()]
combine[null_columns].isnull().sum()
combine.describe()
for column in combine:
    print (str(column) + ": " + str(len(combine[column].unique())))
#combine.apply(lambda x: len(x.unique()))
#Filter categorical variables
categorical_columns = [x for x in combine.dtypes.index if combine.dtypes[x]=='object']
#Exclude ID cols and source:
categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier','source']]
#Print frequency of categories
for col in categorical_columns:
    #print ('\nFrequency of Categories for variable %s'%col)
    combine[col].value_counts().plot(kind='barh')
    plt.title('Frequency of Categories for %s'%col)
    plt.show()
print ('Before Number of Missing Rows: %d'% sum(combine['Item_Weight'].isnull()))
combine["Item_Weight"] = combine.groupby("Item_Identifier").transform(lambda x: x.fillna(x.mean()))
print ('After Number of Missing Rows: %d'% sum(combine['Item_Weight'].isnull()))
#Import mode function:
from scipy.stats import mode
print ('Before Number of Missing Rows: %d'% sum(combine['Outlet_Size'].isnull()))
combine['Outlet_Size'] = combine.groupby('Outlet_Type').transform(lambda x: x.fillna(mode(x).mode[0]))
print ('After Number of Missing Rows: %d'%sum(combine['Outlet_Size'].isnull()))
avg_visibility= combine.groupby('Item_Identifier')['Item_Visibility'].mean()
is_bool = (combine['Item_Visibility'] == 0)
print ('Number of 0 values initially: %d'%sum(is_bool))
combine.loc[is_bool,'Item_Visibility'] = combine.loc[is_bool,'Item_Identifier'].apply(lambda x: avg_visibility.loc[x])
print ('Number of 0 values after: %d'%sum(combine['Item_Visibility'] == 0))
combine['Item_Visibility_MeanRatio'] = combine.apply(lambda x: x.loc['Item_Visibility']/avg_visibility.loc[x.loc['Item_Identifier']], axis=1)
print (combine['Item_Visibility_MeanRatio'].describe())
# avg_MRP= combine.groupby('Outlet_Location_Type')['Item_MRP'].mean()
# combine['Item_MRP_MeanRatio'] = combine.apply(lambda x: x.loc['Item_MRP']/avg_MRP.loc[x.loc['Outlet_Location_Type']], axis=1)
# print (combine['Item_MRP_MeanRatio'].describe())
#Get the first two characters of ID:
combine['Item_Type_Combined'] = combine['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
combine['Item_Type_Combined'] = combine['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
print (combine['Item_Type_Combined'].value_counts().to_string())
combine['Item_Type_Combined'].value_counts().plot(kind='barh')
plt.show()
#Years:
combine['Outlet_Years'] = 2013 - combine['Outlet_Establishment_Year']
combine['Outlet_Years'].describe()
#Change categories of low fat:
print ('Original Categories:')
print (combine['Item_Fat_Content'].value_counts())

print ( '\nModified Categories:')
combine['Item_Fat_Content'] = combine['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})
print (combine['Item_Fat_Content'].value_counts())
combine.head()
#Mark non-consumables as separate category in low_fat:
combine.loc[combine['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
combine['Item_Fat_Content'].value_counts()
#Import library:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#New variable for outlet
combine['Outlet'] = le.fit_transform(combine['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
le = LabelEncoder()
for i in var_mod:
    combine[i] = le.fit_transform(combine[i])

#One Hot Coding:
combine = pd.get_dummies(combine, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                              'Item_Type_Combined','Outlet'])
combine.dtypes
combine[['Item_Fat_Content_0','Item_Fat_Content_1','Item_Fat_Content_2']].head(10)
#Drop the columns which have been converted to different types:
combine.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

#Divide into test and train:
train = combine.loc[combine['source']=="train"]
test = combine.loc[combine['source']=="test"]

#Drop unnecessary columns:
test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)

#Export files as modified versions:
train.to_csv("train_modified.csv",index=False)
test.to_csv("test_modified.csv",index=False)

train.head()
test.head()
#Mean based:
mean_sales = train['Item_Outlet_Sales'].mean()

#Define a dataframe with IDs for submission:
base1 = test[['Item_Identifier','Outlet_Identifier']]
base1['Item_Outlet_Sales'] = mean_sales

print (base1.head(10))

#Export submission file
base1.to_csv("alg0.csv",index=False)
#Define target and ID columns:
target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']
from sklearn import model_selection, metrics
def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    #Perform cross-validation:
    cv_score = model_selection.cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20, scoring='neg_mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    #Print model report:
    print ("\nModel Report")
    print ("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
    print ("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    
    #Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])
    
    #Export submission file:
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    print(submission.head(5))
    submission.to_csv(filename, index=False)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
predictors = [x for x in train.columns if x not in [target]+IDcol]
# print predictors
alg1 = LinearRegression(normalize=True)
modelfit(alg1, train, test, predictors, target, IDcol, 'alg1.csv')
coef1 = pd.Series(alg1.coef_, predictors).sort_values()
coef1.plot(kind='bar', title='Model Coefficients')