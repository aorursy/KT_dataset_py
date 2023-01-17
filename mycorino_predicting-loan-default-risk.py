# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import time
import csv
import traceback
import pandas as pd
import numpy as np
import dask.dataframe as dd #reading large datasets out of memory

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#import the files with appropriate encoding

application = pd.read_csv("../input/application_train.csv")
bureau = pd.read_csv("../input/bureau.csv")
bureau_balance = pd.read_csv("../input/bureau_balance.csv")
credit_card_balance = pd.read_csv("../input/credit_card_balance.csv",encoding='utf-8')
HomeCredit_col = pd.read_csv("../input/HomeCredit_columns_description.csv",encoding='latin-1')
#installments_payments = pd.read_csv("../input/installments_payments.csv")
POS_CASH_balance = pd.read_csv("../input/POS_CASH_balance.csv")
previous_application = pd.read_csv("../input/previous_application.csv")
application_test = pd.read_csv("../input/application_test.csv")
application.head()
application_test.head(2)
HomeCredit_col.head(219) #The description file !!!
def get_description(keyword_list): #Let's make a function that returns the description for selected items only
    
    description = HomeCredit_col[['Row','Table','Description']]   
    df = description.loc[description['Row'].str.contains(keyword_list[0], case=False)]
    for i in range(1, len(keyword_list)):
        df = df.append(description.loc[description['Row'].str.contains(keyword_list[i], case=False)])
    return(df)
list(application.columns.values) #Let's see whats in this first application file...
#These items sound the most interesting

keyword_list_application = ['SK_ID_CURR','TARGET','CODE_GENDER','DAYS_EMPLOYED','CNT_CHILDREN',
                                  'DAYS_BIRTH','AMT_INCOME_TOTAL','NAME_INCOME_TYPE',
                                  'NAME_HOUSING_TYPE','DAYS_LAST_PHONE_CHANGE']
#Check if the description matches with what we thought it would be

get_description(keyword_list_application)
keyword_list_bureau = ['SK_ID_CURR', 'CREDIT_ACTIVE','AMT_CREDIT_SUM', 'CREDIT_DAY_OVERDUE']
get_description(keyword_list_bureau)
selected_col_bureau = bureau[keyword_list_bureau]
#create a new dataframe with the selected features only
selected_col_application = application[keyword_list_application] 
list(previous_application.columns.values)
keyword_list_previous_application = ['SK_ID_CURR', 'RATE_INTEREST_PRIMARY', 'AMT_CREDIT','CODE_REJECT_REASON' ]
get_description(keyword_list_previous_application)
selected_col_prev_app = previous_application[keyword_list_previous_application]
merged = selected_col_application.merge(selected_col_bureau, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='inner')
merged = merged.merge(selected_col_prev_app, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='inner')
merged = merged.drop_duplicates('SK_ID_CURR', keep='first')

merged.head()
merged['Age'] = merged['DAYS_BIRTH']/-365
merged['Phone_age'] = merged['DAYS_LAST_PHONE_CHANGE']/-365
merged.drop('DAYS_LAST_PHONE_CHANGE', axis = 1)
merged.drop('DAYS_BIRTH', axis = 1)

merged.head()
merged.describe()
merged['Family_size'] = 0 
merged.loc[merged['CNT_CHILDREN'] > 0 , 'Family_size'] = 1 #Normal
merged.loc[merged['CNT_CHILDREN'] > 3 , 'Family_size'] = 2 #Large
merged['Income/credit'] = merged['AMT_INCOME_TOTAL']/merged['AMT_CREDIT_SUM']
merged['Income_category'] = "Low"
merged.loc[merged['AMT_INCOME_TOTAL'] > 112500 , 'Income_category'] = "Below_average"
merged.loc[merged['AMT_INCOME_TOTAL'] > 157500 , 'Income_category'] = "Above_average"
merged.loc[merged['AMT_INCOME_TOTAL'] > 202500 , 'Income_category'] = "High"
merged.reset_index(drop = True)
merged.head()
merged.describe()
frequency = merged.describe()['TARGET'][1] # since we have only zeros and ones the mean is the frequency
print("Frequency of Target = 1 is : ", frequency)
merged[['Income_category', 'TARGET']].groupby(['Income_category'], as_index=False).mean().sort_values(by='TARGET', ascending=False)
merged[['CODE_GENDER', 'TARGET']].groupby(['CODE_GENDER'], as_index=False).mean().sort_values(by='TARGET', ascending=False)
merged[['NAME_INCOME_TYPE', 'TARGET']].groupby(['NAME_INCOME_TYPE'], as_index=False).mean().sort_values(by='TARGET', ascending=False)
merged[['NAME_HOUSING_TYPE', 'TARGET']].groupby(['NAME_HOUSING_TYPE'], as_index=False).mean().sort_values(by='TARGET', ascending=False)
merged[['CNT_CHILDREN', 'TARGET']].groupby(['CNT_CHILDREN'], as_index=False).mean().sort_values(by='TARGET', ascending=False)
merged[['CODE_REJECT_REASON', 'TARGET']].groupby(['CODE_REJECT_REASON'], as_index=False).mean().sort_values(by='TARGET', ascending=False)
g = sns.FacetGrid(merged, col='TARGET')
g.map(plt.hist, 'Age', bins=20)
g = sns.FacetGrid(merged, col='TARGET')
g.map(plt.hist, 'Phone_age', bins=20)
grid = sns.FacetGrid(merged, col='TARGET', row='CODE_GENDER', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
grid = sns.FacetGrid(merged, row='Income_category', size=2.2, aspect=5)
grid.map(sns.pointplot, 'NAME_HOUSING_TYPE', 'TARGET', 'CODE_GENDER', palette='deep')
grid.add_legend()
grid = sns.FacetGrid(merged, row='Family_size', col='TARGET', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'CODE_GENDER', 'AMT_CREDIT_SUM', alpha=.5, ci=None)
grid.add_legend()
import os
import csv
import traceback
import shutil
import tensorflow as tf
from tensorflow import keras


from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
def scale(feature):
    min_x = merged.min()[feature]
    max_x = merged.max()[feature]
    
    merged[feature] = (merged[feature] - min_x) / (max_x - min_x)


scale('DAYS_EMPLOYED')
scale('CNT_CHILDREN')
scale('AMT_INCOME_TOTAL')
scale('DAYS_LAST_PHONE_CHANGE')
scale('AMT_CREDIT_SUM')
scale('CREDIT_DAY_OVERDUE')
scale('RATE_INTEREST_PRIMARY')
scale('AMT_CREDIT')
scale('Age')
scale('Family_size')
scale('Income/credit')
scale('Phone_age')
scale('DAYS_BIRTH')
cat_columns = merged.select_dtypes(['object']).columns
cat_columns
merged[cat_columns] = merged[cat_columns].astype('category')

merged['NAME_HOUSING_TYPE'] = merged['NAME_HOUSING_TYPE'].cat.codes
merged['NAME_INCOME_TYPE'] = merged['NAME_INCOME_TYPE'].cat.codes
merged['CODE_GENDER'] = merged['CODE_GENDER'].cat.codes
merged['CREDIT_ACTIVE'] = merged['CREDIT_ACTIVE'].cat.codes
merged['CODE_REJECT_REASON'] = merged['CODE_REJECT_REASON'].cat.codes
merged['Income_category'] = merged['Income_category'].cat.codes
merged.set_index('SK_ID_CURR', inplace=True)
merged = merged.dropna() #dropping all null values
#merged = merged.reset_index(drop = True)
merged = merged.drop('RATE_INTEREST_PRIMARY',axis = 1)
np.random.seed(seed=1) #makes result reproducible
msk = np.random.rand(len(merged)) < 0.8
traindf = merged[msk]
evaldf = merged[~msk]
X_train = traindf.drop('TARGET', axis = 1)

Y_train = traindf['TARGET']

X_test = evaldf.drop('TARGET', axis = 1)
Y_test = evaldf['TARGET']
X_train.shape, Y_train.shape, X_test.shape
#KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn

acc_knn_test = round(knn.score(X_test, Y_test) * 100, 2)
acc_knn_test

print('Train:',acc_knn, 'Test:', acc_knn_test)
#Support vector machine

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc

acc_svc_test = round(svc.score(X_test, Y_test) * 100, 2)
acc_svc_test

print('Train:',acc_svc, 'Test:', acc_svc_test)


#random_forest

random_forest = RandomForestClassifier(n_estimators=120)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest

acc_random_forest_test = round(random_forest.score(X_test, Y_test) * 100, 2)
acc_random_forest_test

print('Train:',acc_random_forest, 'Test:', acc_random_forest_test)
# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree_Test = round(decision_tree.score(X_test, Y_test) * 100, 2)
acc_decision_tree_Test

print('Train:',acc_decision_tree, 'Test:', acc_decision_tree_Test)
submission = pd.DataFrame({
        "Real": evaldf["TARGET"],
        "Prediction": Y_pred
    })
submission.head()
submission.describe()
