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
# data analysis and wrangling
import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier , VotingClassifier , AdaBoostClassifier , ExtraTreesClassifier , GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score , StratifiedKFold , GridSearchCV , learning_curve
from xgboost.sklearn import XGBClassifier

train_df= pd.read_csv('../input/train.csv',parse_dates=['Dates'])
test_df= pd.read_csv('../input/test.csv',parse_dates=['Dates'])
train_len=len(train_df)
# Category is the target variable
Y_train=train_df['Category']
train_len
# We will concatenate the train and test datasets , train then test below it , i.e 
# concatenating along axis=0.When we reset the index, the old index is added as a column 
# but drop=True will drop in finally.
# and a new sequential index is used. Even though the data will be sort of clubbed , 
# but we can use the train_len to distinguish the train from test.

dataset=pd.concat(objs=[train_df,test_df],axis=0).reset_index(drop=True)
dataset.head()
# Category is the TARGET Variable
len(Y_train.unique())
# Visualise the Category count in the Training Set

#sns.countplot(x='Category',data=train_df,order=)
# ID Variable from Test Set
IDtest=test_df['Id']

# Target Variable : Category

# Character / Object Type : (Categorical (Ordinal , Nominal))
train_df.describe(include=['O'])
# Dates,Category,Descript , DayofWeek , PDDistrict, Resolution, Address

# Numeric Type  (Continuous , Discrete)
# X  , Y (Both Continuous)

train_df.describe()
train_df.isnull().sum()
# No data is missing
dataset.isnull().sum()
dataset=dataset.drop(['Descript','Resolution'],axis=1)
dataset.head(1)
# Process DATES 
#dataset['Year']=dataset['Dates'].dt.year
dataset['Month']=dataset['Dates'].dt.month
#dataset['Day']=dataset['Dates'].dt.day
dataset['HourofDay']=dataset['Dates'].dt.hour

# DATES can be dropped
dataset=dataset.drop(['Dates'], axis=1)
# Deriving results by referring visualistions from 
# https://www.kaggle.com/keldibek/sf-crime-visualization
# Analysing Hour of Day vs Crime we find that the day can be divided into :
# 1-7 : Morning , 18-24 Evening, 7-18 Day 
dataset['Month']=dataset['Month'].apply(lambda x : 'MonthLow' if x== 12 else ('MonthMed' if x in (2,6,7,8,9,11) else 'MonthHigh'))
dataset['HourofDay']=dataset['HourofDay'].apply(lambda x : 'Morning' if (x >= 1) & (x <=7) else ('Day' if x > 7 & x <=18 else 'Evening'))

dataset.head()
# Replacing DayOfWeek by corresponding Numeric Value
week_dict={
    "Monday":1,
    "Tuesday":2,
    "Wednesday":3,
    "Thursday":4,
    "Friday":5,
    "Saturday":6,
    "Sunday":7
}
dataset['DayOfWeek']=dataset['DayOfWeek'].replace(week_dict)
dataset['DayOfWeek']= dataset['DayOfWeek'].apply(lambda x : 'WeekHigh' if x in (3,5) else ('WeekMed' if x in (2,4,6) else 'WeekLow'))
dataset.head()
dataset['PdDistrict'].unique()
# It can be One Hot Encoded
dataset['Address'].head()
#dataset['StreetNo']=dataset['Address'].apply(lambda x : x.split(' ',1)[0] if x.split(' ',1)[0].isdigit() else 0)
dataset['Intersection']=dataset['Address'].apply(lambda x : 1 if '/' in x else 0)
dataset['Block']=dataset['Address'].apply(lambda x : 1 if 'Block' in x else 0)
dataset['StreetSuffix']=dataset['Address'].apply(lambda x : x.split(' ')[-1] if len(x.split(' ')[-1])==2 else 0) 


dataset=dataset.drop(['Address'],axis=1)
# Longitude [-122.5247, -122.3366]
print('Longitude')
print(dataset['X'].min())
print(dataset['X'].max())

# Lattitude [37.699, 37.8299]
print('Lattitude')
print(dataset['Y'].min())
print(dataset['Y'].max())
dataset.head()
#dataset.to_excel('to_visualize.xlsx')
# # Yet to form a boundary for San Francisco and analyse outliers
# dataset[(dataset['X'] < -122.5) & (dataset['X'] > -122.3)
#         & (dataset['Y'] < 37.6) & (dataset['Y'] > 37.8) ]
dataset['Y']=dataset['Y'].apply(lambda x : x if 37.82 > x else 37.82 )
dataset['X']=dataset['X'].apply(lambda x : x if -122.3 > x else -122.3 )
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
from sklearn.cluster import KMeans
dataXY=dataset[['X','Y']]
plt.scatter(dataXY['X'],dataXY['Y'], )
kmeans=KMeans(n_clusters=10)
kmeans.fit(dataXY)
dataset['LocCluster']=kmeans.labels_
dataset=dataset.drop(['X','Y'], axis=1)
dataset.head()
# Yet to add implicit feature simultaneous crime
dataset=dataset.drop(['Category','Id'], axis=1)
dataset.head()
dataset.describe()
#  Intersection , Block , LocCluster
# Even though they are of numeric type , but we will treat them as categorical.
# Hence explicitly mention them in the pd get dummies
dataset=pd.get_dummies(dataset)
dataset.head()
# Intersection , Block , LocCluster
dataset=pd.get_dummies(dataset,prefix=['Intersection' , 'Block', 'LocCluster' ],columns=['Intersection' , 'Block', 'LocCluster' ])
dataset.head()
# Separate train and test data
X_train = dataset[:train_len]
X_test = dataset[train_len:]
print(X_train.shape)
print(X_test.shape)

 # LR
    
lr_classifier=LogisticRegression()
# lr_param_grid = [{'penalty':['l1','l2'] , 'C':[1]}]
# gs_LR = GridSearchCV(estimator = lr_classifier,
#                            param_grid = lr_param_grid,
#                            scoring = 'accuracy',
#                            cv = kfold,
#                            n_jobs = -1)

lr_classifier=lr_classifier.fit(X_train,Y_train)
# gs_LR = gs_LR.fit(X_train, Y_train)
# lr_best_params = gs_LR.best_params_
# lr_best_score = gs_LR.best_score_
# lr_best=gs_LR.best_estimator_
# lr_best_score

y_pred=lr_classifier.predict(X_test)
cat=pd.DataFrame(train_df.Category.unique()).sort_values(by=[0]).reset_index().drop(['index'],axis=1)[0].to_dict()
submit = pd.DataFrame({'Id': IDtest})
submit.head(10)
for key,value in cat.items():
        submit[value]=0
for item in y_pred:
    for key,value in cat.items():
        if (item==value):
            submit[value][key]=1
            
            
    
submit
y_pred
cat.items()

# #submit
# submit = pd.DataFrame({'Id': IDtest})
# y_pred_Category = pd.Series(y_pred, name="Category")
# y_pred_Category_t #10
# submit_t
# y_pred_t=y_pred[]
# # TESTING
# len(y_pred_t)
# #submit_t
# #for cat in y_pred_t:
# #    print(cat)

# for category in y_pred_Category:
#     submit[category]=np.where(y_pred==category,1,0)
# # for category in y.cat.categories:
# #     submit[category] = np.where(outcomes == category, 1, 0

#results = pd.concat([IDtest,y_pred_Category],axis=1)

submit.to_csv("sf_lr.csv",index=False)

