# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sys

import os

import matplotlib.pyplot as plt

import math

from numpy import array

from numpy import argmax

from keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder

np.random.seed(17154016)

from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC,SVR

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis

from  more_itertools import unique_everseen

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold



from sklearn import preprocessing

from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor

import lightgbm as lgb

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv("/kaggle/input/use-this/attrition.csv")
print("shape of the given data :",df.shape)

print('features in given data  :',df.columns)

df.head()
df1=df.drop(['S.No','Emp Name','DOJ','In Active Date'],axis=1)

df1=df1.set_index('EmpID')

df1.head()
new_df=pd.read_csv("/kaggle/input/ibm-data/ibm.csv")

new_df.columns
new_df=new_df.drop(['BusinessTravel','DailyRate','Department','DistanceFromHome','EducationField','EmployeeCount','EmployeeNumber','EnvironmentSatisfaction','HourlyRate','JobInvolvement','MonthlyRate', 'NumCompaniesWorked',

       'Over18','OverTime', 'PercentSalaryHike','RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',

       'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance','YearsSinceLastPromotion', 'YearsWithCurrManager','YearsInCurrentRole'],axis=1)
new_df=new_df.rename(columns={"JobLevel": "Grade", "JobRole": "Designation","MaritalStatus":"Marital Status","PerformanceRating":"Last Rating","MonthlyIncome":"Monthly Income","YearsAtCompany":"Tenure","JobSatisfaction":"Engagement Score (% Satisfaction)"})

new_df =new_df[['Designation', 'Grade', 'Tenure', 'Gender', 'Education','Age', 'Last Rating', 'Monthly Income','Engagement Score (% Satisfaction)', 'Marital Status','Attrition']]

new_df.head()
df1_new=df1.drop(['Location','Zone','Remarks'],axis=1)

df1_new.head()
print("ibm dataset features :",new_df.columns)

print("bitgrit dataset features :",df1_new.columns)
def gender(str):

    if str=='Male':

        return 1

    else:

        return 0

def marital_status(str):

    if str=='Married':

        return 0

    if str=='Divorced':

        return 1

    else:

        return 2

def attrition(str):

    if str=='Yes':

        return 1

    else:

        return 0

def Grade(str):

    if str=='E1':

        return 1

    if str=='E2':

        return 2

    if str=='M1':

         return 3

    if str=='M2':

        return 4

    if str=='M3':

        return 5

    if str=='M4':

         return 6

    else:

        return 7

def edu(str):

    if str=='Bachelors':

        return 3

    else:

        return 4

def per(str):

    return float(str.replace('%',''))
df1_new['Gender']=df1_new['Gender'].apply(gender)

new_df['Gender']=new_df['Gender'].apply(gender)

df1_new['Marital Status']=df1_new['Marital Status'].apply(marital_status)

new_df['Marital Status']=new_df['Marital Status'].apply(marital_status)

df1_new['Attrition']=df1_new['Attrition'].apply(attrition)

new_df['Attrition']=new_df['Attrition'].apply(attrition)

df1_new['Grade']=df1_new['Grade'].apply(Grade)

df1_new['Education']=df1_new['Education'].apply(edu)
df1_new['Engagement Score (% Satisfaction)']=df1_new['Engagement Score (% Satisfaction)'].apply(per)

df1_new.head()
new_df['Engagement Score (% Satisfaction)']=(95/4)*new_df['Engagement Score (% Satisfaction)']

new_df.head()
print("bitgrit data shape:"+str(df1_new.shape))

print("ibm_dataset:"+str(new_df.shape))
combine=pd.concat(([df1_new,new_df]))

y=combine['Attrition']

combine=combine.drop(['Attrition'],axis=1)

print("shape of combined data:"+str(combine.shape))

combine.head()
from sklearn.preprocessing import LabelEncoder

labelencoder=LabelEncoder()

c=combine.values 

c[:, 0] = labelencoder.fit_transform(c[:, 0])

combine=pd.DataFrame(c)

combine.columns=['Designation', 'Grade', 'Tenure', 'Gender', 'Education', 'Age',

       'Last Rating', 'Monthly Income', 'Engagement Score (% Satisfaction)',

       'Marital Status']

combine.head()
# one hot encode

a= pd.DataFrame(to_categorical(combine['Gender']))

b= pd.DataFrame(to_categorical(combine['Designation']))

c= pd.DataFrame(to_categorical(combine['Marital Status']))

join=pd.concat([a,b,c],axis=1)
combine_dr=combine.drop(['Gender','Designation','Marital Status','Tenure'],axis=1)

train=pd.concat([combine_dr,join],axis=1)
print("shape of combine_dr",combine_dr.shape)

print("all categorical columns",join.shape)

print("shape of final train data after concat",train.shape)
a=pd.DataFrame(combine.iloc[327:,2]*12)## multiplying by 12 as 1 year containg 12 months

a=a.rename(columns={'Tenure':'Tenure in month'})

a.head()
import math

x=combine.iloc[:327,2]## tenure column

year=[]

m=[]#math.modf(x)

month=[]

for i in x:

    b=i.split('.')

    year.append(int(b[0]))

    month.append(int(b[1]))

year=pd.DataFrame(year,columns=['Tenure in month'])

month=pd.DataFrame(month,columns=['Tenure in month'])

month=pd.DataFrame(year*12+month)

month.head()
month=pd.concat([month,a],axis=0)

train_final=pd.concat([month,train],axis=1)

print("Shape of the final training dataset :",train_final.shape)

train_final.head()
import pandas as pd

from sklearn import preprocessing



x = train_final.values #returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(train_final)

x = pd.DataFrame(x_scaled)

x.head()
x_val=x.iloc[:327,:]

y_val=y.iloc[:327]

x_train=x.iloc[327:,:]

y_train=y.iloc[327:,]
classifiers = [

    KNeighborsClassifier(3),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    GradientBoostingClassifier(),

    LinearDiscriminantAnalysis()]
log_cols=["Classifier", "Accuracy", "Log Loss"]

log = pd.DataFrame(columns=log_cols)
for clf in classifiers:

    clf.fit(x_train, y_train)

    name = clf.__class__.__name__

    

    print("-"*30)

    print(name)

    

    print('****Results****')

    train_predictions = clf.predict(x_val)

    acc = accuracy_score(y_val, train_predictions)

    print("Accuracy: {:.4%}".format(acc))

    print("-"*30)

print("="*30)
features = [c for c in x.columns]

target = y_train

target.values

param = {

    'bagging_freq': 5,          'bagging_fraction': 0.38,   'boost_from_average':'false',   'boost': 'gbdt',

    'feature_fraction': 0.045,   'learning_rate': 0.01,     'max_depth': -1,                'metric':'auc',

    'min_data_in_leaf': 80,     'min_sum_hessian_in_leaf': 10.0,'num_leaves': 13,           'num_threads': 8,

    'tree_learner': 'serial',   'objective': 'binary',      'verbosity': 1

}

folds = StratifiedKFold(n_splits=15, shuffle=False, random_state=17154016)

oof = np.zeros(len(x_train))

predictions = np.zeros(len(x_val))
for fold_, (trn_idx, val_idx) in enumerate(folds.split(x_train.values, target.values)):

    print("Fold :{}".format(fold_ + 1))

    trn_data = lgb.Dataset(x_train.iloc[trn_idx][features], label=target.iloc[trn_idx])

    val_data = lgb.Dataset(x_train.iloc[val_idx][features], label=target.iloc[val_idx])

    clf = lgb.train(param, trn_data, 100000, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 1000)

    oof[val_idx] = clf.predict(x_train.iloc[val_idx][features], num_iteration=clf.best_iteration)

    predictions += clf.predict(x_val[features], num_iteration=clf.best_iteration) / folds.n_splits

sys.stdout.write("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
clf = RandomForestClassifier(random_state=0, n_jobs=-1)

# Train model

model = clf.fit(x_train, y_train)

importances = clf.feature_importances_

# Sort feature importances in descending order

indices = np.argsort(importances)[::-1]

plt.rcParams.update({'font.size': 30})

# Rearrange feature names so they match the sorted feature importances

names = train_final.columns

plt.figure(figsize=(60,15))

# Barplot: Add bars

plt.bar(range(x_train.shape[1]), importances[indices])

# Add feature names as x-axis labels

plt.xticks(range(x_train.shape[1]), names, rotation=90, fontsize = 25)



# Create plot title

plt.title("Feature Importance")

# Show plot

plt.show()
data_tenure=train_final 

print("shape of training data for tenure prdiciton",data_tenure.shape)

data_tenure.head()
x = train_final.values #returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(data_tenure)

X = pd.DataFrame(x_scaled,columns=data_tenure.columns)

y_tenure=pd.DataFrame(X['Tenure in month'])

x_tenure=X.drop(['Tenure in month'],axis=1) ## Dropping tenure column

print('Shape of x_tenure :',x_tenure.shape)

x_tenure.head()
y_tenure.head()
x_val_tenure=x_tenure.iloc[:327,:]## first 327 are given examples rest are of ibm dataset examples

y_val_tenure=y_tenure.iloc[:327]

x_train_tenure=x_tenure.iloc[327:,:]

y_train_tenure=y_tenure.iloc[327:]

# labeling column so each column as unique name

x_train_tenure=pd.DataFrame(np.array(x_train_tenure))

y_train_tenure=pd.DataFrame(np.array(y_train_tenure))

x_val_tenure=pd.DataFrame(np.array(x_val_tenure))

y_val_tenure=pd.DataFrame(np.array(y_val_tenure))

print("shape of bitgrt-given dataset (cross-val)   : ",x_val_tenure.shape,y_val_tenure.shape) 

print("shape of ibm public dataset  (training data):",x_train_tenure.shape,y_train_tenure.shape)
#XGBRegressor()

my_model = XGBRegressor()  

my_model.fit(x_train_tenure,y_train_tenure)

xgbpredictions = my_model.predict(x_val_tenure)



#RandomForestRegressor

rf = RandomForestRegressor(n_estimators=200, random_state=1234)

rf.fit(x_train_tenure,y_train_tenure)

rfpredictions = rf.predict(x_val_tenure)



#ExtraTreesRegressor

extra_tree = ExtraTreesRegressor()

extra_tree.fit(x_train_tenure,y_train_tenure)

etpredictions = extra_tree.predict(x_val_tenure)



#GradientBoostingRegressor

gbr = GradientBoostingRegressor()

gbr.fit(x_train_tenure,y_train_tenure)

y_pred =gbr.predict(x_val_tenure)



#SVR

reg=SVR()

reg.fit(x_train_tenure,y_train_tenure)

y_predsvr = reg.predict(x_val_tenure)





print("Root Mean Absolute Error")

print("XGBRegressor              : ",sqrt(mean_squared_error(xgbpredictions, y_val_tenure)))

print("RandomForestRegressor     : ",sqrt(mean_squared_error(rfpredictions, y_val_tenure)))

print("ExtraTreesRegressor       : ",sqrt(mean_squared_error(etpredictions, y_val_tenure)))

print("GradientBoostingRegressor : ",sqrt(mean_squared_error(y_pred, y_val_tenure)))

print("Support Vector Regressor  : ",sqrt(mean_squared_error(y_predsvr, y_val_tenure)))
lgb_train = lgb.Dataset(x_train_tenure,y_train_tenure)

lgb_eval = lgb.Dataset(x_val_tenure,  y_val_tenure, reference=lgb_train)

params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': {'l2', 'l1'},

    'num_leaves': 50,

    'learning_rate': 0.1,

    'feature_fraction': 0.5,

    'bagging_fraction': 0.5,

    'bagging_freq': 20,

    'verbose': 1

}

gbm = lgb.train(params,

                lgb_train,

                num_boost_round=500,

                valid_sets=lgb_eval,

                early_stopping_rounds=5)
y_pred = gbm.predict(x_val_tenure, num_iteration=gbm.best_iteration)

print(" Root Mean Absolute Error by lgbm : ",sqrt(mean_squared_error(y_pred, y_val_tenure)))
importances = rf.feature_importances_

# Sort feature importances in descending order

indices = np.argsort(importances)[::-1]

plt.rcParams.update({'font.size': 20})

# Rearrange feature names so they match the sorted feature importances

names = x_tenure.columns

plt.figure(figsize=(40,15))

# Barplot: Add bars

plt.bar(range(x_train_tenure.shape[1]), importances[indices])

# Add feature names as x-axis labels

plt.xticks(range(x_train_tenure.shape[1]), names, rotation=90, fontsize = 20)



# Create plot title

plt.title("Feature Importance")

# Show plot

plt.show()