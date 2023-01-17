# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/summeranalytics2020/train.csv')

test=pd.read_csv('/kaggle/input/summeranalytics2020/test.csv')
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
pd.options.display.max_columns=80
train.head()
train.isnull().sum()
object_cols=list(set(train.select_dtypes(include='object').columns))

plt.figure(figsize=(20, 50))

for i in range(len(object_cols)):

    plt.subplot(7,1,i+1)

    sns.countplot(train[object_cols[i]], hue = train.Attrition)
f=sns.FacetGrid(train,col="Attrition")

f.map(plt.hist,"Behaviour")
f=sns.FacetGrid(train,col="Attrition")

f.map(plt.hist,"CommunicationSkill")
f=sns.FacetGrid(train,col="Attrition")

f.map(plt.hist,"PerformanceRating")
f=sns.FacetGrid(train,col="Attrition")

f.map(plt.hist,"EnvironmentSatisfaction")
f=sns.FacetGrid(train,col="Attrition")

f.map(plt.hist,"JobInvolvement")
f=sns.FacetGrid(train,col="Attrition")

f.map(plt.hist,"Education")
f=sns.FacetGrid(train,col="Attrition")

f.map(plt.hist,"StockOptionLevel")
num_cols=['Age', 'DistanceFromHome', 'EmployeeNumber', 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike','TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']



for i in range(len(num_cols)):

    sns.scatterplot(x=train[num_cols[i]],y=train.Attrition)

    plt.show()
one_hot_1=pd.get_dummies(train['BusinessTravel'])

one_hot_2=pd.get_dummies(test['BusinessTravel'])

train=train.drop(['BusinessTravel'],axis=1)

train=train.join(one_hot_1)

test=test.drop(['BusinessTravel'],axis=1)

test=test.join(one_hot_2)
one_hot_3=pd.get_dummies(train['Department'])

one_hot_4=pd.get_dummies(test['Department'])

train=train.drop(['Department'],axis=1)

train=train.join(one_hot_3)

test=test.drop(['Department'],axis=1)

test=test.join(one_hot_4)
train.rename(columns={'Human Resources':'Human_resources_dept'},inplace=True)

test.rename(columns={'Human Resources':'Human_resources_dept'},inplace=True)
one_hot_5=pd.get_dummies(train['Gender'])

one_hot_6=pd.get_dummies(test['Gender'])

train=train.drop(['Gender'],axis=1)

train=train.join(one_hot_5)

test=test.drop(['Gender'],axis=1)

test=test.join(one_hot_6)
one_hot_7=pd.get_dummies(train['MaritalStatus'])

one_hot_8=pd.get_dummies(test['MaritalStatus'])

train=train.drop(['MaritalStatus'],axis=1)

train=train.join(one_hot_7)

test=test.drop(['MaritalStatus'],axis=1)

test=test.join(one_hot_8)
one_hot_9=pd.get_dummies(train['EducationField'])

one_hot_10=pd.get_dummies(test['EducationField'])

train=train.drop(['EducationField'],axis=1)

train=train.join(one_hot_9)

test=test.drop(['EducationField'],axis=1)

test=test.join(one_hot_10)
train.rename(columns={'Human Resources':'Human_resources_ed'},inplace=True)

test.rename(columns={'Human Resources':'Human_resources_ed'},inplace=True)
one_hot_11=pd.get_dummies(train['JobRole'])

one_hot_12=pd.get_dummies(test['JobRole'])

train=train.drop(['JobRole'],axis=1)

train=train.join(one_hot_11)

test=test.drop(['JobRole'],axis=1)

test=test.join(one_hot_12)
train['OverTime']=list(map(lambda x: 1 if x=='Yes' else 0,train['OverTime']))

test['OverTime']=list(map(lambda x: 1 if x=='Yes' else 0,test['OverTime']))
train.head()
for i in range(len(num_cols)):

    sns.countplot(x=train[num_cols[i]])

    plt.show()
from scipy.stats import skew



skew_f=train.apply(lambda x:skew(x)).sort_values(ascending=False)

highest_skew=skew_f[skew_f>0.5]

highest_skew
train['YearsSinceLastPromotion']=np.sqrt(train['YearsSinceLastPromotion'])

train['YearsAtCompany']=np.sqrt(train['YearsAtCompany'])

train['MonthlyIncome']=np.sqrt(train['MonthlyIncome'])

train['TotalWorkingYears']=np.sqrt(train['TotalWorkingYears'])

train['YearsInCurrentRole']=np.sqrt(train['YearsInCurrentRole'])

train['YearsWithCurrManager']=np.sqrt(train['YearsWithCurrManager'])

train['NumCompaniesWorked']=np.sqrt(train['NumCompaniesWorked'])

train['DistanceFromHome']=np.sqrt(train['DistanceFromHome'])

train['PercentSalaryHike']=np.sqrt(train['PercentSalaryHike'])
test['YearsSinceLastPromotion']=np.sqrt(test['YearsSinceLastPromotion'])

test['YearsAtCompany']=np.sqrt(test['YearsAtCompany'])

test['MonthlyIncome']=np.sqrt(test['MonthlyIncome'])

test['TotalWorkingYears']=np.sqrt(test['TotalWorkingYears'])

test['YearsInCurrentRole']=np.sqrt(test['YearsInCurrentRole'])

test['YearsWithCurrManager']=np.sqrt(test['YearsWithCurrManager'])

test['NumCompaniesWorked']=np.sqrt(test['NumCompaniesWorked'])

test['DistanceFromHome']=np.sqrt(test['DistanceFromHome'])

test['PercentSalaryHike']=np.sqrt(test['PercentSalaryHike'])
for col in num_cols:

    train[col]=(train[col]-np.mean(train[col]))/np.std(train[col])

    test[col]=(test[col]-np.mean(test[col]))/np.std(test[col])
train=train.drop(['EmployeeNumber', 'Behaviour'],axis=1)

test=test.drop(['EmployeeNumber', 'Behaviour'],axis=1)
train_x=train.drop(["Attrition","Id"],axis=1)

train_y=train['Attrition']
from sklearn.decomposition import PCA



pca=PCA(0.99)

p_c=pca.fit_transform(train_x)

train_xx=pd.DataFrame(data=p_c)
p_c=pca.transform(test.drop(['Id'],axis=1))

test_x=pd.DataFrame(data=p_c)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegressionCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.svm import SVC

import xgboost as xgb

from sklearn.metrics import confusion_matrix, accuracy_score, auc, roc_curve, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV



svc=SVC(kernel='linear', gamma='auto')

svc2=CalibratedClassifierCV(svc)

svc2.fit(train_xx,train_y)
y_pred_svc=svc2.predict_proba(test_x)[:,1]
test['Attrition']=y_pred_svc
test[['Id', 'Attrition']].to_csv('mysubmission.csv', index = False)
## model1=LogisticRegressionCV(10,random_state=1)

## model1.fit(train_xx,train_y)

## y_pred_log=model1.predict_proba(test_x)[:,1]



## model2=DecisionTreeClassifier(criterion='gini', max_depth=10, max_leaf_nodes=23, splitter='best')

## model2.fit(train_xx,train_y)

## y_pred_dec=model2.predict(train_x)



## randomForest = RandomForestClassifier()

## param_grid = { 

##   'criterion' : ['gini', 'entropy'],

##  'n_estimators': [100, 300, 500],

##  'max_features': ['auto', 'log2'],

## 'max_depth' : [3, 5, 7] }



## from sklearn.model_selection import GridSearchCV



# Grid search

## randomForest_CV = GridSearchCV(estimator = randomForest, param_grid = param_grid, cv = 6,scoring='roc_auc')

## randomForest_CV.fit(df_xx, train_y)



## randomForest_CV.best_params_



## randomForestFinalModel = RandomForestClassifier(criterion = 'gini', max_depth = 7, max_features = 'auto', n_estimators = 100,max_leaf_nodes=23)

## randomForestFinalModel.fit(train_xx, train_y)

## y_pred_rnn=randomForestFinalModel.predict_proba(df_x)[:,1]



## gnb=GaussianNB()

## gnb.fit(train_x,train_y)



## regr=xgb.XGBClassifier()

## regr.fit(train_xx,train_y)

## y_pred_xgb=regr.predict_proba(test_x)[:,1]



## ada=AdaBoostClassifier(random_state=2,n_estimators=200)

## ada.fit(train_xx,train_y)



## from sklearn import ensemble

## params={'n_estimators':500, 'max_depth':4, 'min_samples_split':2,'learning_rate':0.01}

## gbc=ensemble.GradientBoostingClassifier(**params)

## gbc.fit(train_xx,train_y)



## y_pred_gbc=gbc.predict_proba(test.drop(['Id'],axis=1))[:,1]

## y_pred_gnb=gnb.predict_proba(test.drop(['Id'],axis=1))[:,1]

## y_pred_ada=ada.predict_proba(test.drop(['Id'],axis=1))[:,1]



## from sklearn.neighbors import KNeighborsClassifier

## clf=KNeighborsClassifier(n_neighbors=15)

## clf.fit(train_x,train_y)



## y_pred_knn=clf.predict_proba(test.drop(['Id'],axis=1))[:,1]



## Y_len=len(test)

## y_pred_ensemble3=np.zeros(Y_len)

## for i in range(Y_len):

     #if(y_pred_rn[i] + y_pred_log[i] + y_pred_svc[i] >=2):

     #y_pred_ensemble3[i]=(y_pred_rn[i]+y_pred_svc[i]+y_pred_log[i]+y_pred_xgb[i]+y_pred_gnb[i])/5

     #y_pred_ensemble3[i]=(y_pred_svc[i]+y_pred_log[i]+y_pred_rn[i]+y_pred_xgb[i]+y_pred_ada[i])/5

     #y_pred_ensemble3[i]=(y_pred_rn[i]+y_pred_svc[i]+y_pred_log[i]+y_pred_knn[i])/4

     #y_pred_ensemble3[i]=(y_pred_svc[i]+y_pred_log[i])/2