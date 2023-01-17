

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



data = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

data.head()



data['Churn'] = np.where(data['Churn']=='Yes', 1, 0)

data.head()

    
data.isnull().any()

#No Null Values are present
#General Distribution

sns.countplot(x="Churn",data=data);
a = sns.catplot(x="Contract", y="Churn", data=data,kind="bar")

a.set_ylabels("Churn Probability")
b = sns.catplot(x="InternetService", y="Churn", data=data,kind="bar")

b.set_ylabels("Churn Probability")
sns.distplot(data['tenure']);
#Box Plot Tenure/Churn

graph = pd.concat([data['tenure'], data['Churn']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='Churn', y="tenure", data=graph)

fig.axis(ymin=0, ymax=200);
#Box Plot Tenure/Churn

graph = pd.concat([data['MonthlyCharges'], data['Churn']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='Churn', y="MonthlyCharges", data=graph)

fig.axis(ymin=0, ymax=200);
from sklearn.preprocessing import LabelEncoder



cat = (data.dtypes == 'object')

object_cols = list(cat[cat].index)



labeled = data.copy()



label_encoder = LabelEncoder()

for col in object_cols:

    labeled[col] = label_encoder.fit_transform(labeled[col])

    
corrmat = labeled.corr()

k = 21

cols = corrmat.nlargest(k, 'Churn')['Churn'].index

cm = np.corrcoef(labeled[cols].values.T)

sns.set(font_scale=1)

plt.figure(figsize=(20,15))

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()



### According to correlation results potentially more significant variables:

# (Numerical) MonthlyCharges, Tenure

# (Categorical) PaperlessBilling, Contract, TechSupport, OnlineSecurity, Partner, Dependents, OnlineBackup
labeled.corrwith(labeled.tenure).abs().sort_values(ascending=False).head(10)
labeled.corrwith(labeled.MonthlyCharges).abs().sort_values(ascending=False).head(10)
labeled.corrwith(labeled.PaperlessBilling).abs().sort_values(ascending=False).head(10)
labeled.corrwith(labeled.Contract).abs().sort_values(ascending=False).head(10)
labeled.corrwith(labeled.TechSupport).abs().sort_values(ascending=False).head(10)
#data.MonthlyCharges.describe()

# MonthlyCharges Distribution - Based On How Much Churn



A= data[['MonthlyCharges','Churn','customerID']] 

A['MonthlyCharges_Grouped'] = pd.cut(A.MonthlyCharges,[-np.Infinity,50,80,np.Infinity])



B = A.groupby('MonthlyCharges_Grouped').agg({'customerID':['count'] , 'Churn' : ['sum']})

B
# MonthlyCharges Categorical Variable Generation

labeled['0-50_MonthlyCharges']=(labeled.MonthlyCharges.between(0,50,inclusive=True))

labeled['51-80_MonthlyCharges']=(labeled.MonthlyCharges.between(51,80,inclusive=True))

labeled['81+_MonthlyCharges']=(labeled.MonthlyCharges.between(80,99999,inclusive=True))
#data.tenure.describe()

# Tenure Distribution - Based On How Much Churn



A= data[['tenure','Churn','customerID']] 

A['Tenure_Grouped'] = pd.cut(A.tenure,[-np.Infinity,10,20,np.Infinity])



B = A.groupby('Tenure_Grouped').agg({'customerID':['count'] , 'Churn' : ['sum']})

B
# Tenure Categorical Variable Generation

labeled['0-10_Tenure']=(labeled.tenure.between(0,10,inclusive=True))

labeled['10-20_Tenure']=(labeled.tenure.between(11,20,inclusive=True))

labeled['20+_Tenure']=(labeled.tenure.between(21,99999,inclusive=True))
from sklearn.model_selection import train_test_split



y = labeled.Churn

X = labeled.drop(['Churn'], axis=1)



X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)
X_train = X_train_full[['0-10_Tenure','10-20_Tenure','20+_Tenure','0-50_MonthlyCharges',

                      '51-80_MonthlyCharges','81+_MonthlyCharges','PaperlessBilling','TechSupport',

                     'OnlineSecurity','Partner','Dependents','OnlineBackup','DeviceProtection',

                        'PaperlessBilling','SeniorCitizen','PaymentMethod']].astype(int)



X_valid = X_valid_full[['0-10_Tenure','10-20_Tenure','20+_Tenure','0-50_MonthlyCharges',

                      '51-80_MonthlyCharges','81+_MonthlyCharges','PaperlessBilling','TechSupport',

                     'OnlineSecurity','Partner','Dependents','OnlineBackup','DeviceProtection',

                        'PaperlessBilling','SeniorCitizen','PaymentMethod']].astype(int)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train_full,y_train)

accuracy_score = model.score(X_valid_full,y_valid)

print('Accuracy:',accuracy_score*100)
# K-fold cross validation evaluation of Logistic Model

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



#y = labeled.Churn

#X = labeled.drop(['Churn'], axis=1)



# Cross Validation

model = LogisticRegression()

kfold = KFold(n_splits=4, random_state=7)

results = cross_val_score(model, X, y, cv=kfold)

print("Accuracy:",(results.mean()*100))
import xgboost



# K-fold cross validation evaluation of XGBoost Model



#y = labeled.Churn

#X = labeled.drop(['Churn'], axis=1)



# Cross Validation

model = xgboost.XGBClassifier()

kfold = KFold(n_splits=4, random_state=7)

results = cross_val_score(model, X, y, cv=kfold)

print("Accuracy:",(results.mean()*100))
from sklearn.ensemble import RandomForestClassifier



scores = []

for i in range(1,100):

    model_loop = RandomForestClassifier(n_estimators = i, random_state = 1) 

    model_loop.fit(X_train,y_train)

    scores.append(model_loop.score(X_valid,y_valid))

    

plt.plot(range(1,100),scores)

plt.xlabel("Range")

plt.ylabel("Accuracy")

plt.show()

model = RandomForestClassifier(n_estimators = 30, random_state = 1) 

model.fit(X_train,y_train)

accuracy_score = model.score(X_valid,y_valid)

print("Accuracy:",accuracy_score*100)
# One Year- Two Year Contracts

contracted = labeled[(labeled.Contract == 1) | (labeled.Contract == 2)]
#General Distribution

sns.countplot(x="Churn",data=contracted);
contracted.corrwith(contracted.Churn).abs().sort_values(ascending=False)
from sklearn.model_selection import train_test_split



y = contracted.Churn

X = contracted.drop(['Churn'], axis=1)



X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)

X_train = X_train_full[['MonthlyCharges','Contract','81+_MonthlyCharges','0-50_MonthlyCharges',

                        'StreamingMovies','PaperlessBilling','StreamingTV','OnlineSecurity','51-80_MonthlyCharges']]





X_valid = X_valid_full[['MonthlyCharges','Contract','81+_MonthlyCharges','0-50_MonthlyCharges',

                        'StreamingMovies','PaperlessBilling','StreamingTV','OnlineSecurity',

                       '51-80_MonthlyCharges']]
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train,y_train)

accuracy_score = model.score(X_valid,y_valid)

print('Accuracy:',accuracy_score)
# K-fold cross validation evaluation of Logistic Model

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



#y = labeled.Churn

#X = labeled.drop(['Churn'], axis=1)



# Cross Validation

model = LogisticRegression()

kfold = KFold(n_splits=10, random_state=7)

results = cross_val_score(model, X, y, cv=kfold)

print("Accuracy:",(results.mean()*100))
import xgboost



# K-fold cross validation evaluation of XGBoost Model



#y = labeled.Churn

#X = labeled.drop(['Churn'], axis=1)



# Cross Validation

model = xgboost.XGBClassifier()

kfold = KFold(n_splits=10, random_state=7)

results = cross_val_score(model, X, y, cv=kfold)

print("Accuracy:",(results.mean()*100))
from sklearn.ensemble import RandomForestClassifier



scores = []

for i in range(1,50):

    model_loop = RandomForestClassifier(n_estimators = i, random_state = 1) 

    model_loop.fit(X_train,y_train)

    scores.append(model_loop.score(X_valid,y_valid))

    

plt.plot(range(1,50),scores)

plt.xlabel("Range")

plt.ylabel("Accuracy")

plt.show()

model = RandomForestClassifier(n_estimators = 3, random_state = 1) 

model.fit(X_train,y_train)

accuracy_score = model.score(X_valid,y_valid)

print("Accuracy:",accuracy_score*100)