# Importing libraries



import pandas as pd

from collections import Counter

import numpy as np

from scipy import stats
# Reading data

data=pd.read_csv('../input/telecom.csv')

data.head()
data.columns=['State','AccountLength','AreaCode','Phone','InternationalPlan','VMailPlan',

              'VMailMessage','DayMins','DayCalls','DayCharge','EveMins','EveCalls','EveCharge',

              'NightMins','NightCalls','NightCharge','InternationalMins','InternationalCalls',

              'InternationalCharge','CustServCalls','Churn']
print(Counter(data.State))

print('Shape of DataFrame : ',data.shape)
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import tree

from sklearn.metrics import accuracy_score

%matplotlib inline

sns.set()
data.describe()
print(data.nunique())
data.isnull().sum()
sns.heatmap(data.isnull(),cbar=False,yticklabels=False,cmap = 'viridis')
from sklearn.preprocessing import LabelEncoder

enc=LabelEncoder()

data.State = enc.fit_transform(data.State)

data.InternationalPlan = enc.fit_transform(data.InternationalPlan)

data.VMailPlan = enc.fit_transform(data.VMailPlan)

data.Churn = enc.fit_transform(data.Churn)

data.head(2)
# Create Column Lists

AllColumns = ['State','AccountLength','AreaCode','InternationalPlan','VMailPlan','VMailMessage',

              'DayMins','DayCalls','DayCharge','EveMins','EveCalls','EveCharge','NightMins',

              'NightCalls','NightCharge','InternationalMins','InternationalCalls','InternationalCharge',

              'CustServCalls','Churn']



ConVarList=['AccountLength','AreaCode','VMailMessage','DayMins','DayCalls','DayCharge','EveMins',

            'EveCalls','EveCharge','NightMins','NightCalls','NightCharge','InternationalMins',

            'InternationalCalls','InternationalCharge','CustServCalls']



AllConVarList=['AccountLength','AreaCode','VMailMessage','DayMins','DayCalls','DayCharge',

               'EveMins','EveCalls','EveCharge','NightMins','NightCalls','NightCharge','InternationalMins',

               'InternationalCalls','InternationalCharge','CustServCalls','Churn']



CatVarList=['State','InternationalPlan','VMailPlan']

OutcomeVar=['Churn']
Delete_Col_List = ['Phone']

data.drop(Delete_Col_List, inplace=True,axis=1)
data.head(2)
data[AllConVarList].corr(method='pearson')
# Multicollearnity of continous variables between x->x and x->y

plt.figure(figsize=(15,5))

sns.heatmap(data[AllConVarList].corr(),cmap='viridis',annot=True) 
# Multicollearnity of continous and categorical variables converted to numbers between x->x and x->y

plt.figure(figsize=(20,10))

sns.heatmap(data.corr(),cmap='viridis',annot=True) 
# p-value:0.000 < 0.05 we reject Null Hypothesis,

#two variables are dependent hence we require InternationalPlan for prediction.

print('------------------------------------------------------------------------------------------------------')

print(pd.crosstab(data.InternationalPlan,data.Churn))

chi, p, dof, expected = stats.chi2_contingency(pd.crosstab(data.InternationalPlan,data.Churn))

print("Chi     :",chi)

print("P-value :",p)

print("dof     :",dof)

print("expected:",expected)

print('------------------------------------------------------------------------------------------------------')

#p-value:0.00 < 0.05 we reject Null Hypothesis, two variables are dependent hence we require State for prediction

print(pd.crosstab(data.State,data.Churn))

chi2, p2, dof2, expected2 = stats.chi2_contingency(pd.crosstab(data.State,data.Churn))

print('------------------------------------------------------------------------------------------------------')

print("Chi     :",chi2)

print("P-value :",p2)

print("dof     :",dof2)

print("expected:",expected2)

print('------------------------------------------------------------------------------------------------------')

#p-value:1.03 > 0.05 we fail to reject Null Hypothesis, two variables are independent hence we do not require VMailPlan for prediction

print(pd.crosstab(data.VMailPlan,data.Churn))

chi3, p3, dof3, expected3 = stats.chi2_contingency(pd.crosstab(data.VMailPlan,data.Churn))

print("Chi     :",chi3)

print("P-value :",p3)

print("dof     :",dof3)

print("expected:",expected3)
Delete_Col_List = ['DayMins','EveMins','NightMins','InternationalMins','VMailPlan']

data.drop(Delete_Col_List, inplace=True,axis=1)
data.head()
print(data.shape)

data_dummy=pd.get_dummies(data['State'],prefix='state')
data_dummy.head()
data.drop('State', inplace=True,axis=1) # deleting State Column from the main DataFrame.
data.head()
data=data.join(data_dummy) # joining main DataFrame with dummy DataFrame.
data_dummy1=pd.get_dummies(data['InternationalPlan'],prefix='InternationalPlan')
data.drop('InternationalPlan', inplace=True,axis=1)
data.head()
data=data.join(data_dummy1)
data.head()
cols=[col for col in data.columns if col not in ['Churn']]
data.shape
X=data[cols]

y=data.Churn
print(X.head())

print(X.shape)

print(y.head())

print(y.shape)
X.head()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=25)
from sklearn.metrics import recall_score

from sklearn.ensemble import RandomForestClassifier

model1 =RandomForestClassifier(max_depth=17,n_estimators=25,max_features=20

                               ,criterion='entropy',random_state=15)

#random state in RF is imp.-which variable is picked for estimators/tress

#max_features in RF, no. of columns in each tree/elemnts in each cluster

#n_estimators - no.of tress

model1.fit(X_train,y_train) # Training

y_predict_rf = model1.predict(X_test) # Test Prediction
print(accuracy_score(y_test, y_predict_rf))

print(recall_score(y_test,y_predict_rf))
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

model2 = XGBClassifier(random_state=10,n_estimators=170,learning_rate=0.1)

model2.fit(X_train, y_train)

y_predict_xgb = model2.predict(X_test)

print("Accuracy :- ",accuracy_score(y_test, y_predict_xgb))

print("Recall :- ",recall_score(y_test,y_predict_xgb))



print(pd.crosstab(y_test, y_predict_xgb))
# Also tried KFold Cross validation for spiting the test and train data but it didn't gave any improved value.So, i'm taking train_test_split as it is more optimized than KFold.
#importing required libraries

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,recall_score,precision_score,roc_curve,roc_auc_score

import matplotlib.pyplot as plt
sm = SMOTE()

X_train_sm,y_train_sm = sm.fit_sample(X_train,y_train)
# now the chuners and non chuners ratio is same in the output

print(Counter(y_train_sm))

print(Counter(y_test))
model3 = XGBClassifier(random_state=10,n_estimators=170,learning_rate=0.1)

model3.fit(X_train, y_train)

y_predict_xgb_sm = model3.predict(X_test)

print("Accuracy :- ",accuracy_score(y_test, y_predict_xgb_sm))

print("Recall :- ",recall_score(y_test,y_predict_xgb_sm))



print(pd.crosstab(y_test, y_predict_xgb_sm))
#RandomForest Smote

from sklearn.metrics import recall_score

from sklearn.ensemble import RandomForestClassifier

model4 =RandomForestClassifier(max_depth=17,n_estimators=25,max_features=20

                               ,criterion='entropy',random_state=15)

model4.fit(X_train,y_train)

y_predict_rf_sm = model4.predict(X_test)

print("Accuracy :- ",accuracy_score(y_test, y_predict_rf_sm))

print("Recall :- ",recall_score(y_test,y_predict_rf_sm))

print(pd.crosstab(y_test, y_predict_rf_sm))
params= {'n_estimators':[10,20,50,70,80,100,120],

         'criterion': ['entropy','gini'],

          'max_features':['auto',5,7,10,15],

          'random_state':range(0,10)

          }
model_cv=RandomizedSearchCV(RandomForestClassifier(),params)

#model_cv =GridSearchCV(RandomForestClassifier(),params)
model_cv.fit(X_train_sm,y_train_sm)
model_cv.best_params_
model_cv.best_estimator_
model_cv.best_score_