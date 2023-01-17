import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import pandas_profiling
data = pd.read_csv('../input/diabetes.csv')

data.head()
pandas_profiling.ProfileReport(data)
data.shape
data.isnull().sum()
# O1 = Q1-1.5(IQR) -> 0.082-(1.5*3.6)

# O2 = Q3+1.5(IQR) -> 3.67+(1.5*3.6) =9

data2=data.copy()

del data2['Outcome']

for i in range(8):

    col=data2.columns[i]

    q3, q1 = np.percentile(data2[col], [75,25])

    iqr = q3 - q1

    o1=q1-1.5*iqr

    o2=q3+1.5*iqr

    lis=[]

    for j in range(768):

        if data2['{}'.format(col)][j]<=o2 and data['{}'.format(col)][j]>=o1:

            lis.append(data2[col][j])

    print("Outliers {} ->".format(col),768-len(lis))
plt.figure(figsize=(8,8))

sns.heatmap(data.corr(),annot=True,cmap='magma')
sns.countplot('Outcome',data=data)
data.describe()
data.info()
sns.boxplot('Outcome','Glucose',data=data)
sns.catplot('Glucose','Insulin',data=data,hue='Outcome')
# O1 = Q1-1.5(IQR) -> 0.082-(1.5*3.6)

# O2 = Q3+1.5(IQR) -> 3.67+(1.5*3.6) =9

data3=data[['Glucose']]

for i in range(1):

    col=data3.columns[i]

    q3, q1 = np.percentile(data3[col], [75,25])

    iqr = q3 - q1

    o1=q1-1.5*iqr

    o2=q3+1.5*iqr

    lis=[]

    mean=np.mean(data[col])

    for j in range(506):

        if data3['{}'.format(col)][j]>o2 or data3['{}'.format(col)][j]<o1:

            data[col][j]=mean
sns.boxplot('Outcome','BloodPressure',data=data)
sns.jointplot('BloodPressure','Insulin',data=data,kind='kde',color='red')
g=sns.catplot('BloodPressure','SkinThickness',data=data,kind='strip',hue='Outcome',legend_out=False)

g.fig.set_size_inches(15,6)
sns.regplot('BloodPressure','SkinThickness',data=data,marker='*',color='red')
g=sns.catplot('BloodPressure','Age',data=data,hue='Outcome',kind='point',legend_out=False)

g.fig.set_size_inches(15,6)
g=sns.relplot('SkinThickness','Insulin',data=data,kind='line',hue='Outcome')

g.fig.set_size_inches(15,6)
sns.boxenplot('Outcome','SkinThickness',data=data)
g=sns.relplot('Insulin','DiabetesPedigreeFunction',data=data,kind='scatter',hue='Outcome')

g.fig.set_size_inches(15,6)
sns.jointplot('Insulin','DiabetesPedigreeFunction',kind='kde',color='y',data=data)
sns.violinplot('Outcome','BMI',data=data,bw='scott',scale='area',split=False,inner='quartile')
data.head()
data['BloodPressure'] = np.where(data['BloodPressure']==0, data['BloodPressure'].mean(), data['BloodPressure'])

data['BMI'] = np.where(data['BMI']==0, data['BMI'].mean(), data['BMI'])

data['Insulin'] = np.where(data['Insulin']==0, data['Insulin'].mean(), data['Insulin'])

data['SkinThickness'] = np.where(data['SkinThickness']==0, data['SkinThickness'].mean(), data['SkinThickness'])
X=data.iloc[:,:-1].values

y=data.iloc[:,-1].values
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)
'''from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score,GridSearchCV

params = {

     'learning_rate': [0.05,0.06],

     'n_estimators': [1000,1100],

     'max_depth':[7,8],

     'reg_alpha':[0.3,0.4,0.5]

    }

 

# Initializing the XGBoost Regressor

xgb_model = XGBClassifier()

 

# Gridsearch initializaation

gsearch = GridSearchCV(xgb_model, params,

                    verbose=True,

                    cv=10,

                    n_jobs=-1)

gsearch.fit(X,y) 

#Printing the best chosen params

print(gsearch.best_params_)'''
import xgboost

from xgboost import XGBClassifier

rfc= XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=1, gamma=0,

              learning_rate=0.06, max_delta_step=0, max_depth=8,

              min_child_weight=1, missing=None, n_estimators=1000, n_jobs=1,

              nthread=None, objective='binary:logistic', random_state=0,

              reg_alpha=0.5, reg_lambda=1, scale_pos_weight=1, seed=None,

              silent=None, subsample=1, verbosity=1)

rfc.fit(X_train,y_train)

y_pred= rfc.predict(X_test)

y_pred_prob=rfc.predict_proba(X_test)
y_pred_prob=y_pred_prob[:,1]
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

print(accuracy_score(y_test,y_pred))
from sklearn.metrics import roc_curve

fpr,tpr,thresholds=roc_curve(y_test,y_pred_prob)

plt.plot(fpr,tpr)

plt.xlim([0.0,1.0])

plt.ylim([0.0,1.0])

plt.title('ROC Curve')

plt.xlabel('FalsePositiveRate')

plt.ylabel('TruePositiveRate')

plt.grid(True)
