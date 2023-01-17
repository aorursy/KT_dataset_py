# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install imblearn
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

np.random.seed(0)
data=pd.read_csv('/kaggle/input/indian-liver-patient-records/indian_liver_patient.csv')

data.head()
data['Dataset'].value_counts()
data.info()
print(data.isnull().sum())
data.shape
data.describe().T
print('Categorical Columns are ','\n',list(data.select_dtypes(include='object')),'\n')

print('Numerical Columns are ','\n',list(data.select_dtypes(exclude='object')))
data.isnull().mean()*100
data['Albumin_and_Globulin_Ratio']=data['Albumin_and_Globulin_Ratio'].fillna(data['Albumin_and_Globulin_Ratio'].mean()) 

     # In this case the distribution is not very much skewed thus we can either go for median or mode.
data['Gender']=data['Gender'].map({'Male':0,'Female':1})
data.drop('Dataset',axis=1).plot(kind='box',layout=(2,5),subplots=True,figsize=(12,6))

plt.show()
data.describe().T
q1=data['Total_Bilirubin'].quantile(0.25)

q3=data['Total_Bilirubin'].quantile(0.75)

iqr=q3-q1

data['Total_Bilirubin']=data['Total_Bilirubin'].apply(lambda x : q1-(1.5*iqr) if x<q1-(1.5*iqr) else x).apply(lambda x : q3+(1.5*iqr) if x>q3+(1.5*iqr) else x)





q1=data['Direct_Bilirubin'].quantile(0.25)

q3=data['Direct_Bilirubin'].quantile(0.75)

iqr=q3-q1

data['Direct_Bilirubin']=data['Direct_Bilirubin'].apply(lambda x : q3+(1.5*iqr) if x>q3+(1.5*iqr) else x).apply(lambda x : q1-(1.5*iqr) if x<q1-(1.5*iqr) else x)



q1=data['Alkaline_Phosphotase'].quantile(0.25)

q3=data['Alkaline_Phosphotase'].quantile(0.75)

iqr=q3-q1

data['Alkaline_Phosphotase']=data['Alkaline_Phosphotase'].apply(lambda x : q3+(1.5*iqr) if x>q3+(1.5*iqr) else x).apply(lambda x : q1-(1.5*iqr) if x<q1-(1.5*iqr) else x)





q1=data['Alamine_Aminotransferase'].quantile(0.25)

q3=data['Alamine_Aminotransferase'].quantile(0.75)

iqr=q3-q1

data['Alamine_Aminotransferase']=data['Alamine_Aminotransferase'].apply(lambda x : q3+(1.5*iqr) if x>q3+(1.5*iqr) else x).apply(lambda x : q1-(1.5*iqr) if x<q1-(1.5*iqr) else x)





q1=data['Aspartate_Aminotransferase'].quantile(0.25)

q3=data['Aspartate_Aminotransferase'].quantile(0.75)

iqr=q3-q1

data['Aspartate_Aminotransferase']=data['Aspartate_Aminotransferase'].apply(lambda x : q3+(1.5*iqr) if x>q3+(1.5*iqr) else x).apply(lambda x : q1-(1.5*iqr) if x<q1-(1.5*iqr) else x)





q1=data['Total_Protiens'].quantile(0.25)

q3=data['Total_Protiens'].quantile(0.75)

iqr=q3-q1

data['Total_Protiens']=data['Total_Protiens'].apply(lambda x : q3+(1.5*iqr) if x>q3+(1.5*iqr) else x).apply(lambda x : q1-(1.5*iqr) if x<q1-(1.5*iqr) else x)





q1=data['Albumin_and_Globulin_Ratio'].quantile(0.25)

q3=data['Albumin_and_Globulin_Ratio'].quantile(0.75)

iqr=q3-q1

data['Albumin_and_Globulin_Ratio']=data['Albumin_and_Globulin_Ratio'].apply(lambda x : q3+(1.5*iqr) if x>q3+(1.5*iqr) else x).apply(lambda x : q1-(1.5*iqr) if x<q1-(1.5*iqr) else x)







# Capping the outlier values to the quantile (+-) 1.5*iqr value.
data.drop('Dataset',axis=1).plot(kind='box',layout=(2,5),subplots=True,figsize=(12,6))

plt.show()
sns.distplot(data['Dataset'],rug=True,color='midnightblue') 
sns.countplot(data['Dataset'])

print('The ratio between the target variables are ',data['Dataset'].value_counts()[2]/data['Dataset'].value_counts()[1])
plt.figure(figsize=(10,6))

sns.heatmap(data.corr(),cmap='magma',annot=True)

plt.show()
data.corr()['Dataset'] # To get exact values of correlation by X-Variables on Y-Variable.
for i in data.columns:

    sns.scatterplot(data[i],data['Dataset'],color='midnightblue')

    plt.show()
plt.figure(figsize=(10,5))

data.corr()['Dataset'].sort_values(ascending=False).plot(kind='bar',color='black')

plt.xticks(rotation=45)

plt.xlabel('Variables in the Data')

plt.ylabel('Correlation Values')

plt.show()
x=data.drop('Dataset',axis=1)

y=data['Dataset']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=43)
x.describe()
x_train.describe()
x_test.describe()
from sklearn.preprocessing import StandardScaler

ss=StandardScaler()
ss.fit(x_train)
x_train=ss.transform(x_train)

x_test=ss.transform(x_test)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score,recall_score

log=LogisticRegression()
log.fit(x_train,y_train)
y_pred=log.predict(x_test)

accuracy_score(y_test,y_pred)
plt.figure(figsize=(10,6))

sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt='d')
print('F1 Score of the prediction is ',f1_score(y_test,y_pred))

print('Recall Score of the prediction is ',recall_score(y_test,y_pred))
from sklearn.feature_selection import RFE
#no of features

nof_list=np.arange(1,19)            

high_score=0

#Variable to store the optimum features

nof=0          

score_list =[]

for n in range(len(nof_list)):

    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 43)

    model = LogisticRegression()

    rfe = RFE(model,nof_list[n])

    X_train_rfe = rfe.fit_transform(X_train,y_train)

    X_test_rfe = rfe.transform(X_test)

    model.fit(X_train_rfe,y_train)

    score = model.score(X_test_rfe,y_test)

    score_list.append(score)

    if(score>high_score):

        high_score = score

        nof = nof_list[n]

print("Optimum number of features: %d" %nof)

print("Score with %d features: %f" % (nof, high_score))

rfe = RFE(model,3)

rfe.fit_transform(X_train,y_train)

y_pred = rfe.predict(X_test)



accuracy_score(y_test,y_pred)

pd.DataFrame(rfe.support_,index=x.columns)
from imblearn.over_sampling import SMOTE

smote=SMOTE()
x=data.drop('Dataset',axis=1)

y=data['Dataset']
x_01=x

y_01=y

# For reusability of the variable
x,y=smote.fit_sample(x,y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=43)
ss.fit(x_train)
x_train=ss.transform(x_train)

x_test=ss.transform(x_test)
pd.DataFrame(y)['Dataset'].value_counts() 
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
pd.DataFrame(dt.feature_importances_,index=x_01.columns,columns=['Feature Importance']).sort_values(by='Feature Importance',ascending=False).T
from sklearn.metrics import cohen_kappa_score
#Cohen-Kappa Score for Statistical Analysis.

cohen_kappa_score(y_test,y_pred)
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
cohen_kappa_score(y_test,y_pred)
from sklearn.model_selection import RandomizedSearchCV #Due to comptational capability picking Randomised Search over Grid Search
param={'n_estimators':range(1,50),'random_state':range(1,10),'criterion':['gini','entropy'],'min_samples_split':range(2,10),'min_samples_leaf':range(2,10)}
rand=RandomizedSearchCV(estimator=rf,param_distributions=param,return_train_score=True,cv=5,random_state=3,scoring='recall')
rand.fit(x_train,y_train)
y_pred=rand.predict(x_test)
accuracy_score(y_test,y_pred)
cohen_kappa_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
plt.figure(figsize=(10,6))

sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt='d')

plt.show()
pd.DataFrame(rf.feature_importances_,index=x_01.columns).sort_values(by=0,ascending=False).T
pd.DataFrame(rf.feature_importances_,index=x_01.columns).sort_values(by=0,ascending=False).plot(kind='bar',color='midnightblue',figsize=(10,6))