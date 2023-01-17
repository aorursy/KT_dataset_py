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



import datetime as dt

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import itertools

import statsmodels.api as sm

from sklearn.externals import joblib
import warnings

warnings.filterwarnings('ignore')
import pandas as pd

data = pd.read_csv("../input/ITSM_data.csv")
data.head()
data.shape
data.columns
# Looking for missing data

data.info()
data.CI_Cat.replace(['database'],'storage',inplace=True)

data.CI_Cat.replace(['applicationcomponent'],'subapplication',inplace=True)

data.CI_Cat.replace(['displaydevice','officeelectronics','Phone','networkcomponents'],'hardware',inplace=True)

data.CI_Cat.replace(np.nan,'application',inplace=True)
data.CI_Subcat.replace('Desktop','Desktop Application',inplace=True)

data.CI_Subcat.replace(['Application Server','Virtual Tape Server','ESX Server','zOS Server','Neoview Server','X86 Server',

                       'Unix Server','Oracle Server','Windows Server in extern beheer','Thin Client','NonStop Server',

                       'Number','Windows Server','Linux Server',np.nan,'SharePoint Farm','Lines'],

                       'Server Based Application',inplace=True)

data.CI_Subcat.replace('RAC Service','Banking Device',inplace=True)

data.CI_Subcat.replace(['Iptelephony','Protocol','Net Device','IPtelephony','ESX Cluster','Standard Application'],

                       'Web Based Application',inplace=True)

data.CI_Subcat.replace(['VMWare','Security Software','zOS Systeem','Firewall','Database Software','VDI','Instance',

                       'MQ Queue Manager','Automation Software','Citrix','SAP','Encryption'],'System Software',inplace=True)

data.CI_Subcat.replace(['UPS','Omgeving'],'Client Based Application',inplace=True)

data.CI_Subcat.replace(['NonStop Storage','NonStop Harddisk','Tape Library','zOS Cluster','DataCenterEquipment',

                       'MigratieDummy'],'Database',inplace=True)

data.CI_Subcat.replace(['Modem','Router'],'Network Component',inplace=True)

data.CI_Subcat.replace('KVM Switches','Switch',inplace=True)
data.No_of_Related_Interactions.replace(np.nan,1,inplace=True)

data.Priority.replace(np.nan,4,inplace=True)

data.No_of_Related_Incidents.replace(np.nan,0,inplace=True)

data.No_of_Related_Changes.replace(np.nan,0,inplace=True)
X = data.loc[:,['CI_Cat','CI_Subcat','WBS','Category']]

y = data.Priority
X.head(2)
# Label Encoding

enc= LabelEncoder()

for i in (0,1,2,3):

    X.iloc[:,i] = enc.fit_transform(X.iloc[:,i])
# Splitting the data into test and train for calculating accuracy

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=10)
# Standardization technique

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
X_train.shape
X_test.shape
# Training the model

from sklearn.svm import SVC

rbf_svc = SVC(kernel='rbf',C=10,gamma=0.1).fit(X_train,y_train)
# Predicting the model

y_predict_svm = rbf_svc.predict(X_test)
# Finding accuracy, precision, recall and confusion matrix

print(accuracy_score(y_test,y_predict_svm))

print(classification_report(y_test,y_predict_svm))
confusion_matrix(y_test,y_predict_svm)
# Training the model

from sklearn.tree import DecisionTreeClassifier

model_dtree=DecisionTreeClassifier()

model_dtree.fit(X_train,y_train)
# Predicting the model

y_predict_dtree = model_dtree.predict(X_test)
# Finding accuracy, precision, recall and confusion matrix

print(accuracy_score(y_test,y_predict_dtree))

print(classification_report(y_test,y_predict_dtree))
confusion_matrix(y_test,y_predict_dtree)
# Training the model

from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(max_depth=27)

model_rf.fit(X_train,y_train)
# Predicting the model

y_predict_rf = model_rf.predict(X_test)
# Finding accuracy, precision, recall and confusion matrix

print(accuracy_score(y_test,y_predict_rf))

print(classification_report(y_test,y_predict_rf))
confusion_matrix(y_test,y_predict_rf)
# Training the model

from sklearn.neighbors import KNeighborsClassifier

model_knn = KNeighborsClassifier(n_neighbors=12,metric='euclidean') # Maximum accuracy for n=10

model_knn.fit(X_train,y_train)
# Predicting the model

y_predict_knn = model_knn.predict(X_test)
# Finding accuracy, precision, recall and confusion matrix

print(accuracy_score(y_test,y_predict_knn))

print(classification_report(y_test,y_predict_knn))
confusion_matrix(y_test,y_predict_knn)
# Training the model

from xgboost import XGBClassifier

model_xgb = XGBClassifier()

model_xgb.fit(X_train,y_train)
# Predicting the model

y_predict_xgb = model_xgb.predict(X_test)
# Finding accuracy, precision, recall and confusion matrix

print(accuracy_score(y_test,y_predict_xgb))

print(classification_report(y_test,y_predict_xgb))
confusion_matrix(y_test,y_predict_xgb)
# Training the model

from sklearn.neural_network import MLPClassifier

model_mlp = MLPClassifier()

model_mlp.fit(X_train,y_train)
# Predicting the model

y_predict_mlp = model_mlp.predict(X_test)
# Finding accuracy, precision, recall and confusion matrix

print(accuracy_score(y_test,y_predict_mlp))

print(classification_report(y_test,y_predict_mlp))
confusion_matrix(y_test,y_predict_mlp)
# Exporting the trained model

joblib.dump(model_rf,'Predicting_Priority.ml')
# Selecting the predictors

X1 = data.loc[:,['CI_Subcat','WBS','Priority','Category','No_of_Related_Interactions','No_of_Related_Incidents']]

y1 = data.No_of_Related_Changes
X1.head(2)
enc= LabelEncoder()

for i in (0,1,3,4):

    X1.iloc[:,i] = enc.fit_transform(X1.iloc[:,i])
# Splitting into train and test for calculating the accuracy

X1_train, X1_test, y1_train, y1_test = train_test_split(X1,y1,test_size=0.3,random_state=10)
# Standardization technique is used

sc = StandardScaler()

X1_train = sc.fit_transform(X1_train)

X1_test = sc.transform(X1_test)
X1_train.shape
X1_test.shape
# Training the model

from sklearn.tree import DecisionTreeClassifier

model1_dtree=DecisionTreeClassifier()

model1_dtree.fit(X1_train,y1_train)
# Predicting the model

y1_predict_dtree = model1_dtree.predict(X1_test)
# Finding accuracy, precision, recall and confusion matrix

print(accuracy_score(y1_test,y1_predict_dtree))

print(classification_report(y1_test,y1_predict_dtree))
confusion_matrix(y1_test,y1_predict_dtree)
# Training the model

from sklearn.ensemble import RandomForestClassifier

model1_rf = RandomForestClassifier()

model1_rf.fit(X1_train,y1_train)
# Predicting the model

y1_predict_rf = model1_rf.predict(X1_test)
# Finding accuracy, precision, recall and confusion matrix

print(accuracy_score(y1_test,y1_predict_rf))

print(classification_report(y1_test,y1_predict_rf))
confusion_matrix(y1_test,y1_predict_rf)
# Exporting the trained model

joblib.dump(model1_dtree,'Predict_RFC.ml')
# Imporing the necessary columns

incfrq = data.loc[:,['Incident_ID','Open_Time']]
incfrq.head()
# Coverting all the values in proper Datetime format

for i in range(len(incfrq.Open_Time)):

    if (incfrq.Open_Time[i][1]=='/'):

        incfrq.Open_Time[i] = dt.datetime.strptime(incfrq.Open_Time[i],'%d/%m/%Y %H:%M').date()

    elif (incfrq.Open_Time[i][2]=='/'):

        incfrq.Open_Time[i] = dt.datetime.strptime(incfrq.Open_Time[i],'%d/%m/%Y %H:%M').date()

    else:

        incfrq.Open_Time[i] = dt.datetime.strptime(incfrq.Open_Time[i],'%d-%m-%Y %H:%M').date()
incfrq.head()
# Adding a new column which will have the number of tickets per day

incfrq['No_Incidents'] = incfrq.groupby('Open_Time')['Incident_ID'].transform('count')
incfrq.drop(['Incident_ID'],axis=1,inplace=True)

incfrq.drop_duplicates(inplace=True)
incfrq.head(3)
# Setting Date as the Index

incfrq = incfrq.set_index('Open_Time')

incfrq.index = pd.to_datetime(incfrq.index)

incfrq.index
incfrq.head()
# Checking range of dates for our values

print(incfrq.index.min(),'to',incfrq.index.max())
# Making a new Series with frequency as Day

data1 = incfrq['No_Incidents']

data1 = data1.asfreq('D')

data1.index
data1.head()
# Plotting number of tickets per day

data1.plot(figsize=(15,6))

plt.show()
# Since not many tickets before October 2013, we consider only the latter values

incfrom2013 = incfrq[incfrq.index > dt.datetime(2013,10,1)]
incfrom2013.head()
# new Series

data2 = incfrom2013['No_Incidents']

data2 = data2.asfreq('D')

data2.index
# Plotting number of tickets per day after October 2013

data2.plot(figsize=(15,6))

plt.show()
# Making a list of values for p,d & q

p = d = q = range(0,2)

pdq = list(itertools.product(p,d,q))
# Checking the AIC values per pairs

for param in pdq:

    mod = sm.tsa.statespace.SARIMAX(data2,order=param,enforce_stationarity=False,enforce_invertibility=False)

    results = mod.fit()

    print('ARIMA{} - AIC:{}'.format(param, results.aic))
# Choosing the model with minimum AIC and the ARIMA Model for Time Series Forecasting

mod = sm.tsa.statespace.SARIMAX(data2,order=(1,1,1))

results = mod.fit()

print(results.summary().tables[1])
# Predicting the future values and the confidence interval

pred = results.get_prediction(start=pd.to_datetime('2014-3-3'),end=pd.to_datetime('2014-10-30'),dynamic=False)

pred_ci = pred.conf_int()

pred.predicted_mean.round()
ax = data2['2013':].plot(label='observed')

pred.predicted_mean.plot(ax=ax,label='One-step ahead Forecast',figsize=(15, 6))

ax.fill_between(pred_ci.index,pred_ci.iloc[:,0],pred_ci.iloc[:,1],color='grey',alpha=0.3)

ax.set_xlabel('Date')

ax.set_ylabel('No of Incidents')

plt.legend()

plt.show()