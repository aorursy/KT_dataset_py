import warnings
warnings.filterwarnings("ignore")
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
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import mean_squared_error     
import matplotlib.dates as dates
import datetime as dt
train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
test=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
submission=pd.read_csv('/kaggle/input/result/covid_submission_13772.csv')
train.head(11)
test.head(11)
train.describe()
train.info()
test.info()
import matplotlib.pyplot as plt
%matplotlib inline
train.plot(kind="scatter", y="Fatalities", x="Country_Region")
plt.xlabel('Country_Region')
plt.ylabel('Fatalities');
train.plot(kind="scatter", y="Fatalities", x="Date")
plt.xlabel('Date')
plt.ylabel('Fatalities');
plt.figure(figsize=(10,15))
train.plot(kind="scatter", x="Country_Region", y="ConfirmedCases")
plt.ylabel('Country_Region')
plt.xlabel('ConfirmedCases');
plt.figure(figsize=(20,10))
plt.plot(train.Id, train.ConfirmedCases)
plt.title('Confirmed Cases')
plt.show()
X_train=train[['Id']]
test['Id']=test['ForecastId']
X_test=test[['Id']]
y_train_cc=train[['ConfirmedCases']]
y_train_ft=train[['Fatalities']]
X_tr=np.array_split(X_train,313)
y_cc=np.array_split(y_train_cc,313)
y_ft=np.array_split(y_train_ft,313)
X_te=np.array_split(X_test,313)
a=np.max(X_tr[0]).values
b=a-71
b=b[0]
train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
test=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
submission=pd.read_csv('/kaggle/input/result/covid_submission_13772.csv')
import matplotlib.pyplot as plt
%matplotlib inline
X_train=train[['Id']]
test['Id']=test['ForecastId']
X_test=test[['Id']]
y_train_cc=train[['ConfirmedCases']]
y_train_ft=train[['Fatalities']]
X_tr=np.array_split(X_train,313)
y_cc=np.array_split(y_train_cc,313)
y_ft=np.array_split(y_train_ft,313)
X_te=np.array_split(X_test,313)
a=np.max(X_tr[0]).values
b=a-71
b=b[0]
X_te[0]=X_te[0]+a
for i in range (312):
    X_te[i+1]=X_te[0] 
for i in range (312):
    X_tr[i+1]=X_tr[0] 
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(2)
y_pred_cc=[]
for i in range (313): #for loop is used to iterate through different regions
    X_tr[i]=poly.fit_transform(X_tr[i])
    X_te[i]=poly.fit_transform(X_te[i])
    model=Lasso()
    model.fit(X_tr[i],y_cc[i]);
    y_pr_cc=model.predict(X_te[i])
    
    y_cc[i]= y_cc[i][71:]
    y_pr_cc=y_pr_cc[b:]
    y_pr_cc=np.append(y_cc[i], y_pr_cc)
    
    y_pred_cc.append(y_pr_cc);
   

y_pred_ft=[]
for i in range (313): #for loop is used to iterate through different regions
    model=Lasso()
    model.fit(X_tr[i],y_ft[i]);
    y_pr_ft=model.predict(X_te[i])
    
    y_ft[i]= y_ft[i][71:]
    y_pr_ft=y_pr_ft[b:]
    y_pr_ft=np.append(y_ft[i], y_pr_ft)
   
    y_pred_ft.append(y_pr_ft);
y_pred_ft_1 = [item for sublist in y_pred_ft for item in sublist]
y_pred_cc_1 = [item for sublist in y_pred_cc for item in sublist]
#print(len(y_pred_cc_1))
result=pd.DataFrame({'ForecastId':submission.ForecastId, 'ConfirmedCases':np.round(y_pred_cc_1), 'Fatalities':np.round(y_pred_ft_1)})
#result.to_csv('/kaggle/input/submission/covid_submission_13772.csv', index=False)
#data=pd.read_csv('/kaggle/input/submission/covid_submission_13772.csv')
result.head(50)