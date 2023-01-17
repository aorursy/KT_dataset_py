import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import plotly.express as px

import cufflinks as cf

cf.go_offline()

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split 

from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics
# Data Reading

train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/train.csv")

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/test.csv")

train.head(3)
# Data Cleaning

train.drop(['County','Province_State'],axis=1,inplace=True)

test.drop(['County','Province_State'],axis=1,inplace=True)



# Changing the Time Format

train['Date']=pd.to_datetime(train['Date'])

train['Month']=train['Date'].apply(lambda x :x.month)

train['Day']=train['Date'].apply(lambda x :x.day)

train.drop(['Date'],axis=1,inplace=True)



test['Date']=pd.to_datetime(test['Date'])

test['Month']=test['Date'].apply(lambda x :x.month)

test['Day']=test['Date'].apply(lambda x :x.day)

test.drop(['Date'],axis=1,inplace=True)

train.head(3)
# For Country

le1=LabelEncoder()

le1.fit(train['Country_Region'])

train['Encoded_Country']=le1.transform(train['Country_Region'])



# For Target

le2=LabelEncoder()

le2.fit(train['Target'])

train['Encoded_Target']=le2.transform(train['Target'])

train.head(3)
y=train['TargetValue']

X=train[['Encoded_Country','Encoded_Target','Weight','Month','Day']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf=RandomForestRegressor()

rf.fit (X_train,y_train)

predictions=rf.predict(X_test)



print("MAE is ", metrics.mean_absolute_error(y_test,predictions))

print("MSE is ", metrics.mean_squared_error(y_test,predictions))

print("RMSE is ", np.sqrt(metrics.mean_squared_error(y_test,predictions)))
plt.figure(figsize=(8,6))

plt.plot(y_test,y_test,color='r')

plt.scatter(y_test,predictions,color='b')

plt.xlabel('Actual Target Value',fontsize=15)

plt.ylabel('Predicted Target Value',fontsize=15)

plt.title('Actual vs. Predicted Target Value',fontsize=20)

plt.show()
metrics.r2_score(y_test,predictions)
# Country

le3=LabelEncoder()

le3.fit(test['Country_Region'])

test['Encoded_Country']=le3.transform(test['Country_Region'])



# Target

le4=LabelEncoder()

le4.fit(test['Target'])



test['Encoded_Target']=le4.transform(test['Target'])

test=test[['Encoded_Country','Encoded_Target','Weight','Month','Day']]
pred=rf.predict(test)
test_1=pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')

output=pd.DataFrame({'Id':test_1['ForecastId'], 'TargetValue':pred})



a=output.groupby(['Id']).quantile(q=0.05).reset_index()

b=output.groupby(['Id']).quantile(q=0.5).reset_index()

c=output.groupby(['Id']).quantile(q=0.95).reset_index()



a.columns=['Id','q0.05']

b.columns=['Id','q0.5']

c.columns=['Id','q0.95']

a['q0.5']=b['q0.5']

a['q0.95']=c['q0.95']

sub=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])



sub['variable']=sub['variable'].apply(lambda x: x.replace('q',''))

sub['var']=sub['variable'].apply(lambda x: str(x))

sub['id']=sub['Id'].apply(lambda x: str(x))

sub['ForecastId_Quantile']=sub['id']+'_'+sub['var']

sub.drop(['Id','variable','var','id'],axis=1,inplace=True)

sub.columns=['TargetValue','ForecastId_Quantile']

sub=sub[['ForecastId_Quantile','TargetValue']]
sub.to_csv("submission.csv",index=False)
sub