import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import time
import datetime
df_seattle=pd.read_csv("../input/seattleWeather_1948-2017.csv")
df_seattle.info()
df_seattle.head()
df_seattle.describe()
plt.figure(figsize=(10,10))
sns.heatmap(pd.isnull(df_seattle),yticklabels=False)
df_seattle[pd.isnull(df_seattle['PRCP'])]
df_seattle[pd.isnull(df_seattle['RAIN'])]
sns.countplot(data=df_seattle, x='RAIN')
df_seattle['PRCP'].mean()
def RAIN_INSERTION(cols):
    RAIN=cols[0]
    if pd.isnull(RAIN):
        return 'False'
    else:
        return RAIN
def PRCP_INSERTION(col):
    PRCP=col[0]
    if pd.isnull(PRCP):
        return df_seattle['PRCP'].mean()
    else:
        return PRCP
df_seattle['RAIN']=df_seattle[['RAIN']].apply(RAIN_INSERTION,axis=1)
df_seattle['PRCP']=df_seattle[['PRCP']].apply(PRCP_INSERTION,axis=1)
df_seattle[pd.isnull(df_seattle['RAIN'])]
df_seattle[pd.isnull(df_seattle['PRCP'])]
plt.figure(figsize=(7,7))
plt.scatter(x='TMIN',y='PRCP',data=df_seattle)
plt.xlabel('Minimum Temperature')
plt.ylabel('PRCP')
plt.title('Precipitation Vs Minimum Temperature')

plt.figure(figsize=(7,7))
plt.scatter(x='TMAX',y='PRCP',data=df_seattle)
plt.xlabel('Maximum Temperature')
plt.ylabel('PRCP')
plt.title('Precipitation Vs Maximum Temperature')
sns.distplot(df_seattle['TMIN'])
sns.distplot(df_seattle['TMAX'])
sns.pairplot(data=df_seattle)
#plt.figure(figsize=(10,7))
sns.boxplot(data=df_seattle)
#Dropping the outliers from TMIN column
df_seattle=df_seattle.drop(df_seattle[df_seattle['TMIN']<17 ].index)

#Dropping the outliers from TMAX columns i.e. the value more than 100
df_seattle=df_seattle.drop(df_seattle[(df_seattle['TMAX']>97.5) | (df_seattle['TMAX']< 21.5)].index)
#Dropping the outliers from PRCP columns i.e. the value more than 0.275
df_seattle=df_seattle.drop(df_seattle[(df_seattle['PRCP']>0.25) | (df_seattle['PRCP']< -0.15) ].index)
sns.boxplot(data=df_seattle)
from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()
from sklearn.model_selection import train_test_split
X=df_seattle.drop(['RAIN','DATE'],axis=1)
y=df_seattle['RAIN']
y=y.astype('str')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
lr.fit(X_train,y_train)
prediction=lr.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print('Confusion Matrix',confusion_matrix(y_test,prediction))
print('\n')
print('Classification Report',classification_report(y_test,prediction))