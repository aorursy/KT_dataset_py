import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df=pd.read_csv('/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv')
df.head()
#Making a copy of the dataframe loaded

df_copy=df.copy()
print('Rows:',df.shape[0])

print('Columns:',df.shape[1])
df.info()
df.describe()
df.describe(include='object')
df['WindDir9am'].value_counts(normalize=True).plot(kind='bar')
df['WindDir3pm'].value_counts(normalize=True).plot(kind='bar')
df['WindGustDir'].value_counts(normalize=True).plot(kind='bar')
#Checking for missing values

df.isnull().sum()
# Missing value impuatation

df['MinTemp']=df['MinTemp'].fillna(df['MinTemp'].median())

df['MaxTemp']=df['MaxTemp'].fillna(df['MaxTemp'].median())

df['Rainfall']=df['Rainfall'].fillna(df['Rainfall'].median())

df['Evaporation']=df['Evaporation'].fillna(method='ffill')

df['Sunshine']=df['Sunshine'].fillna(method='ffill')

df['WindDir9am']=df['WindDir9am'].fillna(method='ffill')

df['WindDir3pm']=df['WindDir3pm'].fillna(method='ffill')

df['Humidity3pm']=df['Humidity3pm'].fillna(df['Humidity3pm'].median())

df['Humidity9am']=df['Humidity9am'].fillna(df['Humidity9am'].median())

df['Pressure3pm']=df['Pressure3pm'].fillna(method='ffill')

df['Pressure9am']=df['Pressure9am'].fillna(method='ffill')

df['Cloud3pm']=df['Cloud3pm'].fillna(method='ffill')

df['Cloud9am']=df['Cloud9am'].fillna(method='ffill')

df['Temp9am']=df['Temp9am'].fillna(df['Temp9am'].median())

df['Temp3pm']=df['Temp3pm'].fillna(df['Temp3pm'].median())

df['Evaporation']=df['Evaporation'].fillna(df['Evaporation'].median())

df['Sunshine']=df['Sunshine'].fillna(df['Sunshine'].median())

df['WindGustSpeed']=df['WindGustSpeed'].fillna(df['WindGustSpeed'].median())

df['WindSpeed3pm']=df['WindSpeed3pm'].fillna(df['WindSpeed3pm'].median())

df['WindSpeed9am']=df['WindSpeed9am'].fillna(df['WindSpeed9am'].median())

df['RainToday']=df['RainToday'].fillna('No')

df['WindGustDir']=df['WindGustDir'].fillna('W')
#Checking for missing values again

df.isnull().sum()
df['Cloud3pm']=df['Cloud3pm'].fillna(df['Cloud3pm'].median())
#Checking for missing values again

df.isnull().sum()
#checking for outliers

df.plot(kind='box',subplots=True,layout=(9,2),figsize=(16,12))
# Detecting outlier's limit(MaxTemp)

#IQR=Q3-Q1

iqr=28.2-17.9

max_u=28.2+1.5*(iqr)

max_l=17.9-1.5*(iqr)

print('Upper Limit:',max_u)

print('Lower Limit:',max_l)
# Capping Values greater than 43.65 to 43.65 and lesser than 2.45 to 2.45

df.loc[df.MaxTemp>43.65,'MaxTemp'] = 43.65
df.loc[df.MaxTemp<2.45,'MaxTemp'] = 2.45
# Detecting outlier's limit(MinTemp)

iqr=16.8-7.6

min_u=16.8+1.5*(iqr)

min_l=7.6-1.5*(iqr)

print('Upper Limit:',min_u)

print('Lower Limit:',min_l)
# Capping Values greater than 30.6 to 30.6 and lesser than -6.2 to -6.2

df.loc[df.MinTemp>30.6,'MinTemp'] = 30.6

df.loc[df.MinTemp<-6.2,'MinTemp'] = -6.2
# Detecting outlier's limit(RainFall)

iqr=0.8-0

rain_u=0.8+1.5*(iqr)

rain_l=0-1.5*(iqr)

print('Upper Limit:',rain_u)

print('Lower Limit:',rain_l)
# Capping Values greater than 2.0 to 2.0 and lesser than -1.2 to -1.2

df.loc[df.Rainfall>2.0,'Rainfall'] = 2.0

df.loc[df.Rainfall<-1.2,'Rainfall'] = -1.2
# Detecting outlier's limit(Evaporation)

iqr=7.4-2.6

ev_u=7.4+1.5*(iqr)

ev_l=2.6-1.5*(iqr)

print('Upper Limit:',ev_u)

print('Lower Limit:',ev_l)
# Capping Values greater than 14.6 to 14.6 and lesser than -4.6 to -4.6

df.loc[df.Evaporation>14.6,'Evaporation'] = 14.6

df.loc[df.Evaporation<-1.2,'Evaporation'] = -1.2
# Detecting outlier's limit(WindGustSpeed)

iqr=48-31

wg_u=48+1.5*(iqr)

wg_l=31-1.5*(iqr)

print('Upper Limit:',wg_u)

print('Lower Limit:',wg_l)
# Capping Values greater than 73.5 to 73.5 and lesser than 5.5 to 5.5

df.loc[df.WindGustSpeed>73.5,'WindGustSpeed'] = 73.5

df.loc[df.WindGustSpeed<5.5,'WindGustSpeed'] = 5.5
# Detecting outlier's limit(WindSpeed9am)

iqr=19-7

u=19+1.5*(iqr)

l=7-1.5*(iqr)

print('Upper Limit:',u)

print('Lower Limit:',l)
# Capping Values greater than 37.0 to 37.0 and lesser than -11.0 to -11.0

df.loc[df.WindSpeed9am>37.0,'WindSpeed9am'] = 37.0

df.loc[df.WindSpeed9am<-11.0,'WindSpeed9am'] = -11.0
# Detecting outlier's limit(WindSpeed3pm)

iqr=24.0-13.0

u=24.0+1.5*(iqr)

l=13.0-1.5*(iqr)

print('Upper Limit:',u)

print('Lower Limit:',l)
# Capping Values greater than 40.5 to 40.5 and lesser than -3.5 to -3.5

df.loc[df.WindSpeed3pm>40.5,'WindSpeed3pm'] = 40.5

df.loc[df.WindSpeed3pm<-3.5,'WindSpeed3pm'] = -3.5
# Detecting outlier's limit(humidity9am)

iqr=83.0-57.0

u=83.0+1.5*(iqr)

l=57.0-1.5*(iqr)

print('Upper Limit:',u)

print('Lower Limit:',l)
# Capping Values 

df.loc[df.Humidity9am>122.0,'Humidity9am'] = 122.0

df.loc[df.Humidity9am<18.0,'Humidity9am'] = 18.0
# Detecting outlier's limit(Pressure9am)

iqr=1022.4-1012.9

u=1022.4+1.5*(iqr)

l=1012.9-1.5*(iqr)

print('Upper Limit:',u)

print('Lower Limit:',l)
# Capping Values 

df.loc[df.Pressure9am>1036.65,'Pressure9am'] = 1036.65

df.loc[df.Pressure9am<998.65,'Pressure9am'] = 998.65
# Detecting outlier's limit(Pressure3pm)

iqr=1020.0-1010.4

u=1020.0+1.5*(iqr)

l=1010.4-1.5*(iqr)

print('Upper Limit:',u)

print('Lower Limit:',l)
# Capping Values 

df.loc[df.Pressure3pm>1034.4,'Pressure3pm'] = 1034.4

df.loc[df.Pressure3pm<996.0,'Pressure3pm'] = 996.0
# Detecting outlier's limit(Temperature9am)

iqr=21.6-12.3

u=21.6+1.5*(iqr)

l=12.3-1.5*(iqr)

print('Upper Limit:',u)

print('Lower Limit:',l)
# Capping Values 

df.loc[df.Temp9am>35.55,'Temp9am'] = 35.55

df.loc[df.Temp9am<-1.65,'Temp9am'] = -1.65
# Detecting outlier's limit(Temperature3pm)

iqr=26.4-16.6

u=26.4+1.5*(iqr)

l=16.6-1.5*(iqr)

print('Upper Limit:',u)

print('Lower Limit:',l)
# Capping Values 

df.loc[df.Temp3pm>41.1,'Temp3pm'] = 41.1

df.loc[df.Temp3pm<1.9,'Temp3pm'] = 1.9
#Using boxcox for RISK_MM

from scipy.stats import boxcox

l=list((boxcox(df.RISK_MM+1)[0]))

df['RISK_MM']=l
df.plot(kind='box',subplots=True,layout=(9,2),figsize=(16,12))
#We will convert the target variables to label 0 and 1.

df['RainTomorrow'].replace({'Yes':'1','No':'0'},inplace=True)

df['RainTomorrow']=df['RainTomorrow'].astype(int)
#checking for correlation

plt.figure(figsize=(12,10))

corr=df.corr()

sns.heatmap(corr,annot=True)
pd.crosstab(df['RainToday'],df['RainTomorrow']).plot(kind='bar')
sns.scatterplot(x='RISK_MM',y='RainTomorrow',data=df)
sns.scatterplot(x='Humidity3pm',y='RainTomorrow',data=df)
sns.scatterplot(x='Cloud9am',y='RainTomorrow',data=df)
sns.scatterplot(x='Cloud3pm',y='RainTomorrow',data=df)
l=['Cloud9am','Cloud3pm','Date','Location']

df.drop(l,axis=1,inplace=True)
#Creating dummies

df=pd.get_dummies(data=df,columns=['WindGustDir','WindDir9am','WindDir3pm','RainToday'])
#Creating a copy of our dataframe

df_new=df.copy()
# Defining X and y

X=df.drop('RainTomorrow',axis=1)

y=df['RainTomorrow']
#Train and test split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
#Model Building

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(X_train,y_train)
y_train_pred=lr.predict(X_train)

y_test_pred=lr.predict(X_test)

from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,roc_auc_score,roc_curve

print('Train Accuracy',accuracy_score(y_train,y_train_pred))

print('Test Accuracy',accuracy_score(y_test,y_test_pred))
from sklearn.metrics import cohen_kappa_score

cohen_kappa_score(y_test,y_test_pred)
print('Confusion Matrix(test)',confusion_matrix(y_train,y_train_pred))

cm=confusion_matrix(y_test,y_test_pred)

tn=cm[0,0]

tp=cm[1,1]

fn=cm[1,0]

fp=cm[0,1]

accuracy=(tp+tn)/(tp+tn+fp+fn)

misclassification_error=1-accuracy

sensitivity=tp/float(tp+fn)

specificity=tn/float(tn+fp)
print('The misclassifiction error is',misclassification_error)
print('Negative Liklihood ratio is',(1-sensitivity)/specificity)
print('Positive Liklihood ratio is',sensitivity/(1-specificity))