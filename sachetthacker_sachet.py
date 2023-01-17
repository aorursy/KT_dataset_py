import pandas as pd

import numpy as np

from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/health-insurance-cross-sell-prediction/train.csv')
df.head(1)
df2 = df["Response"].map({1: 'Intrested' , 0 : 'Not Intrested' })
df3 = df.drop(columns =['Response'])
df3.head(2)
df4 = pd.concat([df2 , df3] ,axis = 'columns')
df4.head(2)
intrested = df4[df4.Response=="Intrested"]

not_intrested = df4[df4.Response == 'Not Intrested']
intrested.shape
not_intrested.shape
df4.groupby('Response').mean()
pd.crosstab(df4.Driving_License,df4.Response).plot(kind='bar')
pd.crosstab(df4.Gender,df4.Response).plot(kind='bar')

pd.crosstab(df4.Vehicle_Age,df4.Response).plot(kind='bar')
pd.crosstab(df4.Vehicle_Damage,df4.Response).plot(kind='bar')
pd.crosstab(df4.Previously_Insured,df4.Response).plot(kind='bar')
intrested.groupby('Gender').count()['id'].plot(kind = 'pie')

intrested['Gender'].value_counts()
intrested.groupby('Driving_License').count()['id'].plot(kind = 'pie')

intrested['Driving_License'].value_counts()
intrested.groupby('Previously_Insured').count()['id'].plot(kind = 'pie')

intrested['Previously_Insured'].value_counts()
intrested.groupby('Vehicle_Age').count()['id'].plot(kind = 'pie')

intrested['Vehicle_Age'].value_counts()
intrested.groupby('Vehicle_Damage').count()['id'].plot(kind = 'pie')

intrested['Vehicle_Damage'].value_counts()
df.head(1)
dummies_Gender = pd.get_dummies(df.Gender, prefix = "Gender")

dummies_License = pd.get_dummies(df.Gender, prefix = "Driving_License")

dummies_Insured = pd.get_dummies(df.Previously_Insured, prefix = "Previously_Insured")

dummies_age = pd.get_dummies(df.Vehicle_Age, prefix = "Vehicle_Age")

dummies_damage = pd.get_dummies(df.Vehicle_Damage, prefix = "Vehicle_Damage")

subdf = df[['Age' , 'Annual_Premium']]
final_df = pd.concat([subdf, dummies_Gender ,dummies_License ,  dummies_Insured ,dummies_age , dummies_damage ],  axis = 'columns')
final_df.head(1)
X = final_df
y = df.Response
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.3)

model = LogisticRegression()

model.fit(X_train, y_train)
model.score(X_test,y_test)
df_test = pd.read_csv('../input/health-insurance-cross-sell-prediction/test.csv')

dummies_Gender_tst = pd.get_dummies(df_test.Gender, prefix = "Gender")

dummies_License_tst = pd.get_dummies(df_test.Gender, prefix = "Driving_License")

dummies_Insured_tst = pd.get_dummies(df_test.Previously_Insured, prefix = "Previously_Insured")

dummies_age_tst = pd.get_dummies(df_test.Vehicle_Age, prefix = "Vehicle_Age")

dummies_damage_tst = pd.get_dummies(df_test.Vehicle_Damage, prefix = "Vehicle_Damage")

subdf_tst = df_test[['Age' , 'Annual_Premium']]
final_df_tst = pd.concat([subdf_tst, dummies_Gender_tst ,dummies_License_tst ,  dummies_Insured_tst ,dummies_age_tst , dummies_damage_tst ],  axis = 'columns')
final_predictions = model.predict(final_df_tst)
final_predictions