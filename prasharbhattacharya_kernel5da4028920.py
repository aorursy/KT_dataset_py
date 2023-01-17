import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

pd.options.display.max_columns = 9999
df = pd.read_csv('../input/cervical-cancer-risk-classification/kag_risk_factors_cervical_cancer.csv')

df.info()
df.isnull().sum()
df.head()
df[df=='?']=np.nan
df.isnull().sum()
from sklearn.preprocessing import Imputer

imputer_mean = Imputer(missing_values='NaN',strategy='mean')

imputer_most_frequent = Imputer(missing_values='NaN',strategy='most_frequent')
df.iloc[:,1:2] = imputer_most_frequent.fit_transform(df.iloc[:,1:2])

df.iloc[:,2:3] = imputer_most_frequent.fit_transform(df.iloc[:,2:3])

df.iloc[:,3:4] = imputer_most_frequent.fit_transform(df.iloc[:,3:4])

df.iloc[:,4:5] = imputer_most_frequent.fit_transform(df.iloc[:,4:5])

df.iloc[:,5:6] = imputer_most_frequent.fit_transform(df.iloc[:,5:6])

df.iloc[:,6:7] = imputer_most_frequent.fit_transform(df.iloc[:,6:7])

df.iloc[:,7:8] = imputer_most_frequent.fit_transform(df.iloc[:,7:8])

df.iloc[:,8:9] = imputer_most_frequent.fit_transform(df.iloc[:,8:9])

df.iloc[:,9:10] = imputer_most_frequent.fit_transform(df.iloc[:,9:10])

df.iloc[:,10:11] = imputer_most_frequent.fit_transform(df.iloc[:,10:11])

df.iloc[:,11:12] = imputer_most_frequent.fit_transform(df.iloc[:,11:12])

df.iloc[:,12:13] = imputer_most_frequent.fit_transform(df.iloc[:,12:13])

df.iloc[:,13:14] = imputer_most_frequent.fit_transform(df.iloc[:,13:14])

df.iloc[:,14:15] = imputer_most_frequent.fit_transform(df.iloc[:,14:15])

df.iloc[:,15:16] = imputer_most_frequent.fit_transform(df.iloc[:,15:16])

df.iloc[:,16:17] = imputer_most_frequent.fit_transform(df.iloc[:,16:17])

df.iloc[:,17:18] = imputer_most_frequent.fit_transform(df.iloc[:,17:18])

df.iloc[:,18:19] = imputer_most_frequent.fit_transform(df.iloc[:,18:19])

df.iloc[:,19:20] = imputer_most_frequent.fit_transform(df.iloc[:,19:20])

df.iloc[:,20:21] = imputer_most_frequent.fit_transform(df.iloc[:,20:21])

df.iloc[:,21:22] = imputer_most_frequent.fit_transform(df.iloc[:,21:22])

df.iloc[:,22:23] = imputer_most_frequent.fit_transform(df.iloc[:,22:23])

df.iloc[:,23:24] = imputer_most_frequent.fit_transform(df.iloc[:,23:24])

df.iloc[:,24:25] = imputer_most_frequent.fit_transform(df.iloc[:,24:25])

df.iloc[:,26:27] = imputer_most_frequent.fit_transform(df.iloc[:,26:27])

df.iloc[:,27:28] = imputer_most_frequent.fit_transform(df.iloc[:,27:28])
df.isnull().sum()
df.head()
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

sns.violinplot(y=df['Age'],x=df['Smokes'],hue=df['Biopsy'],split=True, ax=axis2)

sns.violinplot(y=df['Age'],x=df['Biopsy'],ax=axis3)

sns.violinplot(df['Age'],ax=axis1)
sns.barplot(y=df['Age'],x=df['Number of sexual partners'],hue=df['Biopsy'])
f, ax = plt.subplots(figsize=(20, 9))

sns.barplot(y=df['Age'],x=df['First sexual intercourse'],hue=df['Biopsy'])
f, ax = plt.subplots(figsize=(20, 9))

sns.barplot(y=df['Age'],x=df['Smokes (years)'])
f, ax = plt.subplots(figsize=(10, 5))

sns.barplot(df['Number of sexual partners'],df['Smokes (years)'],hue=df['Biopsy'])
sns.barplot(df['STDs:AIDS'],df['STDs:HIV'],hue=df['Biopsy'])
sns.violinplot(df['STDs:Hepatitis B'],df['Age'],hue=df['Biopsy'],split=True)
x = df.iloc[:,0:35].values

y = df.iloc[:,-1].values
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
from sklearn.metrics import r2_score,accuracy_score,confusion_matrix
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(accuracy_score(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))