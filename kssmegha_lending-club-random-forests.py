import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/loan_data.csv')
df.info()
df.describe()
df.head()
sns.set_style('whitegrid')

plt.figure(figsize=(10,6))

df[df['credit.policy']==0]['fico'].plot.hist(bins=30,alpha=0.5,color='blue',label='Credit Policy==0')

df[df['credit.policy']==1]['fico'].plot.hist(bins=30,alpha=0.5,color='red',label='Credit Policy==1')

plt.legend()

plt.xlabel('FICO')
# More people with credit score = 1 than with credit score = 0

# People with less than 660(approx) FICO score will not meet the underwriting criteria
sns.set_style('whitegrid')

plt.figure(figsize=(10,6))

df[df['not.fully.paid']==0]['fico'].plot.hist(bins=30,alpha=0.5,color='blue',label='Not fully paid==0')

df[df['not.fully.paid']==1]['fico'].plot.hist(bins=30,alpha=0.5,color='red',label='Not fully paid==1')

plt.legend()

plt.xlabel('FICO')
# More people fully paid their loan

# People with less than 660(approx) FICO score have not fully paid their loans
plt.tight_layout()

plt.figure(figsize=(10,8))

sns.countplot(x='purpose',hue='not.fully.paid',data=df)


sns.jointplot(x='fico',y='int.rate',data=df)
plt.figure(figsize=(11,7))

sns.lmplot(x='fico',y='int.rate',data=df,col='not.fully.paid',hue='credit.policy',palette='Set1')
#As the FICO score increases, better credit so the interest to be paid is decreases

df.info()
cat_feats=['purpose']
final_data = pd.get_dummies(df,columns=cat_feats,drop_first=True)
final_data.head()
from sklearn.model_selection import train_test_split

X=final_data.drop('not.fully.paid',axis=1)

y=final_data['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=101)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(predictions,y_test))

print(confusion_matrix(predictions,y_test))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)
r_pred = rfc.predict(X_test)
print(classification_report(r_pred,y_test))

print('\n')

print(confusion_matrix(r_pred,y_test))
#Random Forest with precision of 100%