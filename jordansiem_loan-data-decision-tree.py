#Data from lending club.dot - ppl who want money and investors - predict if people will pay money back
#Predict not fully paid - paid back or not
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



loans = pd.read_csv("../input/loan_data.csv")
loans.info()
loans.describe()
loans.head()
plt.figure(figsize=(10,6))

loans[loans['credit.policy']==1]['fico'].hist(bins=35,color='blue',label='Credit Policy = 1',alpha=0.6)

loans[loans['credit.policy']==0]['fico'].hist(bins=35,color='red',label='Credit Policy = 0',alpha=0.6)

plt.legend()

plt.xlabel('FICO')
#Credit Policy is if meets underwriting criteria, 0 is otherwise.
plt.figure(figsize=(10,6))

loans[loans['not.fully.paid']==1]['fico'].hist(bins=35,color='blue',label='Not Fully Paid= 1',alpha=0.6)

loans[loans['not.fully.paid']==0]['fico'].hist(bins=35,color='red',label='Not Fully Paid = 0',alpha=0.6)

plt.legend()

plt.xlabel('FICO')
#Majority of ppl have 0 or are fully paying off....spikes are how FICO score works more common
plt.figure(figsize=(11,7))

sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')
sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')
#Higher FICO score lower interest rates
plt.figure(figsize=(11,7))



sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',col='not.fully.paid',palette='Set1')
loans.info()
cat_feats = ['purpose']
final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)
final_data.info()
#drop first used to stop collinearity
from sklearn.model_selection import train_test_split
X = final_data.drop('not.fully.paid',axis=1)

y = final_data['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)



from sklearn.metrics import classification_report, confusion_matrix





print(classification_report(y_test,predictions))



print(confusion_matrix(y_test,predictions))