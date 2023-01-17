import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df=pd.read_csv('../input/HR_comma_sep.csv')

df.head()

               
df.shape
#checking data types of features

df.dtypes
df=df.rename(columns={'sales':'department',

                      'promotion_last_5years':'promotion'})
#left rate

left_rate=df.left.value_counts()/len(df)

left_rate
%matplotlib inline

#left_rate.plot(kind='bar');

locations=[1,2]

labels=['stayed','left']

plt.bar(locations,left_rate,tick_label=labels,alpha=0.7);

plt.title('stayed vs left');
#overview for those who left  vs those who stayed

left_summary=df.groupby('left')

left_summary.mean()
# correlation matrix

corr=df.corr()

corr
#heatmap ofcorrelation

sns.heatmap(corr);
sns.countplot(x='salary',hue='left',data=df).set_title('Distribution of salary');

sns.countplot(x='department',data=df).set_title('distribution of employees department wise');

plt.xticks(rotation=90);
sns.countplot(x='department',hue='left',data=df).set_title('distribution of employee who left department wise');

plt.xticks(rotation=90);
x=df.groupby('left')['satisfaction_level']

x1=x.mean()

x1
sns.countplot(x='number_project', hue='left', data=df);
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

#mapping of string value to integer categorical value

df['department']=df['department'].astype('category').cat.codes

df['salary']=df['salary'].astype('category').cat.codes

df.head()
Y=df['left']

X=df.drop('left',axis=1)

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)



#using logistic Regression

reg=LogisticRegression()

reg=reg.fit(x_train,y_train)

y_predict=reg.predict(x_test)

print ('logistic regression accuracy score:',accuracy_score(y_test,y_predict))



#using DecisionTree

clf=DecisionTreeClassifier(random_state=42)

clf.fit(x_train,y_train)

y_predict=clf.predict(x_test)

print('Decision Tree accuracy score',accuracy_score(y_test,y_predict))



#using Random Forest 

Rtree=RandomForestClassifier()

Rtree.fit(x_train,y_train)

y_predict=Rtree.predict(x_test)

print('Random forest accuracy score',accuracy_score(y_test,y_predict))



importances=Rtree.feature_importances_

indices=np.argsort(importances)[::-1]

print ('feature ranking:')

for i in range(X.shape[1]):

     print ("feature no. {}: {} ({})".format(i+1,X.columns[indices[i]],importances[indices[i]]))
plt.figure(figsize=(10,6));

plt.bar(range(len(indices)),importances[indices],color='red',alpha=0.5,tick_label=X.columns[indices]);

plt.xticks(rotation='-75');