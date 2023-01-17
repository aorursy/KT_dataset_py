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
df = pd.read_csv('../input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv')

df.head()
df.describe().T[1:10]
import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
fig,ax = plt.subplots(1,2, figsize=(15,10))

vc=df['inflation_crises'].value_counts()

#print(vc)

vc.plot(kind='pie',autopct='%1.1f', ax = ax[0])



vc=df['banking_crisis'].value_counts()

#print(vc)

vc.plot(kind='pie',autopct='%1.1f',ax = ax[1])

ax[0].set_title('Percentage of Inflation Crisis')

ax[1].set_title('Percentage of Banking Crisis')

plt.show()
fig,ax = plt.subplots(1,2, figsize=(15,10))

vc=df['currency_crises'].value_counts()

vc.plot(kind='pie',autopct='%1.1f', ax = ax[0])



vc=df['systemic_crisis'].value_counts()

vc.plot(kind='pie',autopct='%1.1f',ax = ax[1])

ax[0].set_title('Percentage of Currency Crisis')

ax[1].set_title('Percentage of Systemic Crisis')

plt.show()
plt.figure(figsize=(10,7))

sns.heatmap(df.corr(),annot=True, cmap='YlGnBu')

plt.show()
plt.figure(figsize=(12,7))

sns.barplot(data=df, x='country', y='case')

plt.show()
plt.figure(figsize=(14,8))

sns.scatterplot(data = df, y='year', x = 'inflation_crises',hue='inflation_crises')

plt.show()
plt.figure(figsize=(10,7))

sns.distplot(df['exch_usd'], bins=20, hist=True, kde=True, color='r')

plt.show()
plt.figure(figsize=(10,7))

sns.pointplot(y=df['exch_usd'] , x= df['inflation_crises'])

plt.title('Exchange Rate vs. Inflation Crisis')

plt.show()
topfive = df.groupby('country')['inflation_crises'].agg('sum').nlargest(5)



df['banking_crisis'] = df['banking_crisis'].map(lambda x : 1 if x == 'crisis' else 0)

topbank = df.groupby('country')['banking_crisis'].agg('sum').nlargest(5)
fig,ax = plt.subplots(1,2,figsize=(15,6))

topfive.plot.bar(color='m', ax = ax[0])

topbank.plot.bar(color='g', ax = ax[1])

ax[0].set_title('Top Five Countries of Inflation Crisis')

ax[1].set_title('Top Five Countries of Banking Crisis')

plt.show()
curr = df.groupby('country')['currency_crises'].agg('sum').nlargest(5)

sys = df.groupby('country')['systemic_crisis'].agg('sum').nlargest(5)

fig,ax = plt.subplots(1,2,figsize=(15,6))

curr.plot.bar(color='r', ax = ax[0])

sys.plot.bar(color='k', ax = ax[1])

ax[0].set_title('Top Five Countries of Currency Crisis')

ax[1].set_title('Top Five Countries of Systemic Crisis')

plt.show()
sns.regplot(data=df, y='inflation_crises', x='exch_usd')

plt.show()
plt.figure(figsize=(10,7))

df.set_index('exch_usd')['inflation_annual_cpi'].plot.line(color='r')

plt.show()
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier 

from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['country'] = le.fit_transform(df['country'])
df['cc3'] = le.fit_transform(df['cc3'])
X = df.drop('inflation_crises', axis=1)

y = df['inflation_crises']

cv_scr = cross_val_score(RandomForestClassifier(), X, y, cv = 10)

cv_scr
sum1 = 0

for i in cv_scr:

    sum1 = sum1+i

    i +=1

print("The average accuracy of Random Forest algorithm with 10 folds is", sum1*100/10, "percent")
cv_scr1 = cross_val_score(LogisticRegression(), X, y, cv = 10)

sum2 = 0

for i in cv_scr1:

    sum2 = sum2+i

    i +=1

print("The average accuracy of Logistic Regression algorithm with 10 folds is", sum2*100/10, "percent")
cv_scr2 = cross_val_score(DecisionTreeClassifier(), X, y, cv = 10)

sum3 = 0

for i in cv_scr2:

    sum3 = sum3+i

    i +=1

print("The average accuracy of Decision Tree algorithm with 10 folds is", sum3*100/10, "percent")
cv_scr3 = cross_val_score(KNeighborsClassifier(), X, y, cv = 10)

sum4 = 0

for i in cv_scr3:

    sum4 = sum4+i

    i +=1

print("The average accuracy of KNN algorithm with 10 folds is", sum4*100/10, "percent")
indices = ['Random Forest','Logistic Regression','Decision Tree', 'KNN']

bar = pd.DataFrame([sum1*10,sum2*10, sum3*10, sum4*10], index = indices)
bar.plot.bar(color='m')

plt.show()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=14)
model = RandomForestClassifier()

model.fit(X_train,y_train)
print("The accuracy of the model is",model.score(X_test,y_test)*100,"percent")
plt.figure(figsize=(10,7))

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

sns.heatmap(cm,annot=True)

plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=14)

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print("The accuracy of the model is",logreg.score(X_test,y_test)*100,"percent")
plt.figure(figsize=(10,7))

y_pred = logreg.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

sns.heatmap(cm,annot=True)

plt.show()
gini = DecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=14)

gini.fit(X_train, y_train)

y_pred = gini.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report

print('Accuracy Score: ',accuracy_score(y_test, y_pred)*100,"percent")
print('Classification Report')

print(classification_report(y_test, y_pred))
plt.figure(figsize=(10,7))

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

sns.heatmap(cm,annot=True)

plt.show()
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=14)
from sklearn.model_selection import GridSearchCV

grid = {'n_neighbors':np.arange(1,25)}

knn_cv = GridSearchCV(knn, grid, cv=5)

knn_cv.fit(X,y)

print("Tuned Parameters are: {}".format(knn_cv.best_params_))

print("Best Score is: {}".format(knn_cv.best_score_))
knn = KNeighborsClassifier(n_neighbors = 5)



knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

print("The accuracy score is", knn.score(X_test, y_test)*100,"percent")
plt.figure(figsize=(10,7))

cm = confusion_matrix(y_test,y_pred)

sns.heatmap(cm,annot=True)

plt.show()