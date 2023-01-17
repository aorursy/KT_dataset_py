# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

import matplotlib.pyplot as plt



df = pd.read_csv('../input/drug-classification/drug200.csv')
df.head()
df.describe()
plt.figure(figsize = (20,10))



sns.countplot(df['Age'])
sns.countplot(df['Sex'])
sns.countplot(df['BP'])
sns.countplot(df['Cholesterol'])
sns.countplot(df['Drug'], hue='Sex', data = df)
sns.boxplot(x = 'Sex', y = 'Na_to_K', data= df)
sns.boxplot(x = 'Drug', y = 'Na_to_K', data = df)
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()



for i in list(df.columns):

    if df[i].dtype=='object':

        df[i]=le.fit_transform(df[i])
plt.figure(figsize = (20,10))

sns.heatmap(df.corr(), annot = True)
df.head()
from sklearn.model_selection import train_test_split
X = df.drop('Drug',axis=1)

y = df['Drug']
X.head()
from sklearn import tree

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
features = list(df.columns[1:])
plt.figure(figsize = (20,10))



tree.plot_tree(dtree,feature_names=features,filled=True,rounded=True)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))
sns.heatmap(confusion_matrix(y_test, rfc_pred), annot=True)

print(classification_report(y_test,rfc_pred))
X = df.drop('Drug',axis=1)

y = df['Drug'] 
X.head()
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=0)
logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

sns.heatmap(confusion_matrix(y_test, predictions), annot=True)

print(classification_report(y_test, predictions))

from sklearn.model_selection import train_test_split

X = df.drop('Drug',axis=1)

y = df['Drug'] 

X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)



from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred))
sns.heatmap(confusion_matrix(y_test, pred), annot=True)

print(classification_report(y_test,pred))

error_rate = []



# Will take some time

for i in range(1,40):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
knn = KNeighborsClassifier(n_neighbors=23)



knn.fit(X_train,y_train)

pred = knn.predict(X_test)



print('WITH K=23')

print('\n')

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
from sklearn.model_selection import train_test_split

X = df.drop('Drug',axis=1)

y = df['Drug'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)
from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix
svc_model = SVC()

svc_model.fit(X_train, y_train)
y_predict = svc_model.predict(X_test)

cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot = True)

print(classification_report(y_test, y_predict))

min_train = X_train.min()

range_train = (X_train-min_train).max()

X_train_scaled = (X_train - min_train)/range_train



min_test = X_test.min()

range_test = (X_test - min_test).max()

X_test_scaled = (X_test - min_test)/range_test
param_grid = {'C' : [0.1, 1, 10, 100], 'gamma' : [1, .1, .01, .001], 'kernel' : ['rbf']}

from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 4)

grid.fit(X_train_scaled, y_train)

grid.best_params_

min_test = X_test.min()

range_test = (X_test - min_test).max()

X_test_scaled = (X_test - min_test)/range_test
grid_predictions = grid.predict(X_test_scaled)

cm = confusion_matrix(y_test, grid_predictions)
sns.heatmap(cm, annot=True)

print(classification_report(y_test, grid_predictions))