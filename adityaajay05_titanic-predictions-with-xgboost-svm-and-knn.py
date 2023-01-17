# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/titanic/train.csv')
np.shape(df)
df.head(10)
df.describe()
df.mode()
sns.countplot(df['Survived'], label='number of survivors')
df.groupby('Sex')[['Survived']].mean()
df.groupby('SibSp')[['Survived']].mean()
sns.countplot(df['Pclass'], label='Passengers')
labels = ['First Class', 'Second Class', 'Third Class']

sizes = [202, 190, 500]

colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

explode = (0.1, 0, 0)



plt.pie(sizes, explode=explode, labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=140)



plt.axis('equal')

plt.show()
cols = ['Sex', 'SibSp', 'Parch', 'Embarked']



n_rows = 2

n_cols = 2



fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*5.0,n_rows*5.0))



for r in range(0,n_rows):

    for c in range(0,n_cols):  

        

        i = r*n_cols+ c      

        ax = axs[r][c] 

        sns.countplot(df[cols[i]], hue=df["Survived"], ax=ax)

        ax.set_title(cols[i])

        ax.legend(title="survived", loc='upper right') 

        

plt.tight_layout()
df.head()
X = df.iloc[:, [2, 4, 5, 6, 7, 9, 11]].values

y = df.iloc[:, 1].values
print(X)
print(pd.unique(X[:, 1]))
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X[:, 1] = le.fit_transform(X[:, 1])
print(pd.unique(X[:, 1]))
print(X)
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(X[:, :-1])

X[:, :-1] = imputer.transform(X[:, :-1])
print(X)
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

imputer.fit(X)

X = imputer.transform(X)
print(X)
pd.isnull(X)
print(X)
print(pd.unique(X[:, -1]))
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [-1])], remainder='passthrough')

X = np.array(ct.fit_transform(X))
print(X)
print(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit_transform(X_train, y_train)

sc.fit_transform(X_test)
print(X)
from xgboost import XGBClassifier

xgb = XGBClassifier()

xgb.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, confusion_matrix 

y_pred_xgb = xgb.predict(X_test)

cm = confusion_matrix(y_test, y_pred_xgb)

print(cm)

accuracy_score(y_test, y_pred_xgb)
from sklearn.svm import SVC

svm = SVC(kernel = 'rbf', random_state = 0)

svm.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, confusion_matrix 

y_pred_svm = svm.predict(X_test)

cm = confusion_matrix(y_test, y_pred_svm)

print(cm)

accuracy_score(y_test, y_pred_svm)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

knn.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix, accuracy_score

y_pred_knn = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred_knn)

print(cm)

accuracy_score(y_test, y_pred_knn)
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = xgb, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
X_td = test_data.iloc[:, [1, 3, 4, 5, 6, 8, 10]].values
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X_td[:, 1] = le.fit_transform(X_td[:, 1])
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(X_td[:, :-1])

X_td[:, :-1] = imputer.transform(X_td[:, :-1])
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

imputer.fit(X_td)

X_td = imputer.transform(X_td)
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [-1])], remainder='passthrough')

X_td = np.array(ct.fit_transform(X_td))
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit_transform(X_td)
y_pred = xgb.predict(X_td)
print(y_pred)
submission_predictions = pd.DataFrame({"PassengerId": test_data['PassengerId'], "Survived": y_pred})
print(submission_predictions)
submission_predictions.to_csv(r'C:\Users\adity.LAPTOP-F6A6F39F.000\Desktop\submission_predictions.csv', index = False, header=True)