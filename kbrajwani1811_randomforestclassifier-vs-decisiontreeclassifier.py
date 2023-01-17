import numpy as np # linear algebra

np.random.seed(42)

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.ensemble import BaggingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report



dataset = pd.read_csv("../input/train.csv")



dataset.head()
dataset.info()
dataset.describe()
dataset.corr()
import seaborn as sns

import matplotlib.pyplot as plt
sns.heatmap(dataset.corr(),annot=True)

plt.show()
Sex_pct = pd.crosstab(

    dataset['Sex'].astype('category'),

    dataset['Survived'].astype('category'),

    margins=True,

#     normalize=True

)

Sex_pct
sns.barplot('Sex','Survived',data=dataset)

plt.show()
dataset.Sex.value_counts()
sns.countplot(dataset.Age.value_counts())

plt.show()
sns.pairplot(dataset)

plt.show()
sns.countplot(dataset.Sex.value_counts())

plt.show()
dataset.isnull().sum(axis=0)
dataset.dropna(inplace=True)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

dataset.Embarked = le.fit_transform(dataset.Embarked)

dataset.Sex = le.fit_transform(dataset.Sex)

dataset.head()
X,y = dataset[['Pclass','Sex','Age','Embarked']],dataset['Survived']

X.head()
y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

tree_clf = DecisionTreeClassifier(random_state=42)

tree_clf.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

y_pred = tree_clf.predict(X_test)

print(tree_clf.__class__.__name__, accuracy_score(y_test, y_pred))

print(f'Classification Report for {tree_clf.__class__.__name__}')

print(classification_report(y_test, y_pred))

print('*'*60)
bag_clf = BaggingClassifier(DecisionTreeClassifier(random_state=42), 

                            n_estimators=500,

                            bootstrap=True, n_jobs=-1)

bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)

print(bag_clf.__class__.__name__, accuracy_score(y_test, y_pred))

print(f'Classification Report for {bag_clf.__class__.__name__}')

print(classification_report(y_test, y_pred))

print('*'*60)
from sklearn.ensemble import RandomForestClassifier



rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)

rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)



accuracy_score(y_test, y_pred_rf)
bag_clf = BaggingClassifier(

    DecisionTreeClassifier(splitter='random', random_state=42),

    n_estimators=500, n_jobs=-1

)

bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)



accuracy_score(y_test, y_pred)
np.sum(y_pred == y_pred_rf) / len(y_pred)  # almost identical predictions
output = pd.DataFrame(X_test)

output['y_pred'] = y_pred
output.head()