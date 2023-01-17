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



from sklearn.metrics import confusion_matrix
train_data = pd.read_csv("../input/titanic/train.csv")

train_data = train_data.drop(columns=['Name', 'PassengerId', 'Fare', 'Ticket', 'Cabin'])

train_data.head(10)
# Graph some correlations

sns.barplot(x="Pclass", y="Survived", data=pd.DataFrame(train_data[['Pclass', 'Survived']]))

plt.xlabel("Socio-Economic Class")

plt.title("Socio-economic class vs Survived")

plt.show()

sns.barplot(x="Sex", y="Survived", data=pd.DataFrame(train_data[['Sex', 'Survived']]))

plt.title("Sex vs Survived")

plt.show()
# Check for na values

display(train_data.isnull().any())

isNull = train_data['Age'].isnull()

print(isNull.sum())



# Drop rows with na values

train_data = train_data.dropna()
# Fill na values with mean of column

train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())

train_data.isna().any()
x_train = train_data.drop(columns=['Survived'])

y_train = train_data['Survived'].to_numpy()

x_train['Sex'] = pd.get_dummies(train_data['Sex'])

x_train['Embarked'] = pd.get_dummies(train_data['Embarked'])



display(x_train)

print(y_train)
# Train Random Forest On Data

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)

model.fit(x_train, y_train)

print("Training accuracy: ", model.score(x_train, y_train))

y_pred_train = model.predict(x_train)

cm = confusion_matrix(y_train, y_pred_train, normalize='true')

print(cm)
# Kth Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=11, weights='distance')

model.fit(x_train, y_train)

print("Training accuracy: ", model.score(x_train, y_train))

y_pred_train = model.predict(x_train)

cm = confusion_matrix(y_train, y_pred_train, normalize='true')

print(cm)
# SVMs

from sklearn import svm

model = svm.SVC()

model.fit(x_train, y_train)

print("Training accuracy: ", model.score(x_train, y_train))

y_pred_train = model.predict(x_train)

cm = confusion_matrix(y_train, y_pred_train, normalize='true')

print(cm)
# Naive Bayes

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(x_train, y_train)

print("Training accuracy: ", model.score(x_train, y_train))

y_pred_train = model.predict(x_train)

cm = confusion_matrix(y_train, y_pred_train, normalize='true')

print(cm)
from sklearn.cluster import KMeans

#K-means elbow

inertias = []

for k in range(1,11):

  kmeans = KMeans(n_clusters = k, init='k-means++')

  kmeans.fit(x_train, y_train)

  inertias.append(kmeans.inertia_)

  print(kmeans.inertia_)





plt.figure(figsize=(10, 6))

plt.xticks(np.arange(1, 11, 1))

plt.plot([k for k in range(1,11)], inertias)



y_pred_train = model.predict(x_train)

cm = confusion_matrix(y_train, y_pred_train, normalize='true')

print(cm)
# SVMs

from sklearn import svm

model = svm.SVC()

model.fit(x_train, y_train)

print("Training accuracy: ", model.score(x_train, y_train))

y_pred_train = model.predict(x_train)

cm = confusion_matrix(y_train, y_pred_train, normalize='true')

print(cm)







y_pred_train = model.predict(x_train)

cm = confusion_matrix(y_train, y_pred_train, normalize='true')

print(cm)

sns.heatmap(data=cm, annot=True)

plt.xlabel('Predicted')

plt.ylabel('True')

plt.title('Normalized Confusion Matrix')
test_data = pd.read_csv("../input/titanic/test.csv")

test_data = test_data.drop(columns=['Name', 'Fare', 'Ticket', 'Cabin'])
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())

test_data.isna().any()
test_data['Sex'] = pd.get_dummies(test_data['Sex'])

test_data['Embarked'] = pd.get_dummies(test_data['Embarked'])



y_pred_test = model.predict(test_data.drop(columns=['PassengerId']))

final_result = pd.DataFrame(test_data['PassengerId'])

final_result['Survived'] = y_pred_test

final_result
final_result.to_csv("titanic_submission.csv", index=False)