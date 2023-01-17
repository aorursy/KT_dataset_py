import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.decomposition import PCA

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB
data = pd.read_csv("../input/iris.csv", 

                   names=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Species'])
data.head()
data.info()
features = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']



for feature in features:

    data[feature] = pd.to_numeric(data[feature], errors='coerce')
data[features].isnull().sum()
for feature in features:

    data[feature] = data[feature].fillna(data[feature].mean())



# check if there are still any missing values

data.isnull().sum()
data.describe()
sns.pairplot(data, hue="Species")
sns.countplot(data['Species'])
x_reduced = PCA(n_components=2).fit_transform(data[features])



plt.figure(figsize=(10,8))

sns.scatterplot(x_reduced[:,0], x_reduced[:,1], hue=data['Species'])

plt.xlabel('1st Principle Componenet', fontsize=12)

plt.ylabel('2nd Principle Componenet', fontsize=12)

plt.title('First two Principle Components', fontsize=14)

plt.show()
#Input

X = data[features]

#Encode output

le = LabelEncoder()

y = le.fit_transform(data['Species'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

print('Training dataset samples : %d' %(X_train.shape[0]))

print('Testing dataset samples  : %d' %(X_test.shape[0]))
neighbors = [1, 2, 3, 4, 5, 6, 7]



for n in neighbors:

    knn = KNeighborsClassifier(n_neighbors=n)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    print("Accuracy for n_neighbors=%d : %0.2f" %(n, accuracy_score(y_test, y_pred)))
nb = GaussianNB()

nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

print("Accuracy score for Naive Baye : %0.2f" %(accuracy_score(y_test, y_pred)))