from sklearn import datasets

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
# Loading iris dataset and transforming it on a Data Frame

iris_ds = datasets.load_iris()

iris = pd.DataFrame(data=np.c_[iris_ds['data'], iris_ds['target']], columns= iris_ds['feature_names'] + ['target'])

# Printing the classes of flowers

print(iris_ds['target_names'])

# Printing the first 5 rows

iris.head()
fig, axs = plt.subplots(4, 1, figsize=(15, 10))

plt.subplots_adjust(hspace=0.5)

fig.suptitle('Iris characteristics', fontsize=20)

sns.lineplot(x='sepal length (cm)', y='target', data=iris, ax=axs[0])

sns.lineplot(x='sepal width (cm)', y='target', data=iris, ax=axs[1])

sns.lineplot(x='petal length (cm)', y='target', data=iris, ax=axs[2])

sns.lineplot(x='petal width (cm)', y='target', data=iris, ax=axs[3])

plt.show()
features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

X = iris[features]

y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=10)



model = DecisionTreeClassifier()

model.fit(X_train, y_train)

model
y_predicted = model.predict(X_test)



predicted_data = X_test.copy()

predicted_data['target'] = y_predicted



fig, axs = plt.subplots(4, 1, figsize=(15, 10))

plt.subplots_adjust(hspace=0.5)

fig.suptitle('Iris characteristics (predictions)', fontsize=20)

sns.lineplot(x='sepal length (cm)', y='target', data=predicted_data, ax=axs[0])

sns.lineplot(x='sepal width (cm)', y='target', data=predicted_data, ax=axs[1])

sns.lineplot(x='petal length (cm)', y='target', data=predicted_data, ax=axs[2])

sns.lineplot(x='petal width (cm)', y='target', data=predicted_data, ax=axs[3])

plt.show()
acc = accuracy_score(y_test, y_predicted)

print('Accuracy:', acc, 'that is, %.2f'%(acc*100),'%')



dmetric = classification_report(y_test, y_predicted)

print('Evaluation report\n', dmetric)