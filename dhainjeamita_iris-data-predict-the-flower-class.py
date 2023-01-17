import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

plt.style.use('seaborn-colorblind')

col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
dataset = pd.read_csv('../input/iris-dataset/iris.csv', header= None, names=col_names)
iris_class = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
dataset['species_num'] = [iris_class[i] for i in dataset.species]
X = dataset.drop(['species', 'species_num'], axis=1)
dataset.head()
dataset[['sepal_length','sepal_width','petal_length', 'petal_width']].plot.hist(alpha=0.7);
dataset[['sepal_length','sepal_width','petal_length', 'petal_width']].plot();
dataset[['sepal_length','sepal_width','petal_length', 'petal_width']].plot.kde();
dataset[['sepal_length','sepal_width','petal_length', 'petal_width']].plot.box();
sns.pairplot(dataset[['sepal_length','sepal_width','petal_length', 'petal_width','species']], hue='species', diag_kind='kde', size=2);
## Create an 'X' matrix by dropping the irrelevant columns.
X = dataset.drop(['species', 'species_num'], axis=1)
y = dataset.species_num
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
## Import the Classifier.

## Instantiate the model with 5 neighbors. 
knn = KNeighborsClassifier(n_neighbors=5)
## Fit the model on the training data.
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print ("The accuracy of the model - ", format(accuracy_score(y_test, pred) * 100))
# synthetic dataset for simple regression
from sklearn.datasets import make_regression
plt.figure()
plt.title('Sample regression problem with one input variable')
X_R1, y_R1 = make_regression(n_samples = 100, n_features=1,
                            n_informative=1, bias = 150.0,
                            noise = 30, random_state=0)
plt.scatter(X_R1, y_R1, marker= 'o', s=50)
plt.show()
from sklearn.neighbors import KNeighborsRegressor

X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1, random_state = 0)
knnreg = KNeighborsRegressor(n_neighbors = 5).fit(X_train, y_train)
train_score = knnreg.score(X_train, y_train)
test_score = knnreg.score(X_test, y_test)
plt.plot(X_train, y_train, 'o', alpha=0.9, label='Train')
plt.plot(X_test, y_test, '^', alpha=0.9, label='Test')
plt.xlabel('Input feature')
plt.ylabel('Target value')
plt.title('KNN Regression (K={})\n Train $R^2 = {:.3f}$,  Test $R^2 = {:.3f}$'.format(5, train_score, test_score))
plt.legend()
fig = plt.figure(figsize=(18, 18))

plt.show()
#print(knnreg.predict(X_test))
#print('R-squared test score: {:.3f}'.format(knnreg.score(X_test, y_test)))
