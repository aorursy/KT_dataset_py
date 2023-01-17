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
names = ['sepal.length', 'sepal.width','petal.length','petal.width','variaty']
data_iris = pd.read_csv(os.path.join(dirname, filename), names=names)
data_iris.head()
data_iris.describe()
target_0 = np.zeros(50); target_1 = np.zeros(50)+1; target_2 = np.zeros(50)+2
target = np.hstack((target_0,target_1,target_2))
target
sns.set_style("whitegrid")
plt.figure(figsize=(10, 8))
plt.scatter(data_iris['sepal.length'], data_iris['sepal.width'], c=target, alpha=0.5, s=data_iris['petal.length']*100)
plt.xlabel('sepal.length')
plt.ylabel('sepal.width')
plt.show()
import seaborn as sns
sns.set()
plt.hist(data_iris['sepal.length'])
plt.xlabel('sepal.length (cm)')
plt.ylabel('count')
plt.title('Histogram (sepal.length/count)')
plt.show()
sns.set_style("whitegrid")
sns.pairplot(data_iris, hue='variaty')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_iris, target, test_size=0.2)

print("X_train shape :", X_train.shape)
print("X_test shape :", X_test.shape)
from sklearn.neighbors import KNeighborsClassifier
plt.figure(figsize=(12,8))
plt.subplot(121)
plt.scatter(X_train['sepal.length'], X_train['sepal.width'], c=y_train, alpha=0.8)
plt.title("Train_set")
plt.subplot(122)
plt.scatter(X_test['sepal.length'], X_test['sepal.width'], c=y_test, alpha=0.8)
plt.title("Test_set")
plt.show()
model = KNeighborsClassifier()

X_train = X_train[['sepal.length' ,'sepal.width' ,'petal.length' ,'petal.width']]
X_test = X_test[['sepal.length' ,'sepal.width' ,'petal.length' ,'petal.width']]

model.fit(X_train, y_train)

print("Score train :", model.score(X_train, y_train))
print("Score Test :", model.score(X_test, y_test))
from sklearn.model_selection import validation_curve
k = np.arange(1,50)

train_score, val_score = validation_curve(model, X_train, y_train, 'n_neighbors',k, cv=5)
plt.figure()
plt.plot(k, train_score.mean(axis=1), label="train_score")
plt.plot(k, val_score.mean(axis=1), label="val_score")
plt.legend()
plt.title("validation curve")
plt.show()
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': np.arange(1, 50), 'metric': ['euclidean', 'manhattan']}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)
grid.best_score_
grid.best_params_
#save model
model = grid.best_estimator_
model.score(X_test, y_test)
from sklearn.model_selection import learning_curve
N, train_score, val_score = learning_curve(model, X_train, y_train, train_sizes=np.linspace(0.1,1.0,10), cv=5)

plt.figure()
plt.plot(N, train_score.mean(axis=1), label='train_score')
plt.plot(N, val_score.mean(axis=1), label='validation_score')
plt.legend()
plt.title("Validation curve")
plt.xlabel('Samples')
plt.ylabel('Score')
plt.show()
def predict_flowers(model, sepal_length, sepal_width, petal_length, petal_width):
    X_predict = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1,4)
    predict_value = model.predict(X_predict)
    print("predict :", predict_value)
    print("predict proba :", model.predict_proba(X_predict))
predict_flowers(model, 5.1,3.5,1.4,0.2)
predict_flowers(model, 5.0, 2.0, 3.5,1.0)
predict_flowers(model, 6.9, 3.2 ,5.7 ,2.3)