import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
print('Вхідні параметри -  fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol')
print('Вихідні параметри - quality')
data.head()
if True in pd.isnull(data):
    print('Null values in data, alert!!!')
else: print('Неповних зразків даних немає')
data.describe()
print('Всі залежності між даними можемо бачити на графіках')
print('Так як маємо всього 6 оцiнок, то маємо справу iз задачею класифiкацiї')
sns.pairplot(data=data, hue='quality')
data = data[data['total sulfur dioxide']<data['total sulfur dioxide'].quantile(0.99)]
data = data[data['citric acid']<data['citric acid'].quantile(0.99)]
data = data[data['alcohol']<data['alcohol'].quantile(0.99)]
sns.pairplot(data=data, hue='quality')
Y = data['quality']
X = data.drop('quality', axis=1)
del(data)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=228)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
mlp1 = MLPClassifier(random_state=1, hidden_layer_sizes = 200, max_iter=500, activation='relu').fit(X_train, Y_train)
print('Кiлькiсть шарiв: ', mlp1.n_layers_)
print('Точнiсть: ', mlp1.score(X_test, Y_test))
print('Вся iнформацiя по классифiкацi: \n', classification_report(Y_test, mlp1.predict(X_test)))
mlp2 = MLPClassifier(random_state=1, hidden_layer_sizes = 200, max_iter=500, activation='logistic').fit(X_train, Y_train)
print('Кiлькiсть шарiв: ', mlp2.n_layers_)
print('Точнiсть: ', mlp2.score(X_test, Y_test))
print('Вся iнформацiя по классифiкацi: \n', classification_report(Y_test, mlp2.predict(X_test)))
mlp3 = MLPClassifier(random_state=1, hidden_layer_sizes = (1000, 1000), max_iter=500, activation='logistic').fit(X_train, Y_train)
print('Кiлькiсть шарiв: ', mlp3.n_layers_)
print('Точнiсть: ', mlp3.score(X_test, Y_test))
print('Вся iнформацiя по классифiкацi: \n', classification_report(Y_test, mlp3.predict(X_test)))
mlp4 = MLPClassifier(random_state=1, hidden_layer_sizes = (500, 400, 200, 100, 50), max_iter=500, activation='logistic').fit(X_train, Y_train)
print('Кiлькiсть шарiв: ', mlp4.n_layers_)
print('Точнiсть: ', mlp4.score(X_test, Y_test))
print('Вся iнформацiя по классифiкацi: \n', classification_report(Y_test, mlp4.predict(X_test)))
mlp5 = MLPClassifier(random_state=1, hidden_layer_sizes = (500, 400, 300, 200, 100, 50, 20), max_iter=500, activation='relu').fit(X_train, Y_train)
print('Кiлькiсть шарiв: ', mlp5.n_layers_)
print('Точнiсть: ', mlp5.score(X_test, Y_test))
print('Вся iнформацiя по классифiкацi: \n', classification_report(Y_test, mlp5.predict(X_test)))

