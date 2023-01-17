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
iris_data = pd.read_csv("/kaggle/input/iris-flower-dataset/IRIS.csv")
iris_data.describe()
iris_data.tail(5)
import seaborn as sns
sns.distplot(a=iris_data['petal_length'], kde=False)
sns.kdeplot(data=iris_data['petal_length'], shade=True)
sns.jointplot(x=iris_data['petal_length'], y=iris_data['sepal_width'], kind="kde")
sns.barplot(x=iris_data['petal_length'], y=iris_data['sepal_width'])
sns.scatterplot(x=iris_data['sepal_length'], y=iris_data['sepal_width'], hue=iris_data['species'])

sns.scatterplot(x=iris_data['petal_length'], y=iris_data['petal_width'], hue=iris_data['species'])
sns.lmplot(x="sepal_length", y="sepal_width", hue="species", data=iris_data)
sns.lmplot(x="petal_length", y="petal_width", hue="species", data=iris_data)
sns.swarmplot(x=iris_data['species'],

              y=iris_data['petal_width'])
sns.swarmplot(x=iris_data['species'],

              y=iris_data['petal_length'])
sns.swarmplot(x=iris_data['species'],

              y=iris_data['sepal_width'])
sns.swarmplot(x=iris_data['species'],

              y=iris_data['sepal_length'])
missing_values_count = iris_data.isnull().sum()

missing_values_count[0:]
iris_data['species'].replace('Iris-setosa', 0,inplace=True)

iris_data['species'].replace('Iris-versicolor', 1,inplace=True)

iris_data['species'].replace('Iris-virginica', 2,inplace=True)

iris_data.head(5)
atrib_prev = ['species']
atributos =['sepal_length','petal_width','sepal_width','petal_length']
Y = iris_data[atrib_prev].values

X = iris_data[atributos].values
X
import sklearn as sk

sk.__version__
from sklearn.model_selection import train_test_split
split_test_size = 0.30
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size = split_test_size)
print("{0:0.2f}% nos dados de treino".format((len(X_treino)/len(df.index)) * 100))

print("{0:0.2f}% nos dados de teste".format((len(X_teste)/len(df.index)) * 100))
from sklearn.naive_bayes import GaussianNB
modelo_v1 = GaussianNB()
modelo_v1.fit(X_treino, Y_treino.ravel())
from sklearn import metrics
nb_predict_train = modelo_v1.predict(X_treino)
nb_predict_train = modelo_v1.predict(X_treino)

print("Exatidão (Accuracy): {0:.4f}".format(metrics.accuracy_score(Y_treino, nb_predict_train)))

print()
nb_predict_test = modelo_v1.predict(X_teste)
nb_predict_test = modelo_v1.predict(X_teste)

print("Exatidão (Accuracy): {0:.4f}".format(metrics.accuracy_score(Y_teste, nb_predict_test)))

print()
print("Confusion Matrix")



print("{0}".format(metrics.confusion_matrix(Y_teste, nb_predict_test, labels = [1, 0])))

print("")



print("Classification Report")

print(metrics.classification_report(Y_teste, nb_predict_test, labels = [1, 0]))
from sklearn.ensemble import RandomForestClassifier
modelo_v2 = RandomForestClassifier(random_state = 42)

modelo_v2.fit(X_treino, Y_treino.ravel())
rf_predict_train = modelo_v2.predict(X_treino)

print("Exatidão (Accuracy): {0:.4f}".format(metrics.accuracy_score(Y_treino, rf_predict_train)))
rf_predict_test = modelo_v2.predict(X_teste)

print("Exatidão (Accuracy): {0:.4f}".format(metrics.accuracy_score(Y_teste, rf_predict_test)))

print()
print("Confusion Matrix")



print("{0}".format(metrics.confusion_matrix(Y_teste, rf_predict_test, labels = [1, 0])))

print("")



print("Classification Report")

print(metrics.classification_report(Y_teste, rf_predict_test, labels = [1, 0]))
from sklearn.linear_model import LogisticRegression
modelo_v3 = LogisticRegression(C = 0.7, random_state = 42, max_iter = 1000)

modelo_v3.fit(X_treino, Y_treino.ravel())

lr_predict_test = modelo_v3.predict(X_teste)
print("Exatidão (Accuracy): {0:.4f}".format(metrics.accuracy_score(Y_teste, lr_predict_test)))

print()

print("Classification Report")

print(metrics.classification_report(Y_teste, lr_predict_test, labels = [1, 0]))
rf_predict_test = modelo_v3.predict(X_teste)

print("Exatidão (Accuracy): {0:.4f}".format(metrics.accuracy_score(Y_teste, rf_predict_test)))

print()
from sklearn import tree

modelo_v4 = tree.DecisionTreeClassifier()

modelo_v4.fit(X_treino, Y_treino.ravel())

lr_predict_test = modelo_v4.predict(X_teste)
rf_predict_test = modelo_v4.predict(X_teste)

print("Exatidão (Accuracy): {0:.4f}".format(metrics.accuracy_score(Y_teste, rf_predict_test)))

print()
print("Exatidão (Accuracy): {0:.4f}".format(metrics.accuracy_score(Y_teste, lr_predict_test)))

print()

print("Classification Report")

print(metrics.classification_report(Y_teste, lr_predict_test, labels = [1, 0]))
tree.plot_tree(modelo_v4)
atrib_prev = ['species']
atributos =['petal_width','petal_length']
split_test_size = 0.30
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size = split_test_size)
modelo_v1 = GaussianNB()
modelo_v1.fit(X_treino, Y_treino.ravel())
nb_predict_train = modelo_v1.predict(X_treino)

print("Exatidão (Accuracy): {0:.4f}".format(metrics.accuracy_score(Y_treino, nb_predict_train)))

print()
nb_predict_test = modelo_v1.predict(X_teste)

print("Exatidão (Accuracy): {0:.4f}".format(metrics.accuracy_score(Y_teste, nb_predict_test)))

print()
modelo_v2 = RandomForestClassifier(random_state = 42)

modelo_v2.fit(X_treino, Y_treino.ravel())
rf_predict_test = modelo_v2.predict(X_teste)

print("Exatidão (Accuracy): {0:.4f}".format(metrics.accuracy_score(Y_teste, rf_predict_test)))

print()
modelo_v3 = LogisticRegression(C = 0.7, random_state = 42, max_iter = 1000)

modelo_v3.fit(X_treino, Y_treino.ravel())

lr_predict_test = modelo_v3.predict(X_teste)
rf_predict_test = modelo_v3.predict(X_teste)

print("Exatidão (Accuracy): {0:.4f}".format(metrics.accuracy_score(Y_teste, rf_predict_test)))

print()
#Best model to use is the gaussianNB with all of the columns but species as the X parameters, the accuaracy was of 100%. 