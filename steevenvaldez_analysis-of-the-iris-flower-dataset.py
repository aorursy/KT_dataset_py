import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
dataset_iris=pd.read_csv("/kaggle/input/iris/Iris.csv")
# Ниже первые пять строк
dataset_iris.head()
# Названия столбцов
dataset_iris.columns
# Форма набора данных (строки, столбцы)
dataset_iris.shape
# Размер набора данных
dataset_iris.size
# Удаление столбца Id
dataset_iris = dataset_iris.drop('Id',axis=1)
# Количество образцов, доступных для каждого вида
dataset_iris["Species"].value_counts()
# Описательные статистические данные
# dataset.describe() включают те, которые суммируют центральную тенденцию, дисперсию и форму распределения набора данных.
print(dataset_iris.describe())
# Построение соответствующих гистограмм каждого цветка
sns.FacetGrid(dataset_iris,hue="Species",height=3).map(sns.distplot,"SepalLengthCm").add_legend()
sns.FacetGrid(dataset_iris,hue="Species",height=3).map(sns.distplot,"SepalWidthCm").add_legend()
sns.FacetGrid(dataset_iris,hue="Species",height=3).map(sns.distplot,"PetalLengthCm").add_legend()
sns.FacetGrid(dataset_iris,hue="Species",height=3).map(sns.distplot,"PetalWidthCm").add_legend()
plt.show()
# Построение Box Plot для набора данных
sns.boxplot(x="Species",y="PetalLengthCm",data=dataset_iris)
plt.show()
# Построение Violin Plot для набора данных
sns.violinplot(x="Species",y="PetalLengthCm",data=dataset_iris)
plt.show()
# Построение Pair Plot для набора данных
sns.pairplot(dataset_iris, hue="Species")
plt.show()
X = dataset_iris.iloc[:, 1:4].values
y = dataset_iris.iloc[:, 4].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) 
#test_size: if integer, number of examples into test dataset; if between 0.0 and 1.0, means proportion
print('В тренировочном наборе {} образцов, в тестовом наборе {} образцов'.format(X_train.shape[0], X_test.shape[0]))
#Масштабирование данных
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#X_train_std и X_test_std - это масштабированные наборы данных, которые будут использоваться моделями
#Применение SVC (поддержка векторной классификации)
from sklearn.svm import SVC

svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)
svm.fit(X_train_std, y_train)
print('Точность классификатора SVM по данным обучения составляет {:.2f}'.format(svm.score(X_train_std, y_train)))
print('Точность классификатора SVM по данным испытания составляет {:.2f}'.format(svm.score(X_test_std, y_test)))
#Применение k-NN (Метод k-ближайших соседей)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 7, p = 2, metric='minkowski')
knn.fit(X_train_std, y_train)

print('Точность классификатора k-NN по данным обучения составляет {:.2f}'.format(knn.score(X_train_std, y_train)))
print('Точность классификатора k-NN по данным испытания составляет {:.2f}'.format(knn.score(X_test_std, y_test)))
#Применение XGBoost
import xgboost as xgb

xgb_clf = xgb.XGBClassifier()
xgb_clf = xgb_clf.fit(X_train_std, y_train)

print('Точность классификатора XGBoost по данным обучения составляет {:.2f}'.format(xgb_clf.score(X_train_std, y_train)))
print('Точность классификатора XGBoost по данным испытания составляет {:.2f}'.format(xgb_clf.score(X_test_std, y_test)))
#Применение дерева решений
from sklearn import tree

decision_tree = tree.DecisionTreeClassifier(criterion='gini')
decision_tree.fit(X_train_std, y_train)

print('Точность классификатора дерева решений по данным обучения составляет {:.2f}'.format(decision_tree.score(X_train_std, y_train)))
print('Точность классификатора дерева решений по данным испытания составляет {:.2f}'.format(decision_tree.score(X_test_std, y_test)))
#Применение случайного леса
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier()
random_forest.fit(X_train_std, y_train)

print('Точность классификатора случайных лесов по данным обучения составляет {:.2f}'.format(random_forest.score(X_train_std, y_train)))
print('Точность классификатора случайных лесов по данным испытания составляет {:.2f}'.format(random_forest.score(X_test_std, y_test)))