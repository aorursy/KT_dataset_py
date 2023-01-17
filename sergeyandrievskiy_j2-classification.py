import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pylab as plt

from pylab import rcParams

rcParams['figure.figsize'] = 20, 10
df = pd.read_csv(r'../input/j-data-classification-for-task/j_data_classification_for_task.csv')
df.head(10)
# делим датасет на тренировочный и тестовый наборы данных



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(df.loc[:, 'AGE':'FEATURE_11'], df['TARGET'], random_state=0)
# Строим модель на основе метода k ближайших соседей



from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)



print("Accuracy on the train set: {:f}".format(knn.score(X_train, y_train)))

print("Accuracy on the test set: {:f}".format(knn.score(X_test, y_test)))



# Видим, что точность при n_neighbors = 3 недостаточно высокая, поэтому попробуем другую модель
# Строим модель на основе метода логистической регрессии



from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC



lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)



print("Accuracy on the train set: {:f}".format(lr.score(X_train, y_train)))

print("Accuracy on the test set: {:f}".format(lr.score(X_test, y_test)))



# Видим, что при использовании логистической регрессии точность гораздо лучше, чем при методе k ближайших соседей
# Строим модель на основе дерева решений



from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier(max_depth=5).fit(X_train, y_train)



print("Accuracy on the train set: {:f}".format(tree.score(X_train, y_train)))

print("Accuracy on the test set: {:f}".format(tree.score(X_test, y_test)))



# Здесь видно, что точность на тренировочном наборе данных выше,чем при логистической регрессии, что хорошо, однако при этом упала точность на тестовом наборе данных
# Сортируем и выводим важности признаков для исключения незначимых



from sklearn.ensemble import RandomForestClassifier



forest = RandomForestClassifier(n_estimators=250,

                              random_state=0)



forest.fit(X_train, y_train)

importances = forest.feature_importances_



features = []



for i in zip(df.columns[0:13], importances):

    features.append([i[0], i[1]])



for i in sorted(features, key=lambda x: x[1], reverse=True):

    if i[0] == 'AGE' or i[0] == 'GENDER':

        print(i[0], i[1], sep='\t\t')

    else:

        print(i[0], i[1], sep='\t')
# Видим, что у FEATURE_4 самая низкая значимость среди остальных признаков, поэтому попробуем исключить его и заново обучить модели    
X_train_wo4, X_test_wo4, y_train_wo4, y_test_wo4 = train_test_split(df.drop(['FEATURE_4'], axis=1).loc[:, 'AGE':'FEATURE_11'], df['TARGET'], random_state=0)
knn_wo4 = KNeighborsClassifier(n_neighbors=3).fit(X_train_wo4, y_train_wo4)



print("(knn)Accuracy on the train set: {:f}".format(knn.score(X_train, y_train)))

print("(knn)Accuracy on the test set: {:f}".format(knn.score(X_test, y_test)))

print("(knn_wo4)Accuracy on the train set: {:f}".format(knn_wo4.score(X_train_wo4, y_train_wo4)))

print("(knn_wo4)Accuracy on the test set: {:f}".format(knn_wo4.score(X_test_wo4, y_test_wo4)), end='\n\n')



lr_wo4 = LogisticRegression(max_iter=1000).fit(X_train_wo4, y_train_wo4)



print("(lr)Accuracy on the train set: {:f}".format(lr.score(X_train, y_train)))

print("(lr)Accuracy on the test set: {:f}".format(lr.score(X_test, y_test)))

print("(lr_wo4)Accuracy on the train set: {:f}".format(lr_wo4.score(X_train_wo4, y_train_wo4)))

print("(lr_wo4)Accuracy on the test set: {:f}".format(lr_wo4.score(X_test_wo4, y_test_wo4)), end='\n\n')



tree_wo4 = DecisionTreeClassifier(max_depth=5).fit(X_train_wo4, y_train_wo4)



print("(tree)Accuracy on the train set: {:f}".format(tree.score(X_train, y_train)))

print("(tree)Accuracy on the test set: {:f}".format(tree.score(X_test, y_test)))

print("(tree_wo4)Accuracy on the train set: {:f}".format(tree_wo4.score(X_train_wo4, y_train_wo4)))

print("(tree_wo4)Accuracy on the test set: {:f}".format(tree_wo4.score(X_test_wo4, y_test_wo4)))



# При сравнении точностей видно, что при методах k соседей и логистической регрессии после исключения FEATURE_4 точность не изменилась. Можно предположить,

# что этот признак коррелирует с другим признаком, поэтому не влияет на точность. Однако при этом снизилась точность на тренировочном наборе при методе дерева решений.