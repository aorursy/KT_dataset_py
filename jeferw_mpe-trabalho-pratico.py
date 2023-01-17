import datetime

datetime.datetime.now()
from sklearn import datasets

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn import tree

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn import metrics
iris = datasets.load_iris()
x = iris.data

y = iris.target
x_df = pd.DataFrame(x, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

y_df = pd.DataFrame(y)
x_df.shape
x_df.boxplot()
x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.10, random_state=42)
import seaborn as sns
sns.scatterplot(data=x_df)
from sklearn.model_selection import cross_val_score

clf = SVC(kernel='linear', C=1)

scores = cross_val_score(clf, iris.data, iris.target, cv=5)

print("Acuracia SVM Cross-validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
svm_cf = SVC(kernel='linear')

svm_cf.fit(x_train, y_train)

svm_predic = svm_cf.predict(x_test)

print(f'MSE SVM: {mean_squared_error(y_test, svm_predic)}')

print(f'MAE SVM: {mean_absolute_error(y_test, svm_predic)}')

print(f'R2 SVM: {r2_score(y_test, svm_predic)}')

print(f'Precisão SVM: {metrics.accuracy_score(y_test, svm_predic)}')
tree_cf = tree.DecisionTreeClassifier()

tree_cf.fit(x_train, y_train)

tree_predic = tree_cf.predict(x_test)

print(f'MSE Tree: {mean_squared_error(y_test, tree_predic)}')

print(f'MAE Tree: {mean_absolute_error(y_test, tree_predic)}')

print(f'R2 Tree: {r2_score(y_test, tree_predic)}')

print(f'Precisão Tree: {metrics.accuracy_score(y_test, tree_predic)}')
lr_cf = LogisticRegression()

lr_cf.fit(x_train, y_train)

lr_predic = lr_cf.predict(x_test)

print(f'MSE Tree: {mean_squared_error(y_test, lr_predic)}')

print(f'MAE Tree: {mean_absolute_error(y_test, lr_predic)}')

print(f'R2 Tree: {r2_score(y_test, lr_predic)}')

print(f'Precisão Tree: {metrics.accuracy_score(y_test, lr_predic)}')