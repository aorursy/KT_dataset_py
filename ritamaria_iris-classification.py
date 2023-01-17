from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
plt.style.use("bmh")
iris_data = load_iris()
iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris_df["Class"] = iris_data.target
iris_df.head()
iris_df.shape
iris_data.target_names
# check missing data
iris_df.isnull().sum()
sns.pairplot(iris_df, hue="Class", corner=True)
iris_df.dtypes
iris_df.describe()
# Split data
iris_data_X = iris_data.data
iris_data_X
iris_data_y = iris_data.target
iris_data_y
Xtrain, Xtest, ytrain, ytest = train_test_split(iris_data_X, iris_data_y, test_size=0.3, random_state=2)
lrparam = {"fit_intercept": [True, False], "normalize": [True, False], "copy_X": [True, False]}
lrgrid = GridSearchCV(LinearRegression(), lrparam, cv=10)
lrgrid.fit(Xtrain, ytrain)
print("Best Linear Regression score:", lrgrid.best_score_)
print("Best Linear Regression estimator:", lrgrid.best_estimator_)
logrparam = {"penalty": ["l1", "l2"], "solver": ["liblinear"], "C": np.linspace(0.00002, 1, 100)}
logrrand = RandomizedSearchCV(LogisticRegression(max_iter=1000), logrparam, cv=5, n_iter=15, scoring="accuracy")
logrrand.fit(Xtrain, ytrain)
print("Best Logistic Regression score:", logrrand.best_score_)
print("Best Logistic Regression estimator:", logrrand.best_estimator_)
svcparam = {"C": [0.1, 1, 10, 100, 1000], "kernel": ["rbf", "linear", "poly", "sigmoid"]}
svcgrid = GridSearchCV(SVC(), svcparam, cv=5, scoring="accuracy")
svcgrid.fit(Xtrain, ytrain)
print("Best SVC score:", svcgrid.best_score_)
print("Best SVC estimator:", svcgrid.best_estimator_)
gnbparam = {"var_smoothing": np.logspace(0, -9, num=100)}
gnbgrid = GridSearchCV(GaussianNB(), gnbparam, cv=10, scoring="accuracy")
gnbgrid.fit(Xtrain, ytrain)
print("Best GaussianNB score:", gnbgrid.best_score_)
print("Best GaussianNB estimator:", gnbgrid.best_estimator_)
dtparam = {"max_depth": [3, None], "max_features": randint(1, 4), "criterion": ["gini", "entropy"]}
dtrand = RandomizedSearchCV(DecisionTreeClassifier(), dtparam, cv=5, n_iter=15, scoring="accuracy")
dtrand.fit(Xtrain, ytrain)
print("Best DecisionTreeClassifier score:", dtrand.best_score_)
print("Best DecisionTreeClassifier estimator:", dtrand.best_estimator_)
rfparam = {"n_estimators": [int(x) for x in np.linspace(200, 2000, 10)], "max_features": ["log2", "sqrt", "auto"],
           "criterion": ["entropy", "gini"], "max_depth": [int(x) for x in np.linspace(10, 110, 11)],
           "min_samples_split": [2, 3, 5, 10], "min_samples_leaf": [1, 2, 4, 5, 8], "bootstrap": [True, False]}
rfrand = RandomizedSearchCV(RandomForestClassifier(), rfparam, cv=5, n_iter=5, scoring="accuracy")
rfrand.fit(Xtrain, ytrain)
print("Best RandomForestClassifier score:", rfrand.best_score_)
print("Best RandomForestClassifier estimator:", rfrand.best_estimator_)
prediction = svcgrid.best_estimator_.predict(Xtest)
mat = confusion_matrix(ytest, prediction)
sns.heatmap(mat, square=True, annot=True)
plt.xlabel("Predicted value")
plt.ylabel("True value")