import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns # Statistical data visualization
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer
# keys - Column dictionary

cancer.keys()
print(cancer["DESCR"])
print(cancer["target"])
print(cancer["target_names"])
print(cancer["feature_names"])
cancer["data"].shape # Shape of the data. 560 rows & 30 columns (features)
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))
df_cancer.head() # dataframe
df_cancer.tail()
sns.pairplot(df_cancer, vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'], hue="target" )
sns.countplot(df_cancer["target"])
sns.scatterplot(x="mean area", y="mean smoothness", data=df_cancer, hue="target")
plt.figure(figsize=(20, 10))

sns.heatmap(df_cancer.corr(), annot=True)
X = df_cancer.iloc[:, :-1]
X.head()
y = df_cancer.iloc[:, -1]
y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
X_train.head()
y_train.head()
# It may be a case where your training dataset contains mostly case for target 1 or 2 or any other class (if there)

# In this case it will be a biased sample and our model will be underfit

sns.countplot(y_train )# so you can we have good number of samples for both of the classes.
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
svc = SVC()
svc.fit(X_train, y_train)
y_predict = svc.predict(X_test)
y_predict
cm = confusion_matrix(y_test, y_predict)
cm
sns.heatmap(cm, annot=True)
min_train = X_train.min()

range_train = (X_train - min_train).max()

X_train_scaled = (X_train - min_train)/range_train
X_train_scaled.head()
# Before scaling

sns.scatterplot(x=X_train["mean area"], y=X_train["mean smoothness"],  hue=y_train)
# After scaling

sns.scatterplot(x=X_train_scaled["mean area"], y=X_train_scaled["mean smoothness"],  hue=y_train)
X_train_scaled.head()
# Scale test

min_test = X_test.min()

range_test = (X_train - min_test).max()

X_test_scaled = (X_test - min_test)/range_train
svc.fit(X_train_scaled, y_train)
y_predict = svc.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot=True)
print(classification_report(y_test, y_predict))
from sklearn.model_selection import GridSearchCV
param_grid = {

    "C" : [0.1, 1, 4, 10, 50, 100],

    "gamma" : [0.1, 1, 5, 10, 100],

    "kernel" : ["rbf"]

}
grid = GridSearchCV(svc, param_grid = param_grid, cv=10, verbose=5, refit=True)
grid.fit(X_train_scaled, y_train)
print("best params", grid.best_params_)

print("best score",grid.best_score_)
# SO wonderfull with given params we got 98% accuracy

# lets plot confusion matrix.
grid_predict = grid.predict(X_test_scaled)
cm= confusion_matrix(y_test, grid_predict)
sns.heatmap(cm, annot=True)
print(classification_report(y_test, grid_predict))
# Well.. we got an accuracy of 97% & recall of 92% which is very good.

# Thank you.