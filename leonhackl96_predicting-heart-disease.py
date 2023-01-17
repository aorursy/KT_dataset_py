import pandas as pd

import os

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import style

from matplotlib.pyplot import plot



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



style.use("seaborn-whitegrid")

%matplotlib inline

%config InlineBackend.figure_format = "retina"
df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
data_size = df.memory_usage().sum() / 1024 / 1024

print("Data memory size: %.2f MB" % data_size)
df.head()
df.shape
df.info()
plt.figure(figsize = (14,2))

sns.countplot(data = df, y = "target", palette = "viridis")

plt.title("Target Variable")

plt.show()
plt.figure(figsize = (14,7))

sns.distplot(df.age)

plt.title("Age Distribution")

plt.show()
plt.figure(figsize = (14,2))

sns.countplot(data = df, y = "sex", hue = "target", palette = "viridis")

plt.title("Gender")

plt.show()
plt.figure(figsize = (14,4))

sns.countplot(data = df, y = "cp", hue = "target", palette = "viridis")

plt.title("Gender")

plt.show()
plt.figure(figsize = (14,7))

sns.distplot(df.trestbps)

plt.title("Resting Blood Pressure")

plt.show()
plt.figure(figsize = (14,7))

sns.distplot(df.chol)

plt.title("Serum Cholestoral")

plt.show()
plt.figure(figsize = (14,2))

sns.countplot(data = df, y = "fbs", hue = "target", palette = "viridis")

plt.title("Fasting Blood Sugar")

plt.show()
plt.figure(figsize = (14,3))

sns.countplot(data = df, y = "restecg", hue = "target", palette = "viridis")

plt.title("Resting Electrocardiographic")

plt.show()
plt.figure(figsize = (14,7))

sns.distplot(df.thalach)

plt.title("Maximum Heart Rate")

plt.show()
plt.figure(figsize = (14,2))

sns.countplot(data = df, y = "exang", hue = "target", palette = "viridis")

plt.title("Exercise Induced Angina")

plt.show()
corr = df.corr()

plt.figure(figsize = (12,10))

sns.heatmap(corr, cmap = "viridis", linewidth = 4, linecolor = "white")

plt.title("Correlation")

plt.show()
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
X = df.drop(["target"], axis = 1)

Y = df["target"]
scaler = StandardScaler()

X_pca = scaler.fit_transform(X)



pca = PCA(n_components = 2)

X_pca_transformed = pca.fit_transform(X_pca)



plt.figure(figsize = (14,7))



for i in Y.unique():

    X_pca_filtered = X_pca_transformed[Y == i, :]

    plt.scatter(X_pca_filtered[:, 0], X_pca_filtered[:, 1], s = 30, label = i, alpha = 0.8)

    

plt.legend()

plt.title("PCA", fontsize = 14)

plt.show()
from sklearn.linear_model import LogisticRegression 

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier, StackingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

from sklearn.pipeline import Pipeline
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
pipe_rf = Pipeline([

    ("scaler", StandardScaler()),

    ("rf", RandomForestClassifier(criterion = "gini", 

                                  max_features = "auto"))])



params_rf = {

    "rf__n_estimators" : [200, 250, 300, 400, 600],

    "rf__max_depth" : [1, 3, 5, 7, 9]}



grid = GridSearchCV(pipe_rf, params_rf, cv = StratifiedKFold(n_splits = 5), n_jobs = -1)

grid.fit(X_train, Y_train)



print(grid.best_params_)



grid = grid.best_estimator_

rf_score = cross_val_score(grid, X_test, Y_test, cv = 10, n_jobs = -1)



print("")

print("Cross-Validated-Score: " + str(round(rf_score.mean(), 6)))



train_size, train_score, test_score = learning_curve(grid, X_train, Y_train, cv = 10, n_jobs = -1)



print("")

print("Learning Curve:")



plt.figure(figsize = (12,6))

plt.plot(train_size, np.mean(train_score, axis = 1), label = "Train scores")

plt.plot(train_size, np.mean(test_score, axis = 1), label = "Test scores")

plt.title("Random Forest")

plt.legend()

plt.show()
pipe_knn = Pipeline([

    ("scaler", StandardScaler()),

    ("knn", KNeighborsClassifier(algorithm = "auto"))])



params_knn = {

    "knn__n_neighbors" : [2, 3, 5, 7, 9],

    "knn__leaf_size" : [10, 20, 30, 40]}



grid = GridSearchCV(pipe_knn, params_knn, cv = StratifiedKFold(n_splits = 5), n_jobs = -1)

grid.fit(X_train, Y_train)



print(grid.best_params_)



grid = grid.best_estimator_

knn_score = cross_val_score(grid, X_test, Y_test, cv = 10, n_jobs = -1)



print("")

print("Cross-Validated-Score: " + str(round(knn_score.mean(), 6)))



train_size, train_score, test_score = learning_curve(grid, X_train, Y_train, cv = 10, n_jobs = -1)



print("")

print("Learning Curve:")



plt.figure(figsize = (12,6))

plt.plot(train_size, np.mean(train_score, axis = 1), label = "Train scores")

plt.plot(train_size, np.mean(test_score, axis = 1), label = "Test scores")

plt.title("K Nearest Neighbor")

plt.legend()

plt.show()
pipe_log = Pipeline([

    ("scaler", StandardScaler()),

    ("log", LogisticRegression())])



params_log = {

    "log__C" : [0.001, 0.01, 0.1, 1, 1.1, 10],

    "log__max_iter" : [10000],

    "log__solver" : ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]}



grid = GridSearchCV(pipe_log, params_log, cv = StratifiedKFold(n_splits = 5), n_jobs = -1)

grid.fit(X_train, Y_train)



print(grid.best_params_)



grid = grid.best_estimator_

log_score = cross_val_score(grid, X_test, Y_test, cv = 10, n_jobs = -1)



print("")

print("Cross-Validated-Score: " + str(round(log_score.mean(), 6)))



train_size, train_score, test_score = learning_curve(grid, X_train, Y_train, cv = 10, n_jobs = -1)



print("")

print("Learning Curve:")



plt.figure(figsize = (12,6))

plt.plot(train_size, np.mean(train_score, axis = 1), label = "Train scores")

plt.plot(train_size, np.mean(test_score, axis = 1), label = "Test scores")

plt.title("Logistic Regression")

plt.legend()

plt.show()
all_results = [log_score, rf_score, knn_score]



result_names = [ "Logistic Regression", 

                "Random Forest", 

                "KNN"]



fig = plt.figure(figsize = (12,6))

fig.suptitle("Algorithm Comparison")

ax = fig.add_subplot(111)

plt.boxplot(all_results)

ax.set_xticklabels(result_names)

plt.show()