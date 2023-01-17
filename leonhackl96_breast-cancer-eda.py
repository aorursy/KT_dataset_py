import pandas as pd

import numpy as np

import os

import time



import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import style

from matplotlib.pyplot import plot



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



style.use("seaborn-whitegrid")

%matplotlib inline

%config InlineBackend.figure_format = 'retina'
df = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")
df.head()
df.shape
df.info()
df_ml = df.drop(["Unnamed: 32", "id"], axis = 1)
plt.figure(figsize = (14, 2))

sns.countplot(data = df_ml, y = "diagnosis", palette = "viridis")

plt.title("Diagnosis", fontsize = 14)

plt.show()
def multi_plot(feature_name, mean, worst, se):

    fig, ax = plt.subplots(1, 2, figsize = (16,6))

    sns.kdeplot(df_ml[mean], shade = True, ax = ax[0], label = "Mean")

    sns.kdeplot(df_ml[worst], shade = True, ax = ax[0], label = "Worst")

    sns.kdeplot(df_ml[se], shade = True, ax = ax[1], label = "Standard Error")

    ax[0].set_title("Mean/Worst " + feature_name, fontsize = 14)

    ax[1].set_title(feature_name + " standard error", fontsize = 14)

    plt.show()
multi_plot("Radius", "radius_mean", "radius_worst", "radius_se")
multi_plot("Texture", "texture_mean", "texture_worst", "texture_se")
multi_plot("Perimeter", "perimeter_mean", "perimeter_worst", "perimeter_se")
multi_plot("Area", "area_mean", "area_worst", "area_se")
multi_plot("Smoothness", "smoothness_mean", "smoothness_worst", "smoothness_se")
multi_plot("Compactness", "compactness_mean", "compactness_worst", "compactness_se")
multi_plot("Concavity", "concavity_mean", "concavity_worst", "concavity_se")
multi_plot("Concave points", "concave points_mean", "concave points_worst", "concave points_se")
multi_plot("Symmetry", "symmetry_mean", "symmetry_worst", "symmetry_se")
multi_plot("Fractal dimension", "fractal_dimension_mean", "fractal_dimension_worst", "fractal_dimension_se")
corr = df_ml.corr()

plt.figure(figsize = (10,8))

sns.heatmap(corr, cmap = "viridis", linewidth = 4, linecolor = "white")

plt.title("Correlation", fontsize = 14)

plt.show()



upper = corr.where(np.triu(np.ones(corr.shape), k = 1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

to_drop
for value in to_drop:

    df_ml = df_ml.drop([value], axis = 1)
from sklearn.linear_model import LogisticRegression 

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report, confusion_matrix

from catboost import CatBoostClassifier

import lightgbm as lgb
df_ml = df_ml.sample(frac = 1)
X = df_ml.drop("diagnosis", axis = 1)

Y = df_ml["diagnosis"]



X_train, X_test, Y_train, Y_test = train_test_split(X, 

                                                    Y, 

                                                    test_size = 0.25, 

                                                    random_state = 0)
scaler = StandardScaler()

X = scaler.fit_transform(X)
print(X_train.shape)

print(X_test.shape)
def model_builder(pipeline, params):

    grid = GridSearchCV(pipeline, params, cv = 5, n_jobs = -1)

    grid = grid.fit(X_train, Y_train)

    print(grid.best_params_)

    grid = grid.best_estimator_

    grid = grid.fit(X_train, Y_train)

    

    Y_predicted_proba = grid.predict_proba(X_test)[:, 1]

    Y_predicted = grid.predict(X_test)

    

    print("")

    print("-" * 60)

    print("Classification Report :")

    print("")

    print(classification_report(Y_test, Y_predicted))

    print("-" * 60)

    print("Confusion Matrix :")

    print("")

    print(confusion_matrix(Y_test, Y_predicted))

    print("-" * 60)

    

    score = grid.score(X_test, Y_test)

    print("Accuracy : " + str(round(score, 4)))

    

    train_size, train_score, test_score = learning_curve(grid, X_train, Y_train, cv = 5, n_jobs = -1)



    plt.figure(figsize = (8,5))

    plt.plot(train_size, np.mean(train_score, axis = 1), label = "Train scores")

    plt.plot(train_size, np.mean(test_score, axis = 1), label = "Test scores")

    plt.legend()

    plt.title("Learning Curve", fontsize = 14)

    plt.show()

        

    return score, Y_predicted_proba
pipe_log = Pipeline([

    ("log", LogisticRegression())])



params_log = {

    "log__C" : [0.001, 0.01, 0.1, 1, 1.1, 10, 15],

    "log__max_iter" : [10000],

    "log__solver" : ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]}



score_log, Y_pred_log = model_builder(pipe_log, params_log)
pipe_knn = Pipeline([

    ("knn", KNeighborsClassifier(algorithm = "auto"))])



params_knn = {

    "knn__n_neighbors" : [2, 3, 5, 7],

    "knn__leaf_size" : [3, 5, 10, 20, 30]}



score_knn, Y_pred_knn = model_builder(pipe_knn, params_knn)
pipe_rf = Pipeline([

    ("rf", RandomForestClassifier())])



params_rf = {

    "rf__n_estimators" : [100, 200, 250 , 300, 500],

    "rf__max_depth" : [3, 5, 7, 9, 11],

    "rf__min_samples_split" : [1, 2, 3],

    "rf__min_samples_leaf" : [1, 2, 3]}



score_rf, Y_pred_rf = model_builder(pipe_rf, params_rf)
pipe_svm = Pipeline([

    ("svm", SVC(kernel = "rbf", probability = True))])



params_svm = {

    "svm__C" : [0.0001, 0.001, 0.01, 0.1, 1, 1.1, 2, 3],

    "svm__gamma" : [0.0001, 0.001, 0.01, 0.1, 1, 1.1, 2, 3]}



score_svm, Y_pred_svm = model_builder(pipe_svm, params_svm)
pipe_gbm = Pipeline([

    ("scaler", StandardScaler()),

    ("lgb", lgb.LGBMClassifier())])



params_gbm = {

        "lgb__num_leaves": [3, 5, 7, 40, 60, 100],

        "lgb__n_estimators": [250, 300, 700, 1000],

        "lgb__learning_rate" : [0.0001, 0.001, 0.01, 0.1, 1]}



score_gbm, Y_pred_gbm = model_builder(pipe_gbm, params_gbm)
pipe_boost = Pipeline([

    ("boost", GradientBoostingClassifier())])



params_boost = {

    "boost__n_estimators" : [200, 300, 500, 700],

    "boost__learning_rate" : [0.005, 0.01, 0.1, 1, 1.1],

    "boost__max_depth" : [2, 3, 5],

    "boost__max_features" : [2, 3, 5]}



score_boost, Y_pred_boost = model_builder(pipe_boost, params_boost)