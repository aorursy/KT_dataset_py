import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import fbeta_score,confusion_matrix
import eli5
from eli5.sklearn import PermutationImportance
from pdpbox import pdp
print(os.listdir("../input"))
data = pd.read_csv("../input/data.csv", index_col="id")
data.head()
data.describe(include="all")
data.drop("Unnamed: 32", axis=1, inplace=True)
ax = sns.countplot(data.diagnosis)
ax.set_xticklabels(["Malignant", "Benign"])
ax.set_xlabel("Diagnosis")
ax.set_ylabel("Number of Data Samples")
X = data.drop("diagnosis", axis=1)
y = data.diagnosis
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=y)
X_train = X_train.copy()
y_train = y_train.copy()
X_test = X_test.copy()
y_test = y_test.copy()
scaler = StandardScaler()
X_train[X_train.columns] = scaler.fit_transform(X_train)
X_test[X_test.columns] = scaler.transform(X_test)
lr_param_grid = dict(class_weight=(None, "balanced"),
                     penalty=("l1", "l2"),
                     C=np.logspace(-3, 3, 7))
lr_cv = GridSearchCV(LogisticRegression(solver="liblinear", random_state=0),
                     lr_param_grid,
                     cv=5,
                     iid=False)
best_lr_params = lr_cv.fit(X_train, y_train).best_params_
lr_model = LogisticRegression(penalty=best_lr_params["penalty"],
                              C=best_lr_params["C"],
                              class_weight=best_lr_params["class_weight"],
                              solver="liblinear",
                              random_state=0)
lr_model.fit(X_train, y_train)
svm_param_grid = dict(class_weight=(None, "balanced"),
                      C=np.logspace(-3, 3, 7),
                      kernel=("linear", "poly", "rbf", "sigmoid"),
                      degree=(2, 3),
                      gamma=("auto", "scale"),
                      shrinking=(True, False))
svm_cv = GridSearchCV(SVC(random_state=0),
                      svm_param_grid,
                      cv=5,
                      iid=False)
best_svm_params = svm_cv.fit(X_train, y_train).best_params_
print(f"A {best_svm_params['kernel']} SVM kernel should be used for this task.")
lsvm_param_grid = dict(class_weight=(None, "balanced"),
                       penalty=("l1", "l2"),
                       loss=("hinge", "squared_hinge"),
                       dual=(True, False),
                       C=np.logspace(-3, 3, 7))
lsvm_cv = GridSearchCV(LinearSVC(random_state=0),
                       lsvm_param_grid,
                       cv=5,
                       iid=False,
                       error_score=np.nan)
best_lsvm_params = lsvm_cv.fit(X_train, y_train).best_params_
svm_model = LinearSVC(class_weight=best_lsvm_params["class_weight"],
                      penalty=best_lsvm_params["penalty"],
                      loss=best_lsvm_params["loss"],
                      dual=best_lsvm_params["dual"],
                      C=best_lsvm_params["C"],
                      random_state=0)
svm_model.fit(X_train, y_train)
print(f"The test accuracy score of the LR model is {lr_model.score(X_test, y_test)}.")
print("The test accuracy score of the linear SVM model is "
      f"{svm_model.score(X_test, y_test)}.")
lr_preds = lr_model.predict(X_test)
svm_preds = svm_model.predict(X_test)
print("The test F2-Score of the LR model is "
      f"{fbeta_score(y_test, lr_preds, beta=2, pos_label='M')}.")
print("The test F2-Score of the linear SVM model is "
      f"{fbeta_score(y_test, svm_preds, beta=2, pos_label='M')}.")
confusion = pd.DataFrame(confusion_matrix(y_test, lr_preds))
confusion = confusion.div(confusion.sum().sum())
confusion.columns = ["Predicted Negative", "Predicted Positive"]
confusion.index = ["Actual Negative", "Actual Positive"]
ax = sns.heatmap(confusion, vmin=0, vmax=1, annot=True, fmt=".0%")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.collections[0].colorbar.set_ticks((0, .25, .5, .75, 1))
ax.collections[0].colorbar.set_ticklabels(("0%", "25%", "50%", "75%", "100%"))
permutations = PermutationImportance(lr_model, random_state=0).fit(X_test, y_test)
eli5.show_weights(permutations, top=None, feature_names=X_test.columns.tolist())
feature_names=X_test.columns
top_features = ("texture_worst", "radius_se", "perimeter_se",
                "area_se", "compactness_se")
for i, feature in enumerate(top_features):
    pdp_feature = pdp.pdp_isolate(lr_model, X_test, feature_names, feature)
    pdp.pdp_plot(pdp_feature, feature)