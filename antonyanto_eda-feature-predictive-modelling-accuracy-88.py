import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, roc_curve, classification_report, confusion_matrix, roc_auc_score
# %matplotlib inline
# plt.rcdefaults()
# sns.set_style()
# plt.rc("figure", figsize=[9, 5])
# plt.style.use("seaborn")
# sns.set(rc={"figure.figsize": [9, 5]})
data = pd.read_csv("../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")
data.drop(labels=["sl_no"], axis=1, inplace=True)
data.head()
data.dtypes
data.info()
data.isna().sum().sort_values(ascending=False)
data.loc[data["salary"].isna(), :]    # You can also use this code :) -->  data[data["salary"].isna()]
data.describe()
fig = plt.figure()
ax = fig.add_subplot()

sns.countplot(x="status", data=data, ax=ax)
plt.title("Target Class Distribution")
plt.xlabel("Class Label")
plt.show()
fig = plt.figure()
ax = fig.add_subplot()

sns.countplot(x="gender", hue="status", data=data, ax=ax)
plt.xlabel("Gender")
plt.title("Gender vs Placement")
plt.show()
fig = plt.figure()
ax = fig.add_subplot()

sns.countplot(x="specialisation", hue="status", data=data, ax=ax)
plt.xlabel("Specialisation")
plt.title("Specialisation vs Placement")
plt.show()
fig = plt.figure()
ax = fig.add_subplot()

sns.countplot(x="workex", hue="status", data=data, ax=ax)
plt.xlabel("Work Experience")
plt.title("Work Experience vs Placement")
plt.show()
fig = plt.figure()
ax = fig.add_subplot()

sns.countplot(x="degree_t", hue="status", data=data, ax=ax)
plt.title("Degree Priority for Placement")
plt.xlabel("Degree")
plt.show()
fig = plt.figure()
ax = fig.add_subplot()

sns.countplot(x="ssc_b", hue="status", data=data, ax=ax)
plt.title("Board of Education vs Placement")
plt.xlabel("Board of Education")
plt.show()
fig = plt.figure()
ax = fig.add_subplot()

sns.countplot(x="hsc_s", hue="status", data=data, ax=ax)
plt.xlabel("HSC Groups")
plt.title("HSC Groups vs Placement")
plt.show()
fig = plt.figure()
ax = fig.add_subplot()

sns.barplot(x="status", y="etest_p", data=data, ax=ax, ci=None)
plt.title("Employability Test vs Placement")
plt.xlabel("Status")
plt.ylabel("Employability Test")
plt.show()
fig = plt.figure(figsize=[11, 6])
ax = fig.add_subplot()

sns.scatterplot(x="ssc_p", y="hsc_p", hue=data["status"].tolist(),
                style=data["ssc_b"].tolist(), size=data["hsc_s"].tolist(), data=data, ci=None, ax=ax)
plt.xlabel("Secondary School Percentage")
plt.ylabel("Higher Secondary School Percentage")
plt.show()
categorical_variables = data.select_dtypes(include="object").columns.tolist()

treat_not_as_same = ["degree_t", "hsc_s"]

treat_as_same = [var for var in categorical_variables if not var in treat_not_as_same]
for var in treat_as_same[:-1]:
    dict_to_map = {j:i for i, j in enumerate(data[var].unique())}
    data[var] = data[var].map(dict_to_map)

data["status"] = data["status"].map({"Not Placed": 0, "Placed": 1})
for var in treat_not_as_same:
    data = pd.concat(objs=[data, pd.get_dummies(data=data[var])], axis=1)
    data.drop(labels=var, axis=1, inplace=True)
    
data.drop(labels="salary", axis=1, inplace=True)
fig = plt.figure(figsize=[11, 6])
ax = fig.add_subplot()

sns.heatmap(data=data.corr(), cmap="RdYlGn", annot=True, fmt=".2f", ax=ax)
plt.title("Correlation Among all Variables")
plt.show()
X = data.drop(labels="status", axis=1).values
y = data["status"].values
models = [("Logistic Regression", LogisticRegression(random_state=0, n_jobs=-1)),
         ("Linear SVM", SVC(kernel="linear", random_state=0)),
         ("RBF SVM", SVC(kernel="rbf", random_state=0)),
         ("Decision Tree", DecisionTreeClassifier(random_state=0)),
         ("Random Forest", RandomForestClassifier(n_jobs=-1, random_state=0)),
         ("Adaboost RF", AdaBoostClassifier(base_estimator=RandomForestClassifier(n_jobs=-1, random_state=0), random_state=0, learning_rate=0.1)),
         ("Adaboost DT", AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=0), learning_rate=0.1)),
         ("Gradient Boosting", GradientBoostingClassifier(random_state=0))]
stratified = StratifiedKFold()
model_details = {name: [] for name, _ in models}

for train_index, test_index in stratified.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    for name, model in models:
        if name in ["Logistic Regression", "Linear SVM", "RBF SVM"]:
            std = StandardScaler()
            X_train = std.fit_transform(X_train)
            X_test = std.transform(X_test)

        model.fit(X_train, y_train)
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        model_details[name].append((train_accuracy, test_accuracy))
summary_df = pd.DataFrame(index=["Train Score", "Test Score"])

for model, accuracy in zip(model_details.keys(), model_details.values()):
    train_accuracy = [train_accuracy for train_accuracy, _ in accuracy]
    test_accuracy = [test_accuracy for _, test_accuracy in accuracy]
    summary_df[model] = [np.mean(train_accuracy), np.mean(test_accuracy)]
summary_df.T
stratified_split = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

for train_index, test_index in stratified_split.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
std = StandardScaler()
scaled_train = std.fit_transform(X_train)
scaled_test = std.transform(X_test)
logistic = LogisticRegression(random_state=0, n_jobs=-1)
logistic.fit(scaled_train, y_train)
print("Logistic Regression Test Score :", logistic.score(scaled_test, y_test))
con_mat = pd.DataFrame(confusion_matrix(y_test, logistic.predict(scaled_test)))
fig = plt.figure(figsize=[6, 4])
ax = fig.add_subplot()

sns.heatmap(con_mat, annot=True, fmt=".2f", cmap="RdYlGn", cbar=False, ax=ax)
plt.title("Logistic Regression Confusion Matrix")
plt.show()
fpr, tpr, thresshold = roc_curve(y_test, logistic.predict_proba(scaled_test)[:, 1])
auc_score = auc(fpr, tpr)

fig = plt.figure(figsize=[9, 5])
ax = fig.add_subplot()

plt.plot(fpr, tpr, c="darkred", lw=2, label="AUC = {}".format(round(auc_score, 2)))
plt.plot([0, 1], [0, 1], c='black', lw=2, ls='--', label="AUC = {}".format(0.5))
plt.title("ROC Curve - Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc=4)
plt.show()
C = [100, 10, 1.0, 0.1, 0.01]
penalty = ['l2', 'l1']
param = {"C": C, "penalty": penalty}

grid = GridSearchCV(estimator=LogisticRegression(random_state=0, n_jobs=-1), param_grid=param, n_jobs=-1)
grid.fit(scaled_train, y_train)

grid_model = grid.estimator.fit(scaled_train, y_train)
test_score = grid_model.score(scaled_test, y_test)
print("Tuned Logistic Regression Test Score :", test_score)
print(classification_report(y_test, grid_model.predict(scaled_test)))
svc = SVC(kernel="rbf", random_state=0, probability=True)
svc.fit(scaled_train, y_train)
print("SVM Test Score :", svc.score(scaled_test, y_test))
con_mat = pd.DataFrame(confusion_matrix(y_test, svc.predict(scaled_test)))
fig = plt.figure(figsize=[6, 4])
ax = fig.add_subplot()

sns.heatmap(con_mat, annot=True, fmt=".2f", cmap="RdYlGn", cbar=False, ax=ax)
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
fpr, tpr, thresshold = roc_curve(y_test, svc.predict_proba(scaled_test)[:, 1])
auc_score = auc(fpr, tpr)

fig = plt.figure(figsize=[9, 5])
ax = fig.add_subplot()

plt.plot(fpr, tpr, c="darkred", lw=2, label="AUC = {}".format(round(auc_score, 2)))
plt.plot([0, 1], [0, 1], c='black', lw=2, ls='--', label="AUC = {}".format(0.5))
plt.title("ROC Curve - SVM")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc=4)
plt.show()
C = [100, 10, 1.0, 0.1, 0.01]
kernel = ["linear", "RBF", "poly"]
param = {"C": C, "kernel":kernel}

grid = GridSearchCV(estimator=SVC(random_state=0), param_grid=param, n_jobs=-1)
grid.fit(scaled_train, y_train)

grid_model = grid.estimator.fit(scaled_train, y_train)
test_score = grid_model.score(scaled_test, y_test)
print("Tuned SVM Test Score :", test_score)
print(classification_report(y_test, grid_model.predict(scaled_test)))
dec = DecisionTreeClassifier(random_state=0)
dec.fit(X_train, y_train)
print("Decision Tree Classifier Test Score :", dec.score(X_test, y_test))
con_mat = pd.DataFrame(confusion_matrix(y_test, dec.predict(X_test)))
fig = plt.figure(figsize=[6, 4])
ax = fig.add_subplot()

sns.heatmap(con_mat, annot=True, fmt=".2f", cmap="RdYlGn", cbar=False, ax=ax)
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
fpr, tpr, thresshold = roc_curve(y_test, dec.predict_proba(X_test)[:, 1])
auc_score = auc(fpr, tpr)

fig = plt.figure(figsize=[9, 5])
ax = fig.add_subplot()

plt.plot(fpr, tpr, c="darkred", lw=2, label="AUC = {}".format(round(auc_score, 2)))
plt.plot([0, 1], [0, 1], c='black', lw=2, ls='--', label="AUC = {}".format(0.5))
plt.title("ROC Curve - Decision Tree")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc=4)
plt.show()
depth = list(range(1, 11))
min_sample_split = np.arange(5, 30, 5)
min_leaf_sample = np.arange(3, 16, 3)
features = ["auto", "sqrt", "log2"]
max_leaf_nodes = [4, 6, 8, 10]
param = {"max_depth": depth, "min_samples_split": min_sample_split, "min_samples_leaf": min_leaf_sample,
         "max_features": features, "max_leaf_nodes": max_leaf_nodes}

grid = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0), param_grid=param, n_jobs=-1)
grid.fit(X_train, y_train)

grid_model = grid.estimator.fit(X_train, y_train)
test_score = grid_model.score(X_test, y_test)
print("Tuned Decision Tree Test Score :", test_score)
print(classification_report(y_test, grid_model.predict(X_test)))
ran = RandomForestClassifier(random_state=0)
ran.fit(X_train, y_train)
print("Random Forest Classifier Test Score :", ran.score(X_test, y_test))
con_mat = pd.DataFrame(confusion_matrix(y_test, ran.predict(X_test)))
fig = plt.figure(figsize=[6, 4])
ax = fig.add_subplot()

sns.heatmap(con_mat, annot=True, fmt=".2f", cmap="RdYlGn", cbar=False, ax=ax)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
fpr, tpr, thresshold = roc_curve(y_test, ran.predict_proba(X_test)[:, 1])
auc_score = auc(fpr, tpr)

fig = plt.figure(figsize=[9, 5])
ax = fig.add_subplot()

plt.plot(fpr, tpr, c="darkred", lw=2, label="AUC = {}".format(round(auc_score, 2)))
plt.plot([0, 1], [0, 1], c='black', lw=2, ls='--', label="AUC = {}".format(0.5))
plt.title("ROC Curve - Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc=4)
plt.show()
n_estimators = np.arange(100, 140, 10)
max_depth = np.arange(3, 10, 2)
max_features = ["auto", "sqrt"]
max_leaf_nodes = np.arange(3, 10, 2)
param = {"n_estimators": n_estimators, "max_depth": max_depth, "max_features": max_features, "max_leaf_nodes": max_leaf_nodes}

grid = GridSearchCV(estimator=RandomForestClassifier(random_state=0, n_jobs=-1), param_grid=param, n_jobs=-1)
grid.fit(X_train, y_train)

grid_model = grid.estimator.fit(X_train, y_train)
test_score = grid_model.score(X_test, y_test)
print("Tuned Random Forest Test Score :", test_score)
print("AUC afetr parameter tuning : ", roc_auc_score(y_test, grid_model.predict_proba(X_test)[:, 1]))

#confusion matrix
con_mat = pd.DataFrame(confusion_matrix(y_test, grid_model.predict(X_test)))
fig = plt.figure(figsize=[6, 4])
ax = fig.add_subplot()

sns.heatmap(con_mat, annot=True, fmt=".2f", cmap="RdYlGn", cbar=False, ax=ax)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
print(classification_report(y_test, grid_model.predict(X_test)))
ada = AdaBoostClassifier(base_estimator=RandomForestClassifier(random_state=0, n_jobs=-1), learning_rate=0.1, random_state=0)
ada.fit(X_train, y_train)
print("AdaBoost Classifier Test Score :", ada.score(X_test, y_test))
con_mat = pd.DataFrame(confusion_matrix(y_test, ada.predict(X_test)))
fig = plt.figure(figsize=[6, 4])
ax = fig.add_subplot()

sns.heatmap(con_mat, annot=True, fmt=".2f", cmap="RdYlGn", cbar=False, ax=ax)
plt.title("AdaBoost Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
fpr, tpr, thresshold = roc_curve(y_test, ada.predict_proba(X_test)[:, 1])
auc_score = auc(fpr, tpr)

fig = plt.figure(figsize=[9, 5])
ax = fig.add_subplot()

plt.plot(fpr, tpr, c="darkred", lw=2, label="AUC = {}".format(round(auc_score, 2)))
plt.plot([0, 1], [0, 1], c='black', lw=2, ls='--', label="AUC = {}".format(0.5))
plt.title("ROC Curve - AdaBoost")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc=4)
plt.show()
print(classification_report(y_test, ada.predict(X_test)))
grad = GradientBoostingClassifier(learning_rate=0.1, random_state=0)
grad.fit(X_train, y_train)
print("GradientBoost Classifier Test Score :", grad.score(X_test, y_test))
con_mat = pd.DataFrame(confusion_matrix(y_test, grad.predict(X_test)))
fig = plt.figure(figsize=[6, 4])
ax = fig.add_subplot()

sns.heatmap(con_mat, annot=True, fmt=".2f", cmap="RdYlGn", cbar=False, ax=ax)
plt.title("GardientBoost Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
print(classification_report(y_test, grad.predict(X_test)))
estimators = [("LR", LogisticRegression(random_state=0, n_jobs=-1)),
             ("SVC", SVC(random_state=0)),
             ("RF", RandomForestClassifier(random_state=0, n_jobs=-1)),
             ("Ada", AdaBoostClassifier(RandomForestClassifier(random_state=0, n_jobs=-1), learning_rate=0.1, random_state=0)),
             ("Dec", DecisionTreeClassifier(random_state=0))]
vot = VotingClassifier(estimators, n_jobs=-1)
vot.fit(X_train, y_train)
print("Votting Classifier Test Score :", vot.score(X_test, y_test))
con_mat = pd.DataFrame(confusion_matrix(y_test, vot.predict(X_test)))
fig = plt.figure(figsize=[6, 4])
ax = fig.add_subplot()

sns.heatmap(con_mat, annot=True, fmt=".2f", cmap="RdYlGn", cbar=False, ax=ax)
plt.title("Votting Classifier Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
print(classification_report(y_test, vot.predict(X_test)))