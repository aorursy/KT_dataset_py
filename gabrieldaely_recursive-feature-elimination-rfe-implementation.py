import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate

from sklearn.feature_selection import RFECV

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, plot_confusion_matrix

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline

random_state = 123
df = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv", index_col="customerID")

df.head()
df["TotalCharges"] = df["TotalCharges"].apply(pd.to_numeric, errors='coerce')

df["SeniorCitizen"] = df["SeniorCitizen"].apply(lambda x: "Yes" if x == 0 else "No")
df.dtypes
df.isnull().sum()
df = df.dropna()
df.isnull().sum()
for column in df.select_dtypes("number").columns:

    df.pivot(columns="Churn")[column].plot.hist(alpha=0.5)

    plt.title(column)

    plt.show()
for column in df.select_dtypes("object").columns.drop("Churn"):

    df.pivot(columns="Churn")[column].apply(pd.value_counts).plot.bar()

    plt.title(column)

    plt.show()
X = df.drop(columns=["Churn"])

y = df["Churn"]
scaler = StandardScaler()

X[X.select_dtypes("number").columns] = scaler.fit_transform(X.select_dtypes("number"))
ordEnc = OrdinalEncoder(dtype=np.int)

X[X.select_dtypes("object").columns] = ordEnc.fit_transform(X.select_dtypes("object"))
labEnc = LabelEncoder()

y = labEnc.fit_transform(y)
estimator = LogisticRegression(random_state=random_state)

rfecv = RFECV(estimator=estimator, cv=StratifiedKFold(10, random_state=random_state, shuffle=True), scoring="accuracy")

rfecv.fit(X, y)
plt.figure(figsize=(8, 6))

plt.plot(range(1, len(rfecv.grid_scores_)+1), rfecv.grid_scores_)

plt.grid()

plt.xticks(range(1, X.shape[1]+1))

plt.xlabel("Number of Selected Features")

plt.ylabel("CV Score")

plt.title("Recursive Feature Elimination (RFE)")

plt.show()



print("The optimal number of features: {}".format(rfecv.n_features_))
X_rfe = X.iloc[:, rfecv.support_]
print("\"X\" dimension: {}".format(X.shape))

print("\"X\" column list:", X.columns.tolist())

print("\"X_rfe\" dimension: {}".format(X_rfe.shape))

print("\"X_rfe\" column list:", X_rfe.columns.tolist())
X_train, X_test, X_rfe_train, X_rfe_test, y_train, y_test = train_test_split(X, X_rfe, y, 

                                                                             train_size=0.8, 

                                                                             stratify=y,

                                                                             random_state=random_state)

print("Train size: {}".format(len(y_train)))

print("Test size: {}".format(len(y_test)))
clf_keys = ["Logistic Regression", "Support Vector Machine", "Naive Bayes", "k-Nearest Neighbors",

            "Stochastic Gradient Descent", "Decision Tree", "AdaBoost", "Multi-layer Perceptron"]

clf_values = [LogisticRegression(random_state=random_state), SVC(kernel="linear", random_state=random_state),

              GaussianNB(), KNeighborsClassifier(), SGDClassifier(random_state=random_state),

              DecisionTreeClassifier(random_state=random_state), AdaBoostClassifier(random_state=random_state), 

              MLPClassifier(random_state=random_state, max_iter=1000)]

clf_rfe_keys = ["Logistic Regression", "Support Vector Machine", "Naive Bayes", "k-Nearest Neighbors",

                "Stochastic Gradient Descent", "Decision Tree", "AdaBoost", "Multi-layer Perceptron"]

clf_rfe_values = [LogisticRegression(random_state=random_state), SVC(kernel="linear",random_state=random_state),

                  GaussianNB(), KNeighborsClassifier(), SGDClassifier(random_state=random_state),

                  DecisionTreeClassifier(random_state=random_state), AdaBoostClassifier(random_state=random_state), 

                  MLPClassifier(random_state=random_state, max_iter=1000)]

clfs = dict(zip(clf_keys, clf_values))

clfs_rfe = dict(zip(clf_rfe_keys, clf_rfe_values))



# Original dataset

print("Model training using original data: started!")

for clf_name, clf in clfs.items():

    clf.fit(X_train, y_train)

    clfs[clf_name] = clf

    print(clf_name, "training: done!")

print("Model training using original data: done!\n")



# Feature-selected dataset

print("Model training using feature-selected data: started!")

for clf_rfe_name, clf_rfe in clfs_rfe.items():

    clf_rfe.fit(X_rfe_train, y_train)

    clfs_rfe[clf_rfe_name] = clf_rfe

    print(clf_rfe_name, "training: done!")

print("Model training using feature-selected data: done!")
# Original dataset

acc = []

for clf_name, clf in clfs.items():

    y_pred = clf.predict(X_test)

    acc.append(accuracy_score(y_test, y_pred))



# Feature selected dataset

acc_rfe = []

for clf_rfe_name, clf_rfe in clfs_rfe.items():

    y_rfe_pred = clf_rfe.predict(X_rfe_test)

    acc_rfe.append(accuracy_score(y_test, y_rfe_pred))

    

acc_all = pd.DataFrame({"Original dataset": acc, "Feature-selected dataset": acc_rfe},

                       index=clf_keys)

acc_all
print("Accuracy\n" + acc_all.mean().to_string())



ax = acc_all.plot.bar(figsize=(10, 8))

for p in ax.patches:

    ax.annotate(str(p.get_height().round(3)), (p.get_x()*0.985, p.get_height()*1.002))

plt.ylim((0.7, 0.82))

plt.xticks(rotation=90)

plt.title("All Classifier Accuracies")

plt.grid()

plt.show()
scoring = ["accuracy", "roc_auc"]



scores = []

# Original dataset

print("Cross-validation on original data: started!")

for clf_name, clf in clfs.items():

    score = pd.DataFrame(cross_validate(clf, X, y, cv=StratifiedKFold(10, random_state=random_state, shuffle=True), scoring=scoring)).mean()

    scores.append(score)

    print(clf_name, "cross-validation: done!")

cv_scores = pd.concat(scores, axis=1).rename(columns=dict(zip(range(len(clf_keys)), clf_keys)))

print("Cross-validation on original data: done!\n")



scores = []

# Feature-selected dataset

print("Cross-validation on feature-selected data: started!")

for clf_name, clf in clfs_rfe.items():

    score = pd.DataFrame(cross_validate(clf, X_rfe, y, cv=StratifiedKFold(10, random_state=random_state, shuffle=True), scoring=scoring)).mean()

    scores.append(score)

    print(clf_name, "cross-validation: done!")

cv_scores_rfe = pd.concat(scores, axis=1).rename(columns=dict(zip(range(len(clf_keys)), clf_keys)))

print("Cross-validation on feature-selected data: done!")
# Accuracy

cv_acc_all = pd.concat([cv_scores.loc["test_accuracy"].rename("Original data"), cv_scores_rfe.loc["test_accuracy"].rename("Feature-selected data")], 

                       axis=1)



print("Cross-validation accuracy\n" + cv_acc_all.mean().to_string())

ax = cv_acc_all.plot.bar(figsize=(10, 8))

for p in ax.patches:

    ax.annotate(str(p.get_height().round(3)), (p.get_x()*0.985, p.get_height()*1.003))

plt.xticks(rotation=90)

plt.ylim((0.7, 0.82))

plt.title("Cross-validation Accuracy")

plt.grid()

plt.legend()

plt.show()
# ROC AUC

cv_roc_auc_all = pd.concat([cv_scores.loc["test_roc_auc"].rename("Original data"), cv_scores_rfe.loc["test_roc_auc"].rename("Feature-selected data")], 

                           axis=1)



print("Cross-validation ROC AUC score\n" + cv_roc_auc_all.mean().to_string())

ax = cv_roc_auc_all.plot.bar(figsize=(10, 8))

for p in ax.patches:

    ax.annotate(str(p.get_height().round(3)), (p.get_x()*0.985, p.get_height()*1.003))

plt.xticks(rotation=90)

plt.ylim((0.63, 0.88))

plt.title("Cross-validation ROC AUC Score")

plt.grid()

plt.legend()

plt.show()
# Fit time

cv_fit_time_all = pd.concat([cv_scores.loc["fit_time"].rename("Original data"), cv_scores_rfe.loc["fit_time"].rename("Feature-selected data")], 

                           axis=1)



print("Cross-validation fit time\n" + cv_fit_time_all.mean().to_string())

ax = cv_fit_time_all.plot.bar(figsize=(10, 8))

for p in ax.patches:

    ax.annotate(str(p.get_height().round(3)), (p.get_x()*0.985, p.get_height()*1.003))

plt.xticks(rotation=90)

plt.yscale("log")

plt.title("Cross-validation Fit Time")

plt.grid()

plt.legend()

plt.show()
importance = abs(clfs["Logistic Regression"].coef_[0])

plt.barh(X.columns.values[importance.argsort()], importance[importance.argsort()])

plt.title("Logistic Regression - Feature Importance (Original Data)")

plt.grid()

plt.show()



importance_rfe = abs(clfs_rfe["Logistic Regression"].coef_[0])

plt.barh(X_rfe.columns.values[importance_rfe.argsort()], importance_rfe[importance_rfe.argsort()])

plt.title("Logistic Regression - Feature Importance (Feature-selected Data)")

plt.grid()

plt.show()
importance = clfs["AdaBoost"].feature_importances_

plt.barh(X.columns.values[importance.argsort()], importance[importance.argsort()])

plt.title("AdaBoost - Feature Importance (Original Data)")

plt.grid()

plt.show()



importance_rfe = clfs_rfe["AdaBoost"].feature_importances_

plt.barh(X_rfe.columns.values[importance_rfe.argsort()], importance_rfe[importance_rfe.argsort()])

plt.title("AdaBoost - Feature Importance (Feature-selected Data)")

plt.grid()

plt.show()
importance = abs(clfs["Support Vector Machine"].coef_[0])

plt.barh(X.columns.values[importance.argsort()], importance[importance.argsort()])

plt.title("Support Vectore Machine - Feature Importance (Original Data)")

plt.grid()

plt.show()



importance_rfe = abs(clfs_rfe["Support Vector Machine"].coef_[0])

plt.barh(X_rfe.columns.values[importance_rfe.argsort()], importance_rfe[importance_rfe.argsort()])

plt.title("Support Vectore Machine - Feature Importance (Feature-selected Data)")

plt.grid()

plt.show()