import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))
df = pd.read_csv("../input/heart.csv")

df.head()
df.columns
# CHEST PAIN TYPE



cp = df["cp"]

plt.hist(cp)

plt.title("Distribution of Chest Pain Type")

plt.xlabel("Pain Type")

plt.ylabel("Frequency")

plt.show()
cp1 = df[df["target"] == 1]["cp"]

cp0 = df[df["target"] == 0]["cp"]



bar_width = 0.25



plt.hist(cp1 - bar_width, color = "red", rwidth=1, label="Disease")

plt.hist(cp0, color="blue", rwidth=1, label="No Disease")

plt.title("Distribution of Chest Pain Type")

plt.xlabel("Pain Type")

plt.ylabel("Frequency")

plt.xticks([])

plt.legend()

plt.show()
cp_probs = {}

for i in set(df["cp"]):

    total = df[df["cp"] == i]

    cp_probs[i] = sum(total["target"] == 1) / len(total), sum(total["target"] == 0) / len(total)

keys = np.array(list(cp_probs.keys()))

values= np.array([np.array(w) for w in cp_probs.values()])

width = 0.25

plt.bar(keys, values[:, 0], width, label="Disease", color="red")

plt.bar(keys, values[:, 1], width, bottom=values[:, 0], label="No Disease", color="blue")

plt.title("Probability Distribution of Disease with Chest Pain Type")

plt.xlabel("Pain Type")

plt.ylabel("Probability")

plt.xticks(keys, [1, 2, 3, 4])

plt.legend()

plt.show()


from sklearn.model_selection import train_test_split



#Getting all the feature columns

features = list(df.columns)

features.remove("target")



X_train, X_test, y_train, y_test = train_test_split(df["cp"], df["target"])
from sklearn.dummy import DummyClassifier

from sklearn.metrics import accuracy_score, recall_score, precision_score, auc, roc_auc_score, roc_curve



d_clf = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

predicted = d_clf.predict(X_test)

print("Accuracy: {}".format(accuracy_score(y_test, predicted)))

print("Recall: {}".format(recall_score(y_test, predicted)))

print("Precision: {}".format(precision_score(y_test, predicted)))

# print("AUC: {}".format(auc(y_test, predicted)))

d_fpr, d_tpr, d_thresholds = roc_curve(y_test, predicted)

plt.plot(d_fpr, d_tpr)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.show()

print("AUC Score: {}".format(roc_auc_score(y_test, predicted)))
from sklearn.linear_model import LogisticRegression



clf = LogisticRegression().fit(np.array(X_train).reshape(-1, 1), y_train)

predicted = clf.predict(np.array(X_test).reshape(-1, 1))

print("Accuracy: {}".format(accuracy_score(y_test, predicted)))

print("Recall: {}".format(recall_score(y_test, predicted)))

print("Precision: {}".format(precision_score(y_test, predicted)))

# print("AUC: {}".format(auc(y_test, predicted)))

fpr, tpr, d_thresholds = roc_curve(y_test, predicted)

plt.plot(fpr, tpr, label="Logistic Classifier")

plt.plot(d_fpr, d_tpr, color="green", linestyle="--", label="Dummy Classifier")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

print("AUC Score: {}".format(roc_auc_score(y_test, predicted)))
ag1 = df[df["target"] == 1]["age"]

ag0 = df[df["target"] == 0]["age"]



bar_width = 0.25



plt.hist(ag1 - bar_width, color = "red", bins=25, rwidth=1, label="Disease")

plt.hist(ag0, color="blue", rwidth=1, bins=50, label="No Disease")

plt.title("Distribution of Age")

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.legend()

plt.show()
ag_probs = {}

for i in set(df["age"]):

    total = df[df["age"] == i]

    ag_probs[i] = sum(total["target"] == 1) / len(total), sum(total["target"] == 0) / len(total)

keys = np.array(list(ag_probs.keys()))

values= np.array([np.array(w) for w in ag_probs.values()])

# width = 0.25

plt.bar(keys, values[:, 0], label="Disease", color="red")

plt.bar(keys, values[:, 1], bottom=values[:, 0], label="No Disease", color="blue")

plt.title("Probability Distribution of Disease with Age")

plt.xlabel("Pain Type")

plt.ylabel("Probability")

plt.legend()

plt.plot()
X_train, X_test, y_train, y_test = train_test_split(df["age"], df["target"])



clf = LogisticRegression().fit(np.array(X_train).reshape(-1, 1), y_train)

predicted = clf.predict(np.array(X_test).reshape(-1, 1))

print("Accuracy: {}".format(accuracy_score(y_test, predicted)))

print("Recall: {}".format(recall_score(y_test, predicted)))

print("Precision: {}".format(precision_score(y_test, predicted)))

# print("AUC: {}".format(auc(y_test, predicted)))

fpr, tpr, d_thresholds = roc_curve(y_test, predicted)

plt.plot(fpr, tpr, label="Logistic Classifier")

plt.plot(d_fpr, d_tpr, color="green", linestyle="--", label="Dummy Classifier")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

print("AUC Score: {}".format(roc_auc_score(y_test, predicted)))
ch1 = df[df["target"] == 1]["chol"]

ch0 = df[df["target"] == 0]["chol"]



bar_width = 0.25



plt.hist(ch1, color = "red", bins=25, rwidth=1, label="Disease")

plt.hist(ch0, color="blue", rwidth=1, bins=50, label="No Disease")

plt.title("Distribution of Cholesterol")

plt.xlabel("Cholesterol")

plt.ylabel("Frequency")

plt.legend()

plt.show()
X_train, X_test, y_train, y_test = train_test_split(df["chol"], df["target"])



clf = LogisticRegression().fit(np.array(X_train).reshape(-1, 1), y_train)

predicted = clf.predict(np.array(X_test).reshape(-1, 1))

print("Accuracy: {}".format(accuracy_score(y_test, predicted)))

print("Recall: {}".format(recall_score(y_test, predicted)))

print("Precision: {}".format(precision_score(y_test, predicted)))

# print("AUC: {}".format(auc(y_test, predicted)))

fpr, tpr, d_thresholds = roc_curve(y_test, predicted)

plt.plot(fpr, tpr, label="Logistic Classifier")

plt.plot(d_fpr, d_tpr, color="green", linestyle="--", label="Dummy Classifier")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

print("AUC Score: {}".format(roc_auc_score(y_test, predicted)))
bp1 = df[df["target"] == 1]["trestbps"]

bp0 = df[df["target"] == 0]["trestbps"]



bar_width = 0.25



plt.hist(bp1, color = "red", bins=25, rwidth=1, label="Disease")

plt.hist(bp0, color="blue", rwidth=1, bins=50, label="No Disease")

plt.title("Distribution of Blood Pressure with Disease")

plt.xlabel("Blood Pressure")

plt.ylabel("Frequency")

plt.legend()

plt.show()
X_train, X_test, y_train, y_test = train_test_split(df["trestbps"], df["target"])



clf = LogisticRegression().fit(np.array(X_train).reshape(-1, 1), y_train)

predicted = clf.predict(np.array(X_test).reshape(-1, 1))

print("Accuracy: {}".format(accuracy_score(y_test, predicted)))

print("Recall: {}".format(recall_score(y_test, predicted)))

print("Precision: {}".format(precision_score(y_test, predicted)))

# print("AUC: {}".format(auc(y_test, predicted)))

fpr, tpr, d_thresholds = roc_curve(y_test, predicted)

plt.plot(fpr, tpr, label="Logistic Classifier")

plt.plot(d_fpr, d_tpr, color="green", linestyle="--", label="Dummy Classifier")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

print("AUC Score: {}".format(roc_auc_score(y_test, predicted)))
name_dict = {"restecg":"resting electrocardiographic results (values 0,1,2)", "thalach":"maximum heart rate achieved","exang":"exercise induced angina", "oldpeak" :"ST depression induced by exercise relative to rest", "slope":"the slope of the peak exercise ST segment"}



for feat in ['restecg', 'thalach', 'exang', 'oldpeak', 'slope']:

    feat1 = df[df["target"] == 1][feat]

    feat0 = df[df["target"] == 0][feat]



    bar_width = 0.25

    if feat in ['restecg', 'exang', 'slope']:

        plt.hist(feat1 - bar_width, color = "red", rwidth=1, label="Disease")

        plt.hist(feat0, color="blue", rwidth=1, label="No Disease")

    else:

        plt.hist(feat1 - bar_width, color = "red", bins=50, rwidth=1, label="Disease")

        plt.hist(feat0, color="blue", rwidth=1, bins=50, label="No Disease")        

    plt.title("Distribution of {} with Disease".format(name_dict[feat]))

    plt.xlabel(name_dict[feat])

    plt.ylabel("Frequency")

    plt.legend()

    plt.show()
for feat in ['restecg', 'exang', 'slope']:

    feat_probs = {}

    for i in set(df[feat]):

        total = df[df[feat] == i]

        feat_probs[i] = sum(total["target"] == 1) / len(total), sum(total["target"] == 0) / len(total)

    keys = np.array(list(feat_probs.keys()))

    values= np.array([np.array(w) for w in feat_probs.values()])

    # width = 0.25

    plt.bar(keys, values[:, 0], label="Disease", color="red")

    plt.bar(keys, values[:, 1], bottom=values[:, 0], label="No Disease", color="blue")

    plt.title("Probability Distribution of Disease with {}".format(name_dict[feat]))

    plt.xlabel(name_dict[feat])

    plt.ylabel("Probability")

    plt.legend()

    plt.show()
df.dtypes
df.head(2)
df = df.astype({"age": "int64", "sex": "int64", "cp": "object", "trestbps": "int64", "chol": "int64", "fbs": "int64", "restecg": "object", "thalach": "int64", "exang": "int64", "oldpeak": "float64", "slope": "object", "ca": "int", "thal": "object", "target": "int64"})     

df.head()
df.dtypes
df = pd.get_dummies(df)

df.head()
df.dtypes
X = df.loc[:, ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak',

       'ca', 'cp_0', 'cp_1', 'cp_2', 'cp_3', 'restecg_0',

       'restecg_1', 'restecg_2', 'slope_0', 'slope_1', 'slope_2', 'thal_0',

       'thal_1', 'thal_2', 'thal_3']]

y = df["target"]
X.head(2)
X_train, X_test, y_train, y_test = train_test_split(X, y)



clf = LogisticRegression().fit(np.array(X_train), y_train)



predicted = clf.predict(np.array(X_test))



print("Accuracy: {}".format(accuracy_score(y_test, predicted)))

print("Recall: {}".format(recall_score(y_test, predicted)))

print("Precision: {}".format(precision_score(y_test, predicted)))

# print("AUC: {}".format(auc(y_test, predicted)))



fpr, tpr, d_thresholds = roc_curve(y_test, predicted)

plt.plot(fpr, tpr, label="Logistic Classifier")

plt.plot(d_fpr, d_tpr, color="green", linestyle="--", label="Dummy Classifier")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

print("AUC Score: {}".format(roc_auc_score(y_test, predicted)))
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier





clf_names = {SVC: "SVM Classifier", DecisionTreeClassifier: "Tree", RandomForestClassifier: "Random Forest"}

for classifier in [SVC, DecisionTreeClassifier, RandomForestClassifier]:

    clf = classifier().fit(np.array(X_train), y_train)



    predicted = clf.predict(np.array(X_test))

    print("Classifier: {}".format(clf_names[classifier]))

    print("Accuracy: {}".format(accuracy_score(y_test, predicted)))

    print("Recall: {}".format(recall_score(y_test, predicted)))

    print("Precision: {}".format(precision_score(y_test, predicted)))

    # print("AUC: {}".format(auc(y_test, predicted)))

    print("AUC Score: {}".format(roc_auc_score(y_test, predicted)))

    fpr, tpr, d_thresholds = roc_curve(y_test, predicted)

    plt.plot(fpr, tpr, label=clf_names[classifier])

    plt.plot(d_fpr, d_tpr, color="green", linestyle="--", label="Dummy Classifier")

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.show()
