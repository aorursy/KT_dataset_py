import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.transforms as transforms

import seaborn as sns

import sklearn.metrics as metrics

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
df = pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")
df.head()
df.shape
df.info()
df.isnull().sum()
df.describe()
labels = df["class"].unique().tolist()

sizes = df["class"].value_counts().tolist()

colors = ["#C95555", "#D8AFAF"]

explode = (0, 0)

fig, ax = plt.subplots(1,2, figsize=(14,6))

sns.countplot(df["class"], palette="Oranges", ax=ax[0])

ax[0].set_title("Distribution of Patients", size=28, fontweight="bold")

ax[0].set_xlabel("Class", size=18, fontweight="bold")

ax[0].set_ylabel("Count", size=18, fontweight="bold")

plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90, textprops={'fontsize': 14, "fontweight" : "bold"}, colors=colors)

plt.title("Distribution of Patients", size=28, fontweight="bold")
plt.figure(figsize=(14,6))

df_corr = df.corr()

sns.heatmap(df_corr, annot=True, cmap="YlGn")

plt.tight_layout()
plt.figure(figsize=(16,8))

sns.scatterplot(df["pelvic_incidence"], df["sacral_slope"], hue=df["class"], palette="deep", s=100)

plt.ylabel("Sacral Slope", fontsize=15, fontweight="bold")

plt.xlabel("Pelvic Incidence", fontsize=15, fontweight="bold")

plt.title("Distribution of Patients with Respect to Pelvic Incidence \nAnd Sacral Slope", fontsize=22, fontweight="bold")

plt.legend(prop={"size":15})
plt.figure(figsize=(16,8))

sns.scatterplot(df["pelvic_radius"], df["sacral_slope"], hue=df["class"], palette="cubehelix", s=100)

plt.ylabel("Sacral Slope", fontsize=15, fontweight="bold")

plt.xlabel("Pelvic Radius", fontsize=15, fontweight="bold")

plt.title("Distribution of Patients with Respect to Pelvic Radius \nAnd Sacral Slope", fontsize=22, fontweight="bold")

plt.legend(prop={"size":15})
sns.pairplot(df, hue='class', height=2)
df_grouped = df.groupby("class").agg("mean")

df_grouped
df_grouped.plot.bar(rot=0, figsize=(16,8))

plt.title("Comparison of Abnormal and Normal Patients \n(Average Values)", fontsize=22, fontweight="bold")

plt.xlabel("Patient", fontsize=15, fontweight="bold")

plt.grid()

plt.legend(prop={"size":12})
df_grouped.plot.bar(rot=0, figsize=(16,8), subplots=True, layout=(3,2))
df_normal = df[df["class"] == "Normal"]

df_abnormal = df[df["class"] == "Abnormal"]

normal_df = df_normal.drop("class", axis=1)

abnormal_df = df_abnormal.drop("class", axis=1)

normal_columns = normal_df.columns.tolist()

abnormal_columns = abnormal_df.columns.tolist()
fig, axs = plt.subplots(3,2, figsize=(20,12))

fig.suptitle('Distribution Plots of Biomechanical Attributes for Normal Patients', 

             fontsize=25, fontweight="bold")

ax_iter = iter(axs.flat)

for columns in normal_columns:

    ax = next(ax_iter)

    sns.distplot(df_normal[columns], ax=ax)
fig, axs = plt.subplots(3,2, figsize=(20,12))

fig.suptitle('Distribution Plots of Biomechanical Attributes for Abnormal Patients', 

             fontsize=25, fontweight="bold")

ax_iter = iter(axs.flat)

for columns in abnormal_columns:

    ax = next(ax_iter)

    sns.distplot(df_abnormal[columns], ax=ax)
score_dict = {}

df["class"].replace({"Abnormal": 1, "Normal": 0}, inplace=True)

x = df.drop("class", axis=1)

y = df["class"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state=10)
neighbors_list = list(range(3,20,2))

knn_score_list = []



for number in neighbors_list:

    knn = KNeighborsClassifier(n_neighbors=number)

    knn.fit(X_train, y_train)

    y_predict_knn = knn.predict(X_test)

    knn_score_list.append(accuracy_score(y_test, y_predict_knn))

    

fig, ax = plt.subplots(1,1, figsize=(10,6))

plt.plot(neighbors_list, knn_score_list, marker="o", markerfacecolor="red", markersize=8)

plt.xticks(np.arange(3, 20, 2))

plt.xlabel("k value", size=12)

plt.ylabel("Accuracy Score", size=12)

ax.axhline(y = max(knn_score_list) , linewidth = 1.5, color = "red", linestyle="dashed")

trans = transforms.blended_transform_factory(

    ax.get_yticklabels()[0].get_transform(), ax.transData)

ax.text(0, max(knn_score_list), "{:.4f}".format(max(knn_score_list)), color="red", transform=trans, 

        ha="right", va="center")
knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train, y_train)
score_dict["KNN"] = knn.score(X_test, y_test)

y_predict_knn = knn.predict(X_test)

knn.score(X_test, y_test)
cm_knn = confusion_matrix(y_test, y_predict_knn)

plt.figure(figsize=(10,6))

sns.heatmap(cm_knn, annot=True, cmap="Blues")

plt.xlabel("Predicted")

plt.ylabel("True")
print(classification_report(y_test, y_predict_knn))
cross_val_score(estimator=knn, X = X_train, y = y_train, cv=5).mean()
param_grid = {'n_neighbors': np.arange(1,20)}

knn_gscv = GridSearchCV(knn, param_grid, cv=5)

knn_gscv.fit(X_train, y_train)

print("Tuned hyperparameter: {}".format(knn_gscv.best_params_)) 

print("Best score: {}".format(knn_gscv.best_score_))
lr = LogisticRegression(C = 0.1)

lr.fit(X_train, y_train)
score_dict["Logistic Regression"] = lr.score(X_test, y_test)

y_predict_lr = lr.predict(X_test)

lr.score(X_test, y_test)
cm_lr = confusion_matrix(y_test, y_predict_lr)

plt.figure(figsize=(10,6))

sns.heatmap(cm_lr, annot=True, cmap="Blues")

plt.xlabel("Predicted")

plt.ylabel("True")
print(classification_report(y_test, y_predict_lr))
cross_val_score(estimator=lr, X = X_train, y = y_train, cv=5).mean()
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

lr_gscv = GridSearchCV(lr, param_grid, cv=5)

lr_gscv.fit(X_train, y_train)

print("Tuned hyperparameters {}".format(lr_gscv.best_params_)) 

print("Best score: {}".format(lr_gscv.best_score_))
dtc = DecisionTreeClassifier(criterion = 'gini', max_depth = 4, min_samples_split = 50, random_state=10)

dtc.fit(X_train, y_train)
score_dict["Decision Tree Classifier"] = dtc.score(X_test, y_test)

y_predict_dtc = dtc.predict(X_test)

dtc.score(X_test, y_test)
cm_dtc = confusion_matrix(y_test, y_predict_dtc)

plt.figure(figsize=(10,6))

sns.heatmap(cm_dtc, annot=True, cmap="Blues")

plt.xlabel("Predicted")

plt.ylabel("True")
print(classification_report(y_test, y_predict_dtc))
cross_val_score(estimator=dtc, X = X_train, y = y_train, cv=5).mean()
param_grid = {"criterion" : ['gini', 'entropy'], "max_depth" : np.arange(2,21,2), 

              'min_samples_split' : np.arange(10,200,10)}

dtc_gscv = GridSearchCV(dtc, param_grid, cv=5)

dtc_gscv.fit(X_train, y_train)

print("Tuned hyperparameters {}".format(dtc_gscv.best_params_)) 

print("Best score: {}".format(dtc_gscv.best_score_))
rfc = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=10)

rfc.fit(X_train, y_train)
rfc.score(X_test, y_test)
cross_val_score(estimator=rfc, X = X_train, y = y_train, cv=5).mean()
param_grid = {'n_estimators': np.arange(100,500,50), 'max_depth' : [4,5,6,7,8], 

              'criterion' :['gini', 'entropy']}

rfc_gscv = GridSearchCV(rfc, param_grid, cv=5)

rfc_gscv.fit(X_train,y_train)

print("Tuned hyperparameters {}".format(rfc_gscv.best_params_)) 

print("Best score: {}".format(rfc_gscv.best_score_))
rfc = RandomForestClassifier(criterion = 'gini', max_depth = 4, n_estimators = 200, random_state=10)

rfc.fit(X_train, y_train)
score_dict["Random Forest Classifier"] = rfc.score(X_test, y_test)

y_predict_rfc = rfc.predict(X_test)

rfc.score(X_test, y_test)
cross_val_score(estimator=rfc, X = X_train, y = y_train, cv=5).mean()
cm_rfc = confusion_matrix(y_test, y_predict_rfc)

plt.figure(figsize=(10,6))

sns.heatmap(cm_rfc, annot=True, cmap="Blues")

plt.xlabel("Predicted")

plt.ylabel("True")
print(classification_report(y_test, y_predict_rfc))
svm = SVC(kernel="linear", probability=True)

svm.fit(X_train, y_train)
score_dict["SVM"] = svm.score(X_test, y_test)

y_predict_svm = svm.predict(X_test)

svm.score(X_test, y_test)
cross_val_score(estimator=svm, X = X_train, y = y_train, cv=5).mean()
cm_svm = confusion_matrix(y_test, y_predict_svm)

plt.figure(figsize=(10,6))

sns.heatmap(cm_svm, annot=True, cmap="Blues")

plt.xlabel("Predicted")

plt.ylabel("True")
print(classification_report(y_test, y_predict_svm))
score_dict
models = score_dict.keys()

scores = score_dict.values()



plt.figure(figsize=(16,6))

plt.bar(models, scores, color="#A67EB0")

plt.yticks(np.arange(0, 1.05, 0.05))

plt.xlabel("Model", fontsize=15)

plt.ylabel("Accuracy Score", fontsize=15)

for i, v in enumerate(score_dict.values()):

    plt.text(i-0.1, v+0.03, "{:.4f}".format(v), color='black', va='center', fontweight='bold')
prob_knn = knn.predict_proba(X_test)[:,1]

prob_lr = lr.predict_proba(X_test)[:,1]

prob_dtc = dtc.predict_proba(X_test)[:,1]

prob_rfc = rfc.predict_proba(X_test)[:,1]

prob_svm = svm.predict_proba(X_test)[:,1]



prob_dict = {"ROC KNN": prob_knn, "ROC LR": prob_lr, "ROC DTC": prob_dtc, 

             "ROC RFC": prob_rfc, "ROC SVM": prob_svm}



for model, prob in prob_dict.items():

    fpr, tpr, threshold = metrics.roc_curve(y_test, prob)

    roc_auc = metrics.auc(fpr, tpr)

    plt.figure(figsize=(10,6))

    plt.plot(fpr, tpr, color = "b", label = "AUC = %0.2f" %roc_auc)

    plt.legend(loc="lower right", prop={"size":15})

    #plt.xlim([-0.005,1])

    #plt.ylim([0,1.015])

    plt.xlabel("False Positive Rate", size=12)

    plt.ylabel("True Positive Rate", size=12)

    plt.plot([0,1], [0,1], "r--")

    plt.title(str(model), size=20)