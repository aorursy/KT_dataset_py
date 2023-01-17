import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.transforms as transforms

import seaborn as sns

import sklearn.metrics as metrics

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score

import os

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
df.head()
df.shape
df.info()
df.isnull().sum()
df.describe()
plt.figure(figsize=(16,8))

sns.countplot(df["quality"], palette="Oranges")

plt.title("Distribution of Wine Qualities", size=28, fontweight="bold")

plt.xlabel("Quality", size=18, fontweight="bold")

plt.ylabel("Count", size=18, fontweight="bold")
plt.figure(figsize=(16,8))

sns.distplot(df['quality'], color="red")

plt.xlabel("Quality", size=18, fontweight="bold")
mean_alcohol_list = []

mean_ph_list = []

mean_density_list = []

mean_residual_sugar_list = []



quality_list = df.quality.unique()

quality_list.sort()

for i in quality_list:

    df_quality = df[df["quality"]==int(i)]

    mean_alcohol = df_quality["alcohol"].mean()

    mean_ph = df_quality["pH"].mean()

    mean_density = df_quality["density"].mean()

    mean_residual_sugar = df_quality["residual sugar"].mean()

    

    mean_alcohol_list.append(mean_alcohol)

    mean_ph_list.append(mean_ph)

    mean_density_list.append(mean_density)

    mean_residual_sugar_list.append(mean_residual_sugar)



fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(26,12))

ax1.bar(quality_list, mean_alcohol_list, color = "#ECC679")

ax1.set_title("Average Alcohol Values of Qualities ", size=20)

ax1.set_xlabel("Quality", size=13)

ax1.set_ylabel("Alcohol", size=13)

ax2.bar(quality_list, mean_ph_list, color = "skyblue")

ax2.set_title("Average pH Values of Qualities ", size=20)

ax2.set_xlabel("Quality", size=13)

ax2.set_ylabel("pH", size=13)

ax2.set_ylim([3,3.5])

ax3.bar(quality_list, mean_density_list, color = "green")

ax3.set_title("Average Density Values of Qualities ", size=20)

ax3.set_xlabel("Quality", size=13)

ax3.set_ylabel("Density", size=13)

ax3.set_ylim([0.99,1.0])

ax4.bar(quality_list, mean_residual_sugar_list, color = "gray")

ax4.set_title("Average Residual Sugar Values of Qualities ", size=20)

ax4.set_xlabel("Quality", size=13)

ax4.set_ylabel("Residual Sugar", size=13)

ax4.set_ylim([1.5,3])

plt.tight_layout(pad=3)
plt.figure(figsize=(16,8))

df_corr = df.corr()

sns.heatmap(df_corr, annot=True, cmap="GnBu")
for quality in df.quality:

    if quality < 6.5:

        df['quality'] = df['quality'].replace([int(quality)],'Bad')

    else:

        df['quality'] = df['quality'].replace([int(quality)],'Good')

plt.figure(figsize = (16,8))

labels = df["quality"].unique().tolist()

sizes = df["quality"].value_counts().tolist()

colors = ["#59CBC0", "#BFD7D4"]

explode = (0, 0)

plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90, textprops={'fontsize': 14, "fontweight" : "bold"}, colors=colors)

plt.title("Distribution of Wines", size=28, fontweight="bold")
recall_score_dict = {}

acc_score_dict = {}

precision_score_dict = {}



le = LabelEncoder()

df["quality"] = le.fit_transform(df["quality"])
x = df.drop("quality", axis=1)

y = df["quality"]
ss = StandardScaler()

x = ss.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 42)
lr = LogisticRegression()

lr.fit(X_train, y_train)
y_predict_lr = lr.predict(X_test)

print("Accuracy Score :",lr.score(X_test, y_test))
cm_lr = confusion_matrix(y_test, y_predict_lr)

plt.figure(figsize=(10,6))

sns.heatmap(cm_lr, annot=True, cmap="Blues", fmt=".1f")

plt.xlabel("Predicted")

plt.ylabel("True")
print(classification_report(y_test, y_predict_lr))
from imblearn.over_sampling import SMOTE



sm = SMOTE(random_state=42)

X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("Quality Distribution Before SMOTE Operation : \n", y_train.value_counts(), "\n")

print("Quality Distribution After SMOTE Operation : \n" ,y_train_res.value_counts())
plt.figure(figsize=(16,8))

colors = ["#59CBC0", "#BFD7D4"]

labels = ["Bad", "Good"]

y_train_res.value_counts().plot(kind="pie",shadow=True, autopct='%1.1f%%', 

                                textprops={'fontsize': 14, "fontweight" : "bold"},

                                colors = colors, labels=labels)

plt.title("Distribution of Wine Qualities After SMOTE", size=15)
lr2 = LogisticRegression()

lr2.fit(X_train_res, y_train_res)
y_predict_lr2 = lr2.predict(X_test)

acc_score_dict["LR"] = lr2.score(X_test, y_test)

recall_score_dict["LR"] =recall_score(y_test, y_predict_lr2)

precision_score_dict["LR"] = precision_score(y_test, y_predict_lr2)

print("Accuracy Score :",lr2.score(X_test, y_test))
cm_lr2 = confusion_matrix(y_test, y_predict_lr2)

plt.figure(figsize=(10,6))

sns.heatmap(cm_lr2, annot=True, cmap="Blues", fmt=".1f")

plt.xlabel("Predicted")

plt.ylabel("True")
print(classification_report(y_test, y_predict_lr2))
print("CV Score : ", cross_val_score(estimator=lr2, X = X_train_res, y = y_train_res, cv=5).mean())
dtc = DecisionTreeClassifier(criterion = 'gini', min_samples_split = 10, random_state=42)

dtc.fit(X_train_res, y_train_res)
y_predict_dtc = dtc.predict(X_test)

acc_score_dict["DTC"] = dtc.score(X_test, y_test)

recall_score_dict["DTC"] = recall_score(y_test, y_predict_dtc)

precision_score_dict["DTC"] = precision_score(y_test, y_predict_dtc)

print("Accuracy Score :",dtc.score(X_test, y_test))
cm_dtc = confusion_matrix(y_test, y_predict_dtc)

plt.figure(figsize=(10,6))

sns.heatmap(cm_dtc, annot=True, cmap="Blues", fmt=".1f")

plt.xlabel("Predicted")

plt.ylabel("True")
print(classification_report(y_test, y_predict_dtc))
print("CV Score : ", cross_val_score(estimator=dtc, X = X_train_res, y = y_train_res, cv=5).mean())
rfc = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=10)

rfc.fit(X_train_res, y_train_res)
y_predict_rfc = rfc.predict(X_test)

acc_score_dict["RFC"] = rfc.score(X_test, y_test)

recall_score_dict["RFC"] =recall_score(y_test, y_predict_rfc)

precision_score_dict["RFC"] = precision_score(y_test, y_predict_rfc)

print("Accuracy Score :",rfc.score(X_test, y_test))
cm_rfc = confusion_matrix(y_test, y_predict_rfc)

plt.figure(figsize=(10,6))

sns.heatmap(cm_rfc, annot=True, cmap="Blues",fmt=".1f")

plt.xlabel("Predicted")

plt.ylabel("True")
print(classification_report(y_test, y_predict_rfc))
print("CV Score : ", cross_val_score(estimator=rfc, X = X_train_res, y = y_train_res, cv=5).mean())
svm = SVC(probability=True)

svm.fit(X_train_res, y_train_res)
y_predict_svm = svm.predict(X_test)

acc_score_dict["SVM"] = svm.score(X_test, y_test)

recall_score_dict["SVM"] =recall_score(y_test, y_predict_svm)

precision_score_dict["SVM"] = precision_score(y_test, y_predict_svm)

print("Accuracy Score :",svm.score(X_test, y_test))
cm_svm = confusion_matrix(y_test, y_predict_svm)

plt.figure(figsize=(10,6))

sns.heatmap(cm_svm, annot=True, cmap="Blues",fmt=".1f")

plt.xlabel("Predicted")

plt.ylabel("True")
print(classification_report(y_test, y_predict_svm))
print("CV Score : ", cross_val_score(estimator=svm, X = X_train_res, y = y_train_res, cv=5).mean())
print("Accuracy Scores for Each Model After SMOTE: ",acc_score_dict)

print("Recall Scores for Each Model After SMOTE: ",recall_score_dict)

print("Precision Scores for Each Model After SMOTE: ",precision_score_dict)
labels = acc_score_dict.keys()

acc_scores = acc_score_dict.values()

recall_scores = recall_score_dict.values()

precision_scores = precision_score_dict.values()



x = np.arange(len(labels))

width = 0.30



fig, ax = plt.subplots(figsize=(16,8))

rects1 = ax.bar(x - width, acc_scores, width, label='Accuracy', color="#056937")

rects2 = ax.bar(x, recall_scores, width, label='Recall', color="#062D5F")

rects3 = ax.bar(x + width, precision_scores, width, label='Precision', color="#AE4D4D")



ax.set_xlabel('Model', fontsize=15)

ax.set_ylabel('Score', fontsize=15)

ax.set_title('Comparison of Accuracy, Recall and Precision Scores of Models', fontsize=22, fontweight="bold", fontstyle="italic")

ax.set_xticks(x)

ax.set_xticklabels(labels)

plt.ylim([0,1])

plt.xticks(fontsize=16)

legend = ax.legend(bbox_to_anchor=(1, 1), loc='upper left',prop={"size":18})

legend.set_title('Score',prop={'size':20})



for i, v in enumerate(acc_score_dict.values()):

    plt.text(i-0.43, v+0.025, "{:.4f}".format(v), color='#056937', va='center', fontweight='bold', size=14)

for i, v in enumerate(recall_score_dict.values()):

    plt.text(i-0.12, v+0.025, "{:.4f}".format(v), color='#062D5F', va='center', fontweight='bold',size=14)

for i, v in enumerate(precision_score_dict.values()):

    plt.text(i+0.17, v+0.025, "{:.4f}".format(v), color='#AE4D4D', va='center', fontweight='bold',size=14)

prob_lr = lr2.predict_proba(X_test)[:,1]

prob_dtc = dtc.predict_proba(X_test)[:,1]

prob_rfc = rfc.predict_proba(X_test)[:,1]

prob_svm = svm.predict_proba(X_test)[:,1]



prob_dict = {"ROC LR": prob_lr, "ROC DTC": prob_dtc, "ROC RFC": prob_rfc, "ROC SVM": prob_svm}



for model, prob in prob_dict.items():

    fpr, tpr, threshold = metrics.roc_curve(y_test, prob)

    roc_auc = metrics.auc(fpr, tpr)

    plt.figure(figsize=(10,6))

    plt.plot(fpr, tpr, color = "b", label = "AUC = %0.2f" %roc_auc)

    plt.legend(loc="lower right", prop={"size":15})

    plt.xlabel("False Positive Rate", size=12)

    plt.ylabel("True Positive Rate", size=12)

    plt.plot([0,1], [0,1], "r--")

    plt.title(str(model), size=20)