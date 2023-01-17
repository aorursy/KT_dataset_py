%matplotlib inline



#import libraries

import warnings

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import fbeta_score, make_scorer, accuracy_score, precision_score, recall_score

from sklearn.exceptions import UndefinedMetricWarning
#loading data

data = pd.read_csv('../input/heart.csv')



#swap target variable

data["target"] = data["target"]==0
# get shape of dataframe

data.shape
# quick glance at the data

data.head()
# get most important statistics of variables 

data.describe()
# plot distribution of target variable

plt.bar(data["target"].value_counts().index, data["target"].value_counts())

plt.xticks(data["target"].unique())

plt.xlabel("Heart Disease")

plt.ylabel("Count")

plt.show()
data["target"].mean()
plt.figure(figsize=(20,5))

plt.subplot(121)

plt.hist(data["age"])

plt.xlabel("Age")

plt.ylabel("Count")

plt.subplot(122)

sns.violinplot(data["target"], data["age"])

plt.show()
data["age"].describe()
data.loc[data["target"]==1, "age"].describe()
data.loc[data["target"]==0, "age"].describe()
plt.figure(figsize=(20,5))

plt.subplot(131)

plt.bar(data["sex"].value_counts().index, data["sex"].value_counts())

plt.xticks(data["sex"].unique())

plt.xlabel("Sex")

plt.ylabel("Count")

plt.title("Patients")

plt.subplot(132)

plt.bar(data.loc[data["target"]==0,"sex"].value_counts().index, data.loc[data["target"]==0,"sex"].value_counts())

plt.xticks(data["sex"].unique())

plt.xlabel("Sex")

plt.ylabel("Count")

plt.title("Patients without heart disease")

plt.subplot(133)

plt.bar(data.loc[data["target"]==1,"sex"].value_counts().index, data.loc[data["target"]==1,"sex"].value_counts())

plt.xticks(data["sex"].unique())

plt.xlabel("Sex")

plt.ylabel("Count")

plt.title("Patients with heart disease")

plt.show()
data["sex"].mean()
data.loc[data["target"]==0,"sex"].mean()
data.loc[data["target"]==1,"sex"].mean()
plt.figure(figsize=(20,5))

plt.subplot(131)

plt.bar(data["cp"].value_counts().index, data["cp"].value_counts())

plt.xticks(data["cp"].unique())

plt.xlabel("Chest Pain Type")

plt.ylabel("Count")

plt.title("Patients")

plt.subplot(132)

plt.bar(data.loc[data["target"]==0,"cp"].value_counts().index, data.loc[data["target"]==0,"cp"].value_counts())

plt.xticks(data["cp"].unique())

plt.xlabel("Chest Pain Type")

plt.ylabel("Count")

plt.title("Patients without heart disease")

plt.subplot(133)

plt.bar(data.loc[data["target"]==1,"cp"].value_counts().index, data.loc[data["target"]==1,"cp"].value_counts())

plt.xticks(data["cp"].unique())

plt.xlabel("Chest Pain Type")

plt.ylabel("Count")

plt.title("Patients with heart disease")

plt.show()
data["cp"].value_counts() / len(data["cp"]) * 100
data.loc[data["target"]==0,"cp"].value_counts() / len(data.loc[data["target"]==0,"cp"]) * 100
data.loc[data["target"]==1,"cp"].value_counts() / len(data.loc[data["target"]==1,"cp"]) * 100
plt.figure(figsize=(20,5))

plt.subplot(121)

plt.hist(data["trestbps"])

plt.xlabel("Resting blood pressure (in mm Hg)")

plt.ylabel("Count")

plt.subplot(122)

sns.violinplot(data["target"], data["trestbps"])

plt.ylabel("Resting blood pressure (in mm Hg)")

plt.show()
data["trestbps"].describe()
data.loc[data["target"]==0, "trestbps"].describe()
data.loc[data["target"]==1, "trestbps"].describe()
plt.figure(figsize=(20,5))

plt.subplot(121)

plt.hist(data["chol"])

plt.xlabel("Serum Cholestoral (in mg/dl)")

plt.ylabel("Count")

plt.subplot(122)

sns.violinplot(data["target"], data["chol"])

plt.ylabel("Serum Cholestoral (in mg/dl)")

plt.show()
data["chol"].describe()
data.loc[data["target"]==0, "chol"].describe()
data.loc[data["target"]==1, "chol"].describe()
plt.figure(figsize=(20,5))

plt.subplot(131)

plt.bar(data["fbs"].value_counts().index, data["fbs"].value_counts())

plt.xticks(data["fbs"].unique())

plt.xlabel("Fasting blood sugar > 120 mg/dl")

plt.ylabel("Count")

plt.title("Patients")

plt.subplot(132)

plt.bar(data.loc[data["target"]==0,"fbs"].value_counts().index, data.loc[data["target"]==0,"fbs"].value_counts())

plt.xticks(data["fbs"].unique())

plt.xlabel("Fasting blood sugar > 120 mg/dl")

plt.ylabel("Count")

plt.title("Patients without heart disease")

plt.subplot(133)

plt.bar(data.loc[data["target"]==1,"fbs"].value_counts().index, data.loc[data["target"]==1,"fbs"].value_counts())

plt.xticks(data["fbs"].unique())

plt.xlabel("Fasting blood sugar > 120 mg/dl")

plt.ylabel("Count")

plt.title("Patients with heart disease")

plt.show()
data["fbs"].mean()
data.loc[data["target"]==0,"fbs"].mean()
data.loc[data["target"]==1,"fbs"].mean()
plt.figure(figsize=(20,5))

plt.subplot(131)

plt.bar(data["restecg"].value_counts().index, data["restecg"].value_counts())

plt.xticks(data["restecg"].unique())

plt.xlabel("Resting electrocardiographic results")

plt.ylabel("Count")

plt.title("Patients")

plt.subplot(132)

plt.bar(data.loc[data["target"]==0,"restecg"].value_counts().index, data.loc[data["target"]==0,"restecg"].value_counts())

plt.xticks(data["restecg"].unique())

plt.xlabel("Resting electrocardiographic results")

plt.ylabel("Count")

plt.title("Patients without heart disease")

plt.subplot(133)

plt.bar(data.loc[data["target"]==1,"restecg"].value_counts().index, data.loc[data["target"]==1,"restecg"].value_counts())

plt.xticks(data["restecg"].unique())

plt.xlabel("Resting electrocardiographic results")

plt.ylabel("Count")

plt.title("Patients with heart disease")

plt.show()
data["restecg"].value_counts() / len(data["restecg"]) * 100
data.loc[data["target"]==0,"restecg"].value_counts() / len(data.loc[data["target"]==0,"restecg"]) * 100
data.loc[data["target"]==1,"restecg"].value_counts() / len(data.loc[data["target"]==0,"restecg"]) * 100
plt.figure(figsize=(20,5))

plt.subplot(121)

plt.hist(data["thalach"])

plt.xlabel("Maximum Heart Rate (in bpm)")

plt.ylabel("Count")

plt.subplot(122)

sns.violinplot(data["target"], data["thalach"])

plt.ylabel("Maximum Heart Rate (in bpm)")

plt.show()
data["thalach"].describe()
data.loc[data["target"]==0, "thalach"].describe()
data.loc[data["target"]==1, "thalach"].describe()
plt.figure(figsize=(20,5))

plt.subplot(131)

plt.bar(data["exang"].value_counts().index, data["exang"].value_counts())

plt.xticks(data["exang"].unique())

plt.xlabel("Exercise induced angina")

plt.ylabel("Count")

plt.title("Patients")

plt.subplot(132)

plt.bar(data.loc[data["target"]==0,"exang"].value_counts().index, data.loc[data["target"]==0,"exang"].value_counts())

plt.xticks(data["exang"].unique())

plt.xlabel("Exercise induced angina")

plt.ylabel("Count")

plt.title("Patients without heart disease")

plt.subplot(133)

plt.bar(data.loc[data["target"]==1,"exang"].value_counts().index, data.loc[data["target"]==1,"exang"].value_counts())

plt.xticks(data["exang"].unique())

plt.xlabel("Exercise induced angina")

plt.ylabel("Count")

plt.title("Patients with heart disease")

plt.show()
data["exang"].value_counts() / len(data["exang"]) * 100
data.loc[data["target"]==0, "exang"].value_counts() / len(data.loc[data["target"]==0, "exang"]) * 100
data.loc[data["target"]==1, "exang"].value_counts() / len(data.loc[data["target"]==1, "exang"]) * 100
plt.figure(figsize=(20,5))

plt.subplot(121)

plt.hist(data["oldpeak"])

plt.xlabel("ST depression")

plt.ylabel("Count")

plt.subplot(122)

sns.violinplot(data["target"], data["oldpeak"])

plt.ylabel("ST depression")

plt.show()
data["oldpeak"].describe()
data.loc[data["target"]==0, "oldpeak"].describe()
data.loc[data["target"]==1, "oldpeak"].describe()
plt.figure(figsize=(20,5))

plt.subplot(131)

plt.bar(data["slope"].value_counts().index, data["slope"].value_counts())

plt.xticks(data["slope"].unique())

plt.xlabel("Slope of the peak exercise ST segment")

plt.ylabel("Count")

plt.title("Patients")

plt.subplot(132)

plt.bar(data.loc[data["target"]==0,"slope"].value_counts().index, data.loc[data["target"]==0,"slope"].value_counts())

plt.xticks(data["slope"].unique())

plt.xlabel("Slope of the peak exercise ST segment")

plt.ylabel("Count")

plt.title("Patients without heart disease")

plt.subplot(133)

plt.bar(data.loc[data["target"]==1,"slope"].value_counts().index, data.loc[data["target"]==1,"slope"].value_counts())

plt.xticks(data["slope"].unique())

plt.xlabel("Slope of the peak exercise ST segment")

plt.ylabel("Count")

plt.title("Patients with heart disease")

plt.show()
data["slope"].value_counts() / len(data["slope"]) * 100
data.loc[data["target"]==0, "slope"].value_counts() / len(data.loc[data["target"]==0, "slope"]) * 100
data.loc[data["target"]==1, "slope"].value_counts() / len(data.loc[data["target"]==1, "slope"]) * 100
plt.figure(figsize=(20,5))

plt.subplot(131)

plt.bar(data["ca"].value_counts().index, data["ca"].value_counts())

plt.xticks(data["ca"].unique())

plt.xlabel("Number of major vessels")

plt.ylabel("Count")

plt.title("Patients")

plt.subplot(132)

plt.bar(data.loc[data["target"]==0,"ca"].value_counts().index, data.loc[data["target"]==0,"ca"].value_counts())

plt.xticks(data["ca"].unique())

plt.xlabel("Number of major vessels")

plt.ylabel("Count")

plt.title("Patients without heart disease")

plt.subplot(133)

plt.bar(data.loc[data["target"]==1,"ca"].value_counts().index, data.loc[data["target"]==1,"ca"].value_counts())

plt.xticks(data["ca"].unique())

plt.xlabel("Number of major vessels")

plt.ylabel("Count")

plt.title("Patients with heart disease")

plt.show()
data["ca"].value_counts() / len(data["ca"]) * 100
data.loc[data["target"]==0, "ca"].value_counts() / len(data.loc[data["target"]==0, "ca"]) * 100
data.loc[data["target"]==1, "ca"].value_counts() / len(data.loc[data["target"]==1, "ca"]) * 100
plt.figure(figsize=(20,5))

plt.subplot(131)

plt.bar(data["thal"].value_counts().index, data["thal"].value_counts())

plt.xticks(data["thal"].unique())

plt.xlabel("Thallium Stress Tests")

plt.ylabel("Count")

plt.title("Patients")

plt.subplot(132)

plt.bar(data.loc[data["target"]==0,"thal"].value_counts().index, data.loc[data["target"]==0,"thal"].value_counts())

plt.xticks(data["thal"].unique())

plt.xlabel("Thallium Stress Tests")

plt.ylabel("Count")

plt.title("Patients without heart disease")

plt.subplot(133)

plt.bar(data.loc[data["target"]==1,"thal"].value_counts().index, data.loc[data["target"]==1,"thal"].value_counts())

plt.xticks(data["thal"].unique())

plt.xlabel("Thallium Stress Tests")

plt.ylabel("Count")

plt.title("Patients with heart disease")

plt.show()
data["thal"].value_counts() / len(data["thal"]) * 100
data.loc[data["target"]==0, "thal"].value_counts() / len(data.loc[data["target"]==0, "thal"]) * 100
data.loc[data["target"]==1, "thal"].value_counts() / len(data.loc[data["target"]==1, "thal"]) * 100
f = plt.figure(figsize=(19, 15))

plt.matshow(data.corr().apply(abs), fignum=f.number)

plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=45)

plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)

plt.title('Correlation Matrix', fontsize=16);
# dropping variables

data.drop('fbs', axis=1, inplace=True)
# split data in features and labels

features, labels = data.drop("target", axis=1), data["target"]
# encode as dummys

features_dummy = pd.get_dummies(features, columns=['restecg', 'slope', 'thal'], drop_first=True)
features_dummy.head()
# compare distributions

plt.figure(figsize=(20,5))

plt.subplot(1,2,1)

plt.hist(features_dummy['oldpeak'])

plt.title("Not Transformed")

plt.subplot(1,2,2)

plt.hist(np.log(features_dummy["oldpeak"]+1))

plt.title("Log Transformed")

plt.show()
# taking the log

features_dummy["oldpeak_log"] = np.log(features_dummy["oldpeak"]+1)
# drop old variable

features_dummy.drop('oldpeak', axis=1, inplace=True)
features_dummy.head()
ss = StandardScaler()
features_ss = pd.DataFrame(ss.fit_transform(features_dummy), columns=features_dummy.columns)
features_ss.head()
#filling the list with explained variance ratio by components

exp_var = PCA(16).fit(features_ss).explained_variance_ratio_
plt.figure(figsize=(10,5))

plt.plot(np.arange(1,17,1), np.cumsum(exp_var))

plt.bar(np.arange(1,17,1), exp_var)

plt.xticks(np.arange(1,17,1))

plt.xlabel("Number Components")

plt.ylabel("Explained Variance Ratio")

plt.show()
#compressing the data 

pca_4 = PCA(4, random_state=42)

features_compressed = pd.DataFrame(pca_4.fit_transform(features_ss))
#plotting the correlation matrix

f = plt.figure(figsize=(10, 10))

plt.matshow(features_compressed.corr().apply(abs), fignum=f.number)

plt.xticks(range(features_compressed.shape[1]), features_compressed.columns, fontsize=14, rotation=45)

plt.yticks(range(features_compressed.shape[1]), features_compressed.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)

plt.title('Correlation Matrix', fontsize=16);
plt.bar(np.arange(0,4,1),pca_4.explained_variance_ratio_)

plt.xticks(np.arange(0,4,1))

plt.xlabel("Component")

plt.ylabel("Explained Variance")

plt.title("Components")

plt.show()
plt.figure(figsize=(20, 10))

sns.heatmap(pd.DataFrame(pca_4.components_, columns = features_ss.columns).apply(abs), annot = True)

plt.xlabel("Features")

plt.ylabel("Components")

plt.title("Weights of components")

plt.show()
# split the data

features_train, features_test, labels_train, labels_test = train_test_split(features_compressed, labels, test_size=0.3, shuffle=True, random_state=42)
# define possible parameter

grid_param = {'C':[0.0001, 0.0005, 0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100]}

scorer = make_scorer(fbeta_score, beta=2)

lr_clf = LogisticRegression(random_state=42, solver='liblinear') 

lrg_clf = GridSearchCV(lr_clf, grid_param, scoring=scorer, cv=5, iid=True)
# disable metric warnings

warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

# fit models

lrg_clf = lrg_clf.fit(features_train, labels_train)
# get best model

lrg_clf.best_estimator_
# get best score

lrg_clf.best_score_
lrg_preds = lrg_clf.predict(features_test)
def evaluate(preds, labels):

    f2 = fbeta_score(preds, labels, beta=2)

    acc = accuracy_score(labels, preds)

    prec = precision_score(labels, preds)

    rec = recall_score(labels, preds)

    print("F2-Score: {0}, Accuracy: {1}, Precision: {2}, Recall: {3}".format(f2, acc, prec, rec))
evaluate(lrg_preds, labels_test)
# define possible parameter

grid_param = {'C':[0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100],

              'kernel':['linear','poly','rbf','sigmoid']}

svm_clf = SVC(random_state=42, gamma='auto') 

svmg_clf = GridSearchCV(svm_clf, grid_param, scoring=scorer, cv=5, iid=True)
# fit models

svmg_clf = svmg_clf.fit(features_train, labels_train)
# get best model

svmg_clf.best_estimator_
# get best score

svmg_clf.best_score_
svmg_preds = svmg_clf.predict(features_test)
# get evaluation scores

evaluate(svmg_preds, labels_test)
# define possible parameter

grid_param = {'n_estimators':[1,3,10,15,20,30,50,75,100],

              'learning_rate':[0.001,0.005,0.01,0.05,0.1,0.5,1,1.5,2,3,4,5]}

ada_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), random_state=42) 

adag_clf = GridSearchCV(ada_clf, grid_param, scoring=scorer, cv=5, iid=True)
# fit the model

adag_clf = adag_clf.fit(features_train, labels_train)
# get best model

adag_clf.best_estimator_
# get best score

adag_clf.best_score_
adag_preds = adag_clf.predict(features_test)
# get evaluation scores

evaluate(adag_preds, labels_test)
gnb_clf = GaussianNB()
# fitting the classifier

gnb_clf = gnb_clf.fit(features_train, labels_train)
# predicting on test set

gnb_preds = gnb_clf.predict(features_test)
# evaluate model

evaluate(gnb_preds, labels_test)