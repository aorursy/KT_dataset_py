# Basic library for Python

import numpy as np

import pandas as pd

import os
os.listdir("../input/star-dataset/")
# Load star data

data = pd.read_csv("../input/star-dataset/6 class csv.csv")
# see the first five rows of data

data.head()
print("The shape of data is", data.shape)
# Let's see what kind of data types are in data

data.info()
# Take a look whole data

data.describe(include="all")
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
sns.pairplot(data = data, hue = "Star type")
sns.distplot(data["Temperature (K)"])

plt.show()
print("Temperature (k) skewnewss:", data["Temperature (K)"].skew())

print("Temperature (k) kurtosis:", data["Temperature (K)"].kurt())
sns.distplot(data["Luminosity(L/Lo)"])

plt.show()
print("Luminosity(L/Lo) skewnewss:", data["Luminosity(L/Lo)"].skew())

print("Luminosity(L/Lo) kurtosis:", data["Luminosity(L/Lo)"].kurt())
sns.distplot(data["Radius(R/Ro)"])

plt.show()
print("Radius(R/Ro) skewnewss:", data["Radius(R/Ro)"].skew())

print("Radius(R/Ro) kurtosis:", data["Radius(R/Ro)"].kurt())
sns.distplot(data["Absolute magnitude(Mv)"])

plt.show()
print("Absolute magnitude(Mv) skewnewss:", data["Absolute magnitude(Mv)"].skew())

print("Absolute magnitude(Mv) kurtosis:", data["Absolute magnitude(Mv)"].kurt())
ax = sns.countplot(data["Star color"])

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

plt.show()
data["Star color"].unique()
print("Star color has {} unique values.".format(len(data["Star color"].unique())))
data["Star type"].value_counts().plot(kind="bar")

plt.show()
data["Star type"].value_counts()
ax = sns.scatterplot(data = data, x = "Spectral Class", y = "Absolute magnitude(Mv)", hue = "Star type")

ax.invert_yaxis()
mapping_Spec_class = {"O": 0, "B": 1, "A": 2, "F": 3, "G": 4, "K": 5, "M": 6}

data["labeled_spec"] = data["Spectral Class"].map(mapping_Spec_class)
ax = sns.scatterplot(data = data, x = "labeled_spec", y = "Absolute magnitude(Mv)", hue = "Star type")

ax.invert_yaxis()
sns.catplot(y="labeled_spec", col="Star type", kind="count", orient = "h", height = 5, aspect = 1, data=data)

plt.show()
data[["Star type","labeled_spec"]].pivot_table(columns=["Star type"],index=["labeled_spec"], aggfunc=len, fill_value = 0)
sns.heatmap(data[["Star type","labeled_spec"]].pivot_table(columns=["Star type"],index=["labeled_spec"], aggfunc=len, fill_value = 0))

plt.show()
data.info()
#to lower case

data["Star color"] = data["Star color"].apply(lambda x: x.lower())
len(data["Star color"].unique())
#replace -  to " "

data["Star color"] = data["Star color"].apply(lambda x: x.replace("-"," "))

len(data["Star color"].unique())
#remove white space

data["Star color"] = data["Star color"].apply(lambda x: x.replace(" ",""))

len(data["Star color"].unique())
data["Star color"].unique()
dummy_color = pd.get_dummies(data["Star color"]).astype("int64")

new_data = pd.concat([data, dummy_color], axis=1)
new_data.head()
new_data.info()
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
X = new_data.drop(["Star type", "Star color", "Spectral Class"], axis=1)

y = new_data["Star type"]
X.info()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 523)
print("The shape of X_train:", X_train.shape)

print("The shape of X_test:", X_test.shape)
print("The shape of y_train:", y_train.shape)

print("The shape of y_test:", y_test.shape)
stsc = StandardScaler()

X_train_scaled = stsc.fit_transform(X_train.drop("labeled_spec", axis = 1))

X_test_scaled = stsc.transform(X_test.drop("labeled_spec", axis = 1))
X_train.columns
scaled_columns = list(X_train.columns)

scaled_columns.remove("labeled_spec")
X_train.loc[X_train.index,scaled_columns] = X_train_scaled

X_test.loc[X_test.index,scaled_columns] = X_test_scaled
X_train.head()
X_test.head()
stf = StratifiedKFold(n_splits=5, random_state=523)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_jobs=-1, n_neighbors=3)
cv_scores = cross_val_score(cv=stf, estimator=knn, X = X_train, y = y_train, scoring = "accuracy")
print("The Average accuracy for 5 folds KNN is {:.3f} +/- {:.3f}".format(cv_scores.mean(), cv_scores.std()))
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=523)

cv_scores_rf = cross_val_score(cv=stf, estimator=rf_clf, X = X_train, y = y_train, scoring = "accuracy")
print("The Average accuracy for 5 folds RF clf is {:.3f} +/- {:.3f}".format(cv_scores_rf.mean(), cv_scores_rf.std()))
from xgboost import XGBClassifier
xg_clf = XGBClassifier(n_estimators=1000, n_jobs=-1, random_state=523)

cv_scores_xg = cross_val_score(cv=stf, estimator=xg_clf, X = X_train, y = y_train, scoring = "accuracy")
print("The Average accuracy for 5 folds XGB clf is {:.3f} +/- {:.3f}".format(cv_scores_xg.mean(), cv_scores_xg.std()))
from lightgbm import LGBMClassifier
lgbm_clf = LGBMClassifier(n_estimators=1000, n_jobs=-1, random_state=523)

cv_scores_lgbm = cross_val_score(cv=stf, estimator=lgbm_clf, X = X_train, y = y_train, scoring = "accuracy")
print("The Average accuracy for 5 folds LGBM clf is {:.3f} +/- {:.3f}".format(cv_scores_lgbm.mean(), cv_scores_lgbm.std()))
from sklearn.metrics import confusion_matrix, classification_report
%%time

rf_clf.fit(X_train, y_train)

rf_pred = rf_clf.predict(X_test)
cfmx_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cfmx_rf,annot=True,cbar=False)

plt.ylabel('True Label')

plt.xlabel('Predicted Label')

plt.title('Confusion Matrix')

plt.show()
print(classification_report(y_test, rf_pred))
feat_importances = pd.Series(rf_clf.feature_importances_, index=X_train.columns)

feat_importances.nlargest(7).plot(kind='barh').invert_yaxis()
%%time

xg_clf.fit(X_train, y_train)

xg_pred = xg_clf.predict(X_test)
cfmx_xg = confusion_matrix(y_test, xg_pred)

sns.heatmap(cfmx_xg,annot=True,cbar=False)

plt.ylabel('True Label')

plt.xlabel('Predicted Label')

plt.title('Confusion Matrix')

plt.show()
print(classification_report(y_test, xg_pred))
feat_importances = pd.Series(xg_clf.feature_importances_, index=X_train.columns)

feat_importances.nlargest(7).plot(kind='barh').invert_yaxis()