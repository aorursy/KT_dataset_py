# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns



heart = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
heart.describe()
heart.head(5)
heart.info()

# all features are in numerical format
heart.isnull().sum()

# no missing values
####################################### EDA ######################################

plt.figure(figsize=(16,8))

sns.countplot(x="target", data=heart)

plt.show()

# both unique values have almost equal proportions
plt.figure(figsize=(16,8))

sns.countplot(x="target", hue="cp", data=heart)

plt.show()

# patients with no chest pain has least chance to get heart disease
plt.figure(figsize=(16,8))

sns.countplot(x="target", hue="fbs", data=heart)

plt.show() 

# it seems like higher fasting blood sugar level does not necessarily increase chance of getting heart disease
plt.figure(figsize=(16,8))

sns.countplot(x="target", hue="exang", data=heart)

plt.show()

# it shows that exercise induced angina with value 0 has higher chance of inducing heart disease
plt.figure(figsize=(16,8))

sns.violinplot(x="target", y="trestbps", data=heart)

plt.show() 

# it shows that patients with high resting blood pressure has lesser chance of getting heart disease
plt.figure(figsize=(16,8))

sns.violinplot(x="target", y="chol", data=heart)

plt.show() 

# it shows that patients with heart disease have higher cholesterol level.
plt.figure(figsize=(16,8))

sns.violinplot(x="target", y="thalach", data=heart)

plt.show() 

# it seems that patients with higher maximum heart rate achieved have higher chance of getting heart disease
plt.figure(figsize=(16,8))

sns.violinplot(x="target", y="oldpeak", data=heart)

plt.show()

# it shows that patients with oldpeak close to 0 have higher chance of getting heart disease
plt.figure(figsize=(16,8))

sns.countplot(x="target", hue="slope", data=heart)

plt.show()

# it shows that patients with slope of 2 have higher chance of getting heart disease
plt.figure(figsize=(16,8))

sns.lineplot(x="age", y="chol",hue="target", data=heart)

plt.show() 

# at around age 68, it shows that high cholesterol level will increase chance of getting heart disease
plt.figure(figsize=(16,8))

sns.heatmap(heart.corr(), annot=True)

plt.show()
################################## Feature Engineer ##########################

# check for outliers and cardinality

plt.figure(figsize=(16,4))

sns.boxplot(x="age", data=heart)

plt.show()

# no outliers
heart["sex"].value_counts()

# it seems that male patients have higher number than female

# I will leave it untouched since there is nothing to do about it 
heart["cp"].value_counts()

# I will choose to leave it untouched
plt.figure(figsize=(16,4))

sns.boxplot(x="trestbps", data=heart)

plt.show()

# it has outliers at upper end
plt.figure(figsize=(16,4))

sns.boxplot(x="chol", data=heart)

plt.show()

# it has outliers at upper end

# there is extreme outlier at upper end
heart["fbs"].value_counts()

# I will choose to leave it untouched
heart["restecg"].value_counts()

# it has a rare label which consists of only 4 counts
plt.figure(figsize=(16,4))

sns.boxplot(x="thalach", data=heart)

plt.show()

# it has outliers at lower end
heart["exang"].value_counts()

# I will choose to leave it untouched
plt.figure(figsize=(16,4))

sns.boxplot(x="oldpeak", data=heart)

plt.show()

# it has outliers at upper end
heart["slope"].value_counts()

# I will choose to leave it untouched
heart["ca"].value_counts()

# from the description, it mentioned it has range of 0 to 3

# I will remove the rows with "ca" value of 4
heart["thal"].value_counts()

# it has a rare label which consists of only 2 counts
# remove rows with "ca" value of 4

heart = heart.drop(heart[heart["ca"] == 4].index)
# check for constant feature

from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold = 0)

sel.fit(heart)

[i for i in heart.columns if i not in heart.columns[sel.get_support()]]

# return empty list indicates there is no feature with constant value
# check for quasi-constant feature

sel = VarianceThreshold(threshold = 0.01)

sel.fit(heart)

[i for i in heart.columns if i not in heart.columns[sel.get_support()]]

# return empty list indicates there is no feature with quasi-constant value
# check for correlation

corrmat = heart.corr()

corrmat = corrmat.abs().unstack()

corrmat = corrmat[corrmat < 1]

corrmat = corrmat[corrmat >= 0.8]

list(corrmat) 

# empty list indicates there is no strong correlation among features
# as for those features with outliers and rare labels, I will build model with data remains untouched and another with cleaning done on data.
# data remains untouched

predictors = heart.drop("target", axis=1) 

labels = heart["target"]
# split the data into train and test datasets

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(predictors, labels, test_size=0.3, random_state=101)
#################### feature selection #####################################

from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score

model_all_features = XGBClassifier(max_depth=4, n_estimators=500, learning_rate=0.05)

model_all_features.fit(X_train, Y_train)

y_pred_test = model_all_features.predict_proba(X_test)[:,1]

auc_score_all = roc_auc_score(Y_test, y_pred_test)



features = pd.Series(model_all_features.feature_importances_)

features.index = X_train.columns

features.sort_values(ascending=True, inplace=True)

features = list(features.index)



tol = 0.0005

features_to_remove = []



for feature in features:

    model_int = XGBClassifier(max_depth=4, n_estimators=500, learning_rate=0.05)

    model_int.fit(X_train.drop(features_to_remove + [feature], axis=1), Y_train)

    y_pred_test = model_int.predict_proba(X_test.drop(features_to_remove + [feature], axis=1))[:,1]

    auc_score_int = roc_auc_score(Y_test, y_pred_test)

    diff_auc = auc_score_all - auc_score_int

    

    if diff_auc >= tol:

        print("keep", feature)

    else:

        auc_score_all = auc_score_int

        features_to_remove.append(feature)
# remove those least important features

X_train = X_train.drop(features_to_remove, axis=1)

X_test = X_test.drop(features_to_remove, axis=1)
# model building with XBGClassifier

model = XGBClassifier(max_depth=4, n_estimators=500, learning_rate=0.005)

model.fit(X_train, Y_train)

pred = model.predict_proba(X_test)[:,1]

round(np.mean((pred > 0.5) == Y_test),3) 

# accuracy = 81%
pd.crosstab(pred > 0.5, Y_test)

# Sensitivity = 42/(42+3) = 0.933

# Precision = 42/(14+42) = 0.750

# Specificity = 31/(31+14) = 0.689
# in this case, I will try to maximize sensitivity to eliminate any FN

pd.crosstab(pred > 0.187, Y_test)

# accuracy = (16+45)/90 = 0.678

# sensitivity = 45/(0+45) = 1.000
# I will try to build model with some cleaning on data

heart_removed = heart.copy()
heart_removed["trestbps"] = np.where(heart_removed["trestbps"] > 170, 170, heart_removed["trestbps"])



heart_removed["chol"] = np.where(heart_removed["chol"] > 371, 371, heart_removed["chol"])



heart_removed.drop(heart_removed[heart_removed["restecg"] == 2].index, inplace=True)



heart_removed["thalach"] = np.where(heart_removed["thalach"] > 213.25, 213.25, heart_removed["thalach"])



heart_removed["oldpeak"] = np.where(heart_removed["oldpeak"] > 4, 4, heart_removed["oldpeak"])



heart_removed.drop(heart_removed[heart_removed["thal"] == 0].index, inplace=True)
predictors_removed = heart_removed.drop("target", axis=1)

labels_removed = heart_removed["target"]
# split data into train and test data sets

X_train_removed, X_test_removed, Y_train_removed, Y_test_removed = train_test_split(predictors_removed, labels_removed, test_size=0.3, random_state=101)
# remove those least important features

X_train_removed = X_train_removed.drop(features_to_remove, axis=1)

X_test_removed = X_test_removed.drop(features_to_remove, axis=1)
# model building with new data

model = XGBClassifier(max_depth=4, n_estimators=500, learning_rate=0.005)

model.fit(X_train_removed, Y_train_removed)

pred = model.predict_proba(X_test_removed)[:,1]

round(np.mean((pred > 0.5) == Y_test_removed),3)

# accuracy = 86%
pd.crosstab(pred > 0.5, Y_test_removed)

# Sensitivity = 36/(36+7) = 0.837

# Precision = 36/(5+36) = 0.878

# Specificity = 40/(40+5) = 0.889
# in this case, I will try to maximize sensitivity to eliminate any FN

pd.crosstab(pred > 0.155, Y_test_removed)

# accuracy = (17+43)/88 = 0.682

# sensitivity = 43/(0+43) = 1.000
# As a conclusion, I will choose model building with data after done cleaning

# I will sacrifice some accuracy in order to elminiate any false negative(FN), by reducing classification threshold

# This is important because we do not wish any patient with heart disease get tested with negative result