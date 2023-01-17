import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost.sklearn import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import auc, roc_curve

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_recall_fscore_support

from sklearn.preprocessing import StandardScaler as ss

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn import preprocessing
data = pd.read_csv("../input/data.csv")
print(data.head())
data.describe()
print(data.describe())
data.shape
# feature names as a list

col = data.columns
print(col)
# Drop useless variables

data = data.drop(['Unnamed: 32','id'],axis = 1)



# Reassign target

data.diagnosis.replace(to_replace = dict(M = 1, B = 0), inplace = True)
# 2 datasets

M = data[(data['diagnosis'] != 0)]

B = data[(data['diagnosis'] == 0)]
X = data.drop("diagnosis", axis=1)

y = data["diagnosis"].values
scale = ss()

X = scale.fit_transform(X)

print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)
dt = DecisionTreeClassifier()

rf = RandomForestClassifier(n_estimators=50)

etc = ExtraTreesClassifier(n_estimators=50)

knc = KNeighborsClassifier()

xg = XGBClassifier(learning_rate=0.5, reg_alpha= 5, reg_lambda= 0.1)

gbm = GradientBoostingClassifier()
#Train the data

dt1 = dt.fit(X_train,y_train)

rf1 = rf.fit(X_train,y_train)

etc1 = etc.fit(X_train,y_train)

knc1 = knc.fit(X_train,y_train)

xg1 = xg.fit(X_train,y_train)

gbm1 = gbm.fit(X_train,y_train)
#Predict the data

y_pred_dt = dt1.predict(X_test)

y_pred_rf = rf1.predict(X_test)

y_pred_etc= etc1.predict(X_test)

y_pred_knc= knc1.predict(X_test)

y_pred_xg= xg1.predict(X_test)

y_pred_gbm= gbm1.predict(X_test)
#Fetch probabilities

y_pred_dt_prob = dt1.predict_proba(X_test)

y_pred_rf_prob = rf1.predict_proba(X_test)

y_pred_etc_prob = etc1.predict_proba(X_test)

y_pred_knc_prob = knc1.predict_proba(X_test)

y_pred_xg_prob = xg1.predict_proba(X_test)

y_pred_gbm_prob= gbm1.predict_proba(X_test)
#Get accuracy scores

accuracy_score(y_test,y_pred_dt)

accuracy_score(y_test,y_pred_rf)

accuracy_score(y_test,y_pred_etc)

accuracy_score(y_test,y_pred_knc)

accuracy_score(y_test,y_pred_xg)

accuracy_score(y_test,y_pred_gbm)
print(accuracy_score(y_test,y_pred_dt))

print(accuracy_score(y_test,y_pred_rf))

print(accuracy_score(y_test,y_pred_etc))

print(accuracy_score(y_test,y_pred_knc))

print(accuracy_score(y_test,y_pred_xg))

print(accuracy_score(y_test,y_pred_gbm))
#Confusion matrix

confusion_matrix(y_test,y_pred_dt)
confusion_matrix(y_test,y_pred_rf)
#ROC Graph

fpr_dt, tpr_dt, thresholds = roc_curve(y_test, y_pred_dt_prob[: , 1], pos_label= 1)
fpr_rf, tpr_rf, thresholds = roc_curve(y_test, y_pred_rf_prob[: , 1], pos_label= 1)
#Fetch AUC

auc(fpr_dt,tpr_dt)
auc(fpr_rf,tpr_rf)
#Calculate Precision, Recall and F-score

precision_recall_fscore_support(y_test,y_pred_dt)
precision_recall_fscore_support(y_test,y_pred_rf)
#Plotting ROC Curve

fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111)

ax.set_xlabel('False Positive Rate')

ax.set_ylabel('True Positive Rate')

ax.set_title('ROC Curve')

ax.plot(fpr_dt, tpr_dt, label = "dt")

ax.plot(fpr_rf, tpr_rf, label = "rf")

plt.show()
drop_list1 = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean','radius_se','perimeter_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst','compactness_se','concave points_se','texture_worst','area_worst']

data_1 = data.drop(drop_list1,axis = 1 )        

data_1.head()
ax = sns.countplot(y,label="Count")

y = data.diagnosis

B, M = y.value_counts()

print('Number of Benign: ',B)

print('Number of Malignant : ',M)
#correlation map

f,ax = plt.subplots(figsize=(14, 14))

sns.heatmap(data_1.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
cm = confusion_matrix(y_test,rf.predict(X_test))

sns.heatmap(cm,annot=True,fmt="d")
# seaborn version : Uncorrelated features

fig = plt.figure(figsize=(12,12))

palette ={0 : 'lightblue', 1 : 'gold'}

edgecolor = 'blue'

plt.subplot(221)

ax1 = sns.scatterplot(x = data['smoothness_mean'], y = data['texture_mean'], hue = "diagnosis",

                    data = data, palette =palette, edgecolor=edgecolor)

plt.title('smoothness mean vs texture mean')

plt.subplot(222)

ax2 = sns.scatterplot(x = data['radius_mean'], y = data['fractal_dimension_worst'], hue = "diagnosis",

                    data = data, palette =palette, edgecolor=edgecolor)

plt.title('radius mean vs fractal dimension_worst')

plt.subplot(223)

ax3 = sns.scatterplot(x = data['texture_mean'], y = data['symmetry_mean'], hue = "diagnosis",

                    data = data, palette =palette, edgecolor=edgecolor)

plt.title('texture mean vs symmetry mean')

plt.subplot(224)

ax4 = sns.scatterplot(x = data['texture_mean'], y = data['symmetry_se'], hue = "diagnosis",

                    data = data, palette =palette, edgecolor=edgecolor)

plt.title('texture mean vs symmetry se')



fig.suptitle('Uncorrelated features', fontsize = 20)

plt.savefig('2')

plt.show()
confusion_matrix(y_test,y_pred_etc)

confusion_matrix(y_test,y_pred_knc)

confusion_matrix(y_test,y_pred_xg)

confusion_matrix(y_test,y_pred_gbm)
fpr_etc, tpr_etc, thresholds = roc_curve(y_test, y_pred_etc_prob[: , 1], pos_label= 1)
fpr_knc, tpr_knc, thresholds = roc_curve(y_test, y_pred_knc_prob[: , 1], pos_label= 1)
fpr_xg, tpr_xg, thresholds = roc_curve(y_test, y_pred_xg_prob[: , 1], pos_label= 1)
fpr_gbm, tpr_gbm, thresholds = roc_curve(y_test, y_pred_gbm_prob[: , 1], pos_label= 1)
auc(fpr_etc,tpr_etc)
auc(fpr_knc,tpr_knc)
auc(fpr_xg,tpr_xg)
auc(fpr_gbm,tpr_gbm)
precision_recall_fscore_support(y_test,y_pred_etc)
precision_recall_fscore_support(y_test,y_pred_knc)
precision_recall_fscore_support(y_test,y_pred_xg)
precision_recall_fscore_support(y_test,y_pred_gbm)
#Plotting ROC Curve

fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111)

ax.set_xlabel('False Positive Rate')

ax.set_ylabel('True Positive Rate')

ax.set_title('ROC Curve')

ax.plot(fpr_dt, tpr_dt, label = "dt")

ax.plot(fpr_rf, tpr_rf, label = "rf")

ax.plot(fpr_etc, tpr_etc, label = "etc")

ax.plot(fpr_knc, tpr_knc, label = "knc")

ax.plot(fpr_xg, tpr_xg, label = "xg")

ax.plot(fpr_gbm, tpr_gbm, label = "gbm")

plt.show()