# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, classification_report, recall_score, roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("../input/creditcard.csv")

print(dataset.head())

print(dataset.describe())



print(len(dataset[dataset.Class == 1]))

features = dataset.iloc[:, :-1]

print(features.shape)

label = dataset.iloc[:, -1].values

print(label.shape)



# heatmap for correlation, verifying that pca is already done

corrMat = features.corr()

sns.heatmap(corrMat, vmax=0.8)
fraudInd = np.asarray(np.where(label == 1))

noFraudInd = np.where(label == 0)

features = features.values



# data standarization (zero-mean, unit variance) ~ truncation to [-1, 1]

scaler = StandardScaler()

scaler.fit(features)

features = scaler.transform(features)



#import matplotlib.pyplot as plt

#fig = plt.figure()

#ax1 = fig.add_subplot(221)

#ax1.hist(features[noFraudInd,0], 50)

#ax2 = fig.add_subplot(222)

#ax2.hist(features[noFraudInd,1], 50)

#ax3 = fig.add_subplot(223)

#ax3.hist(features[noFraudInd,2], 50)

#ax4 = fig.add_subplot(224)

#ax4.hist(features[noFraudInd,3], 50)
TestPortion = 0.2

RND_STATE = 1



x_tr, x_test, y_tr, y_test = train_test_split(features, label, test_size = TestPortion, random_state = 1)



logreg = LogisticRegression(C = .01, penalty = 'l1', class_weight='balanced')

logreg.fit(x_tr,y_tr)

y_pred= logreg.predict(x_test)

print('------------ Results for LogiRegression ---------------')

print('cm:', confusion_matrix(y_test,y_pred))

#print('cr:', classification_report(y_test,y_pred))

#print('recall_score:', recall_score(y_test,y_pred))

print('roc_auc_score:',roc_auc_score(y_test,y_pred))



precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

area = auc(recall, precision)

print("Area Under P-R Curve: ",area)



clf = ExtraTreesClassifier(n_estimators =100)

clf.fit(x_tr, y_tr)

y_pred = clf.predict(x_test)

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

area = auc(recall, precision)



print('------------ Results for ExtraTreeClassifier ---------------')

print('cm:', confusion_matrix(y_test,y_pred))

#print('cr:', classification_report(y_test,y_pred))

#print('recall_score:', recall_score(y_test,y_pred))

print('roc_auc_score:',roc_auc_score(y_test,y_pred))

print("Area Under P-R Curve: ",area)



clf = RandomForestClassifier(n_estimators=100)    

clf.fit(x_tr, y_tr)

importances = clf.feature_importances_

#plt.figure()

#plt.title("Feature importances")

#plt.bar(range(x_tr.shape[1]), importances,

#       color="r", align="center")

#plt.xticks(range(x_tr.shape[1]))

#plt.xlim([-1, x_tr.shape[1]])

#plt.show()



y_pred = clf.predict(x_test)

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

area = auc(recall, precision)



print('------------ Results for RandomForestClassifier ---------------')

print('cm:', confusion_matrix(y_test,y_pred))

#print('cr:', classification_report(y_test,y_pred))

#print('recall_score:', recall_score(y_test,y_pred))

print('roc_auc_score:',roc_auc_score(y_test,y_pred))

print("Area Under P-R Curve: ",area)

xgb_model = xgb.XGBClassifier(n_estimators=100)

xgb_model.fit(x_tr, y_tr, verbose = 1)



y_pred = xgb_model.predict(x_test)

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

area = auc(recall, precision)



print('------------ Results for XGBClassifier ---------------')

print('cm:', confusion_matrix(y_test,y_pred))

#print('cr:', classification_report(y_test,y_pred))

#print('recall_score:', recall_score(y_test,y_pred))

print('roc_auc_score:',roc_auc_score(y_test,y_pred))

print("Area Under P-R Curve: ",area)