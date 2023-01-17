# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib as mplt

from matplotlib import pyplot as plt

import seaborn as sns

from scipy import stats



%matplotlib inline
df = pd.read_csv("/kaggle/input/minor-project-2020/train.csv")
print(df.head())

print(df.info())

print(df.describe())
df['target'].value_counts()
from sklearn.model_selection import train_test_split



y = df[['target']]

X = df.drop(['target', 'id'], axis = 1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=121)
from imblearn.over_sampling import RandomOverSampler

from sklearn.preprocessing import StandardScaler



ros = RandomOverSampler(random_state=121)



X_train_res, y_train_res = ros.fit_resample(X_train, y_train)



scalar = StandardScaler()

scaled_X_train = scalar.fit_transform(X_train_res)

scaled_X_test = scalar.fit_transform(X_test)
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix



tree = DecisionTreeClassifier()

tree.fit(scaled_X_train, y_train_res)

y_pred = tree.predict(scaled_X_test)

print("Accuracy is : {}".format(tree.score(scaled_X_test, y_test)))

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_curve, auc



plt.style.use('seaborn-pastel')



FPR, TPR, _ = roc_curve(y_test, y_pred)

ROC_AUC = auc(FPR, TPR)

print (ROC_AUC)



plt.figure(figsize =[11,9])

plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)

plt.plot([0,1],[0,1], 'k--', linewidth = 4)

plt.xlim([0.0,1.0])

plt.ylim([0.0,1.05])

plt.xlabel('False Positive Rate', fontsize = 18)

plt.ylabel('True Positive Rate', fontsize = 18)

plt.title('ROC', fontsize= 18)

plt.show()
from xgboost import XGBClassifier



xgb = XGBClassifier()

xgb.fit(scaled_X_train, y_train_res)



y_pred = xgb.predict(scaled_X_test)



print("Accuracy is : {}".format(xgb.score(scaled_X_test, y_test)))

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
plt.style.use('seaborn-pastel')



FPR, TPR, _ = roc_curve(y_test, y_pred)

ROC_AUC = auc(FPR, TPR)

print (ROC_AUC)



plt.figure(figsize =[11,9])

plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)

plt.plot([0,1],[0,1], 'k--', linewidth = 4)

plt.xlim([0.0,1.0])

plt.ylim([0.0,1.05])

plt.xlabel('False Positive Rate', fontsize = 18)

plt.ylabel('True Positive Rate', fontsize = 18)

plt.title('ROC', fontsize= 18)

plt.show()
from imblearn.over_sampling import SMOTE 

sm = SMOTE(random_state = 121) 

X_train_res, y_train_res = sm.fit_sample(X_train, y_train)



scalar = StandardScaler()

scaled_X_train_res = scalar.fit_transform(X_train_res)

scaled_X_test = scalar.transform(X_test)
tree_res = DecisionTreeClassifier()

tree_res.fit(scaled_X_train_res, y_train_res)

y_pred = tree_res.predict(scaled_X_test)



print("Accuracy is : {}".format(tree_rs.score(scaled_X_test, y_test)))

print("Confusion Matrix: ")



print(confusion_matrix(y_test, y_pred))
xgb_res = XGBClassifier()

xgb_res.fit(scaled_X_train_res, y_train_res)



y_pred = xgb.predict(scaled_X_test)



print("Accuracy is : {}".format(xgb.score(scaled_X_test, y_test)))

print(confusion_matrix(y_test, y_pred))
plt.style.use('seaborn-pastel')



FPR, TPR, _ = roc_curve(y_test, y_pred)

ROC_AUC = auc(FPR, TPR)

print (ROC_AUC)



plt.figure(figsize =[11,9])

plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)

plt.plot([0,1],[0,1], 'k--', linewidth = 4)

plt.xlim([0.0,1.0])

plt.ylim([0.0,1.05])

plt.xlabel('False Positive Rate', fontsize = 18)

plt.ylabel('True Positive Rate', fontsize = 18)

plt.title('ROC', fontsize= 18)

plt.show()
df = pd.read_csv("/kaggle/input/minor-project-2020/train.csv")

X_train = df.drop(['id', 'target'], axis = 1)

y = df[['target']]
from imblearn.over_sampling import RandomOverSampler

from sklearn.preprocessing import StandardScaler



ros = RandomOverSampler(random_state=121)



X_train_res, y_res = ros.fit_resample(X_train, y)

scalar = StandardScaler()

scaled_X_train = scalar.fit_transform(X_train_res)
df_test = pd.read_csv("/kaggle/input/minor-project-2020/test.csv")

X_test = df_test.drop(['id'], axis = 1)

scalar = StandardScaler()

scaled_X_test = scalar.fit_transform(X_test)
from xgboost import XGBClassifier



xgb_res = XGBClassifier()

xgb_res.fit(scaled_X_train, y_res)



pred = xgb_res.predict(scaled_X_test)
final_dict = {'id':df_test['id'].values,'target':pred}

finalpd = pd.DataFrame.from_dict(final_dict)

finalpd.to_csv("Submission_attempt5.csv",index=None)