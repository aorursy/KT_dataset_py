%matplotlib inline

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# from google.colab import drive

# drive.mount('/content/drive')



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('../input/minor-project-2020/train.csv')

test_df = pd.read_csv('../input/minor-project-2020/test.csv')

sample_submission_df = pd.read_csv('../input/minor-project-2020/sample_submission.csv')
train_df.head()
train_df.describe()
sns.heatmap(train_df.isnull(), yticklabels = False, cbar = False, cmap='viridis')
sns.set_style('whitegrid')

sns.countplot(x='target', data=train_df)
train_df['target'].value_counts()
train_df.info()
mask = np.zeros_like(train_df.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.set_style('whitegrid')

plt.subplots(figsize = (50,50))

sns.heatmap(train_df.corr(), 

            annot=False,

            mask = mask,

            cmap = 'RdBu', ## in order to reverse the bar replace "RdBu" with "RdBu_r"

            linewidths=.9, 

            linecolor='white',

            fmt='.2g',

            center = 0,

            square=True)
from imblearn.over_sampling import SMOTE



from sklearn.model_selection import train_test_split

y = train_df['target']

X = train_df.drop(['id','target'], axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



print("Shape of X_train dataset: ", X_train.shape)

print("Shape of y_train dataset: ", y_train.shape)

print("Shape of X_test dataset: ", X_test.shape)

print("Shape of y_test dataset: ", y_test.shape)
print("Before OverSampling")

print("Counts of label '1': {}".format(sum(y_train==1)))

print("Counts of label '0': {} \n".format(sum(y_train==0)))



sm = SMOTE(random_state=2)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())



print("After OverSampling")

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))

print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))



print("Counts of label '1': {}".format(sum(y_train_res==1)))

print("Counts of label '0': {}".format(sum(y_train_res==0)))
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

scaled_X_train_res = sc.fit_transform(X_train_res)

scaled_X_test = sc.transform(X_test)
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(max_iter=1000)

LR.fit(scaled_X_train_res, y_train_res)

y_pred = LR.predict(scaled_X_test)
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

print("Confusion Matrix")



print(confusion_matrix(y_test, y_pred))

plot_confusion_matrix(LR, scaled_X_test, y_test, cmap = plt.cm.Blues)

from sklearn.metrics import roc_curve, auc

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

plt.title('ROC for Titanic survivors', fontsize= 18)

plt.show()

X_test = test_df.drop(['id'], axis=1)

y_pred = LR.predict_proba(X_test)[:,1]
output = pd.DataFrame({'id': test_df.id, 'target': y_pred})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
from sklearn.model_selection import GridSearchCV



LR = LogisticRegression(max_iter=1000)

grid={"C":np.logspace(-4,4,20), "penalty":["l1","l2"]}

LR_CV = GridSearchCV(LR, param_grid=grid, n_jobs=-1, verbose=3, cv = 3)

LR_CV.fit(scaled_X_train_res, y_train_res)
LR_CV.best_params_
LR_OP = LogisticRegression(max_iter = 1000, C = 29.763514416313132, penalty = 'l2')

LR_OP.fit(scaled_X_train_res,y_train_res)

y_pred = LR_OP.predict(scaled_X_test)
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

print("Confusion Matrix")



print(confusion_matrix(y_test, y_pred))

plot_confusion_matrix(LR_OP, scaled_X_test, y_test, cmap = plt.cm.Blues)
from sklearn.metrics import roc_curve, auc

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

plt.title('ROC for Titanic survivors', fontsize= 18)

plt.show()
# X_test = test_df.drop(['id'], axis=1)

# y_pred = LR_OP.predict_proba(X_test)[:,1]
# output = pd.DataFrame({'id': test_df.id, 'target': y_pred})

# output.to_csv('my_submission.csv', index=False)

# print("Your submission was successfully saved!")
# from xgboost import XGBClassifier

# xgb = XGBClassifier()

# xgb.fit(X_train_res,y_train_res)

# y_pred = xgb.predict(X_test)
# from sklearn.metrics import roc_curve, auc

# FPR, TPR, _ = roc_curve(y_test, y_pred)

# ROC_AUC = auc(FPR, TPR)

# print (ROC_AUC)
# from imblearn.over_sampling import SMOTE



# from sklearn.model_selection import train_test_split

# y = train_df['target']

# X = train_df.drop(['id','target'], axis=1)



# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



# print("Number transactions X_train dataset: ", X_train.shape)

# print("Number transactions y_train dataset: ", y_train.shape)

# print("Number transactions X_test dataset: ", X_test.shape)

# print("Number transactions y_test dataset: ", y_test.shape)
# print("Before OverSampling")

# print("Counts of label '1': {}".format(sum(y_train==1)))

# print("Counts of label '0': {} \n".format(sum(y_train==0)))



# sm = SMOTE(random_state=2)

# X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())



# print("After OverSampling")

# print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))

# print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))



# print("Counts of label '1': {}".format(sum(y_train_res==1)))

# print("Counts of label '0': {}".format(sum(y_train_res==0)))
# from sklearn.preprocessing import StandardScaler

# sc=StandardScaler()

# scaled_X_train_res = sc.fit_transform(X_train_res)

# scaled_X_test = sc.transform(X_test)
# from sklearn.ensemble import RandomForestClassifier

# rf = RandomForestClassifier()

# rf.fit(scaled_X_train_res, y_train_res)

# y_pred = rf.predict(scaled_X_test)
# from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

# print("Confusion Matrix")



# print(confusion_matrix(y_test, y_pred))



# plot_confusion_matrix(rf, scaled_X_test, y_test, cmap = plt.cm.Blues)



# print("Classification Report: ")

# print(classification_report(y_test, y_pred))
# from sklearn.metrics import roc_curve, auc

# FPR, TPR, _ = roc_curve(y_test, y_pred)

# ROC_AUC = auc(FPR, TPR)

# print (ROC_AUC)
# from sklearn.model_selection import GridSearchCV
# rfc = RandomForestClassifier(random_state=42)

# param_grid = { 

#     'n_estimators': [100,200],

#     # 'max_features': ['auto', 'sqrt', 'log2'],

#     'max_depth' : [4,7,8],

#     'criterion' :['gini', 'entropy']

# }
# CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 3, n_jobs=-1, verbose=3)

# CV_rfc.fit(X_train_res,y_train_res)

# y_pred = CV_rfc.predict(X_test)
# X_test = test_df.drop(['id'], axis=1)

# y_pred = rf.predict(X_test)
# output = pd.DataFrame({'id': test_df.id, 'target': y_pred})

# output.to_csv('my_submission.csv', index=False)

# print("Your submission was successfully saved!")