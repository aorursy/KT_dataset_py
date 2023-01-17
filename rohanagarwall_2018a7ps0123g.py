# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/minor-project-2020/train.csv")
data
data.describe()
zeros=data[data['target']==0]
len(zeros)
ones=data[data['target']==1]
len(ones)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
corr = data.corr()



# print(corr)

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(64, 64))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=False)



plt.show()
Y_train=data['target']
x_train=data.drop(['id','col_39','col_47','col_28','col_0','target', 'col_46'], axis=1).values
x_train
scaled_x_train=sc.fit_transform(x_train)
scaled_x_train
Y_train
type(x_train)
x_train
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_train, Y_train, test_size=0.3, random_state=0)



print("Number transactions X_train dataset: ", X_train.shape)

print("Number transactions y_train dataset: ", y_train.shape)

print("Number transactions X_test dataset: ", X_test.shape)

print("Number transactions y_test dataset: ", y_test.shape)
print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))

print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))



sm = SMOTE(random_state=2)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())



print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))

print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))



print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))

print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
# from sklearn.preprocessing import MinMaxScaler



# # fit scaler on training data

# norm = MinMaxScaler().fit(X_train_res)



# # transform training data

# X_train_res = norm.transform(X_train_res)



# # transform testing dataabs

# X_test_res = norm.transform(X_test)
# from sklearn.model_selection import GridSearchCV

# from sklearn.linear_model import LogisticRegression

# from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report



# parameters = {

#     'C': np.linspace(1, 10, 10)

#              }

# lr = LogisticRegression(max_iter=1000)

# clf = GridSearchCV(lr, parameters, cv=5, verbose=5, n_jobs=-1)

# clf.fit(X_train_res, y_train_res.ravel())
# clf.best_params_
# import xgboost as xgb

# xg_reg = xgb.XGBRegressor(objective ='reg:logistic', colsample_bytree = 0.3, learning_rate = 0.1,

#                 max_depth = 5, alpha = 10, n_estimators = 10)
# xg_reg.fit(X_train_res, y_train_res.ravel())
from sklearn.linear_model import LogisticRegression
lr_sub = LogisticRegression(C=7,penalty='l2', verbose=5,max_iter = 1000)
lr_sub.fit(X_train_res, y_train_res.ravel())
from sklearn.metrics import roc_auc_score

y_pred_cv=lr_sub.predict_proba(X_test)[:,1]

roc_auc_score(y_test, y_pred_cv)
data2= pd.read_csv("/kaggle/input/minor-project-2020/test.csv")
data2
X_sub=data2.drop(['id','col_39','col_47','col_28','col_0','col_46'],axis=1).values
X_sub
# X_sub=sc.fit_transform(X_sub)
# norm = MinMaxScaler().fit(X_sub)

# X_sub=norm.transform(X_train_res)
Y_sub=lr_sub.predict_proba(X_sub)[:,1]
# Y_sub = xg_reg.predict(X_test)
Y_sub
test_id=data2['id']
test_id
res=[]
for i in range(len(test_id)):

    res.append([test_id[i], Y_sub[i]])
res
pd.DataFrame(res).to_csv("./answer.csv", header=['id','target'], index=None)