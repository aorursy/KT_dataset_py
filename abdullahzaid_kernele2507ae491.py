# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/dont-overfit-ii/train.csv')
test = pd.read_csv('/kaggle/input/dont-overfit-ii/test.csv')
labels = train.columns.drop(['id', 'target'])
train.shape
sns.countplot(x = 'target', data = train, palette = 'hls')
plt.show
plt.savefig('count')
X = train.drop(['id','target'],axis = 1)
Y = train['target']
X_eval = test.drop(['id'], axis = 1)
X_eval.shape

modelXGB = XGBClassifier(max_depth = 2, gamma = 2, eta = 0.8, reg_alpha = 0.5, reg_lambda = 0.5)
rfe = RFE(modelXGB)
rfe.fit(X,Y)
print('selected features:')
print(labels[rfe.support_].tolist())
X_fs = rfe.transform(X)
X_fs_eval = rfe.transform(X_eval)

labels_fs = X_fs
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority', n_jobs=-1)
X_sm, y_sm = smote.fit_resample(X_fs,Y)

df = pd.DataFrame(X_sm)
df['target'] = y_sm

sns.countplot(x = 'target', data = df, palette = 'hls')
plt.show
plt.savefig('count')

normX = df.drop(['target'], axis = 1)
normY = df['target']
modelLR = LogisticRegression(solver = 'liblinear',C = 0.05, penalty = 'l2', class_weight ='balanced', max_iter = 10)
modelDT = DecisionTreeClassifier(random_state = 0, max_depth = 3, min_samples_leaf = 3, min_samples_split = 2 )
modelXGB = XGBClassifier(max_depth = 2, gamma = 2, eta = 0.8, reg_alpha = 0.5, reg_lambda = 0.5)
modelSVM = svm.SVC(kernel ='linear', gamma='scale')
modelKNN = KNeighborsClassifier(n_neighbors=3)
modelGNB = GaussianNB()
scaler = StandardScaler()

normX = scaler.fit_transform(normX)

X_eval = scaler.fit_transform(X_fs_eval)

modelLR.fit(normX, normY)
Y_pred_LR = modelLR.predict_proba(X_fs_eval)

modelDT.fit(normX, normY)
Y_pred_DT = modelDT.predict_proba(X_fs_eval)

modelXGB.fit(normX, normY)
Y_pred_XGB = modelXGB.predict_proba(X_fs_eval)

modelSVM.fit(normX, normY)
Y_pred_SVM= modelSVM.predict(X_fs_eval)

modelKNN.fit(normX, normY)
Y_pred_KNN= modelKNN.predict_proba(X_fs_eval)

modelGNB.fit(normX, normY)
Y_pred_GNB= modelGNB.predict_proba(X_fs_eval)
from mlxtend.classifier import StackingClassifier
m = StackingClassifier(
    classifiers=[
        modelLR,
        modelDT,
        modelXGB
    ],
    use_probas=True,
    meta_classifier= modelLR
)

m.fit(normX, normY)

pred = m.predict_proba(X_fs_eval)[:,1]
pred
submission = pd.read_csv('/kaggle/input/dont-overfit-ii/sample_submission.csv')

submission['target'] = pred
submission.to_csv('sample_submission.csv', index = False)