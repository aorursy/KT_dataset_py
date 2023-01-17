# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import Counter

from sklearn.datasets import make_classification

from imblearn.over_sampling import RandomOverSampler

from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import train_test_split

from sklearn.utils import resample

from sklearn.model_selection import cross_val_score

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import svm

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestRegressor 

from sklearn.model_selection import RepeatedStratifiedKFold

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/minor-project-2020/train.csv")

df.head()
df.describe
df.info()
df.isnull().values.any()
df['target'].describe()
sns.set_style("whitegrid")

sns.set_context("poster")
plt.figure(figsize = (12, 6))

plt.hist(df['target'])

plt.title('Histogram of target values in the training set')

plt.xlabel('Count')

plt.ylabel('Target value')

plt.show()

plt.clf()
f = plt.figure(figsize=(240, 140))

plt.matshow(df.corr(), fignum=f.number)

plt.xticks(range(df.shape[1]), df.columns, fontsize=140)

plt.yticks(range(df.shape[1]), df.columns, fontsize=140)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=140)

plt.title('Correlation Matrix', fontsize=400);
rs = np.random.RandomState(0)

corr = df.corr()

corr.style.background_gradient(cmap='coolwarm')
X=df[df.columns[1:89]]

y=df['target']

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=48)
X_train
y_train
X_val
y_val
oversample = RandomOverSampler(random_state=98)

X_over, y_over = oversample.fit_resample(X_train, y_train)
X_train,X_val,y_train,y_val = train_test_split(X_over,y_over,test_size=0.85,random_state=48)
rus = RandomUnderSampler(random_state=48)

X_res, y_res = rus.fit_resample(X_train, y_train)
len(X_res)
Counter(y_res)
# cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)



crange = range(0,100)

grid = {"C": crange  }
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

# grid={"C":range(0,500)}

logreg=LogisticRegression(random_state=63)

logreg_cv=GridSearchCV(logreg,grid,cv=10,n_jobs=-1,verbose=1,scoring='roc_auc')

logreg_cv.fit(X_res,y_res)
print("Best: %f using %s" % (logreg_cv.best_score_, logreg_cv.best_params_))
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

model=LogisticRegression(solver='lbfgs', C=14,penalty='l2')
model.fit(X_res,y_res)
pred=model.predict(X_val)
pro_pred  =  model.predict_proba(X_val)[:,1]
from sklearn.metrics import roc_curve, auc

FPR, TPR, _ = roc_curve(y_val, pro_pred)

ROC_AUC = auc(FPR, TPR)

print (ROC_AUC)
plt.style.use('seaborn-pastel')



FPR, TPR, _ = roc_curve(y_val, pro_pred)

ROC_AUC = auc(FPR, TPR)

print (ROC_AUC)



plt.figure(figsize =[11,9])

plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 5)

plt.plot([0,1],[0,1], 'k--', linewidth = 5)

plt.xlim([0.0,1])

plt.ylim([0.0,1.1])

plt.xlabel('False Positive Rate', fontsize = 20)

plt.ylabel('True Positive Rate', fontsize = 20)

plt.title('ROC for Minor-1', fontsize= 30)

plt.show()
test_data = pd.read_csv("/kaggle/input/minor-project-2020/test.csv")

test_data.head()

X_test = test_data.drop('id',axis=1)

X_test.describe
pred=model.predict_proba(X_test)

pred = pred[:,1]

predicted=np.rint(pred)

test_data['target']=np.array(predicted)

out=test_data[['id','target']]

out=out.astype(int)

out.to_csv('submit_final.csv',index=False)
from sklearn.ensemble import RandomForestClassifier

clf_2 = RandomForestClassifier(bootstrap= True, max_depth= 80,max_features= 2,min_samples_leaf= 3,min_samples_split= 8,n_estimators= 100,random_state=63)

clf_2.fit(X_res,y_res)
pred_y_2 = clf_2.predict_proba(X_val)[:,1]
from sklearn.metrics import roc_curve, auc

FPR, TPR, _ = roc_curve(y_val, pred_y_2)

ROC_AUC = auc(FPR, TPR)

print (ROC_AUC)
clf_3 = RandomForestRegressor(n_estimators= 1000,min_samples_split= 5,min_samples_leaf=4,max_features='sqrt',max_depth= 80,bootstrap= False)

clf_3.fit(X_res,y_res)
pred_y_3 = clf_3.predict(X_val)
from sklearn.metrics import roc_curve, auc

FPR, TPR, _ = roc_curve(y_val, pred_y_3)

ROC_AUC = auc(FPR, TPR)

print (ROC_AUC)
from sklearn.ensemble import AdaBoostClassifier

from imblearn.ensemble import EasyEnsembleClassifier

pred_y_4 = EasyEnsembleClassifier(random_state=0,base_estimator=AdaBoostClassifier(),sampling_strategy='auto')

pred_y_4.fit(X_res,y_res)
pred_y_4 = clf_4.predict_proba(X_val)[:,1]
from sklearn.metrics import roc_curve, auc

FPR, TPR, _ = roc_curve(y_val, pred_y_4)

ROC_AUC = auc(FPR, TPR)

print (ROC_AUC)