import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression



from sklearn.model_selection import RepeatedStratifiedKFold, KFold



from sklearn import preprocessing



import os

print(os.listdir("../input"))





import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

sub = pd.read_csv("../input/sample_submission.csv")
train.head()
sub.head()
test.head()
train['target'].value_counts().plot.bar();
X = train.drop(['id', 'target'], axis=1)

y = train['target']
scaler = preprocessing.StandardScaler()

X = scaler.fit_transform(X)
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.2,random_state = 42)
print(X_train.shape,y_train.shape,X_val.shape)
k_fold = KFold(n_splits=10, shuffle=False, random_state=None)



s_fold = RepeatedStratifiedKFold(n_splits=10, n_repeats=20, random_state=42)



for trn, val in s_fold.split(X,y):

    x_train,y_train = X[trn],y[trn]

    x_val,y_val = X[val],y[val]

    

    model = LogisticRegression(C = 0.1, class_weight = 'balanced', penalty ='l1', solver='liblinear')

    model.fit(x_train,y_train)

    

    preds = model.predict(x_val)

scores = cross_val_score(model, X, y, cv=k_fold, n_jobs=1)
scores
from sklearn import metrics



fpr, tpr, thresholds = metrics.roc_curve(y_val, preds)

auc = metrics.auc(fpr, tpr)



plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc)

plt.legend()

plt.title('ROC curve')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.grid(True)

    
test2 = test.drop(['id'], axis =1)
sub_preds = model.predict(test2)
## 0.676 AUC

submission = pd.DataFrame({

    'id': test['id'],

    'target': sub_preds

})

submission.to_csv("submission.csv", index=False)
from imblearn.over_sampling import SMOTE,ADASYN



sm = ADASYN(random_state=42)



X_os, y_os = sm.fit_sample(X,y)
s_fold = RepeatedStratifiedKFold(n_splits=10, n_repeats=20, random_state=42)



for trn, val in s_fold.split(X_os,y_os):

    x_train,y_train = X_os[trn],y_os[trn]

    x_val,y_val = X_os[val],y_os[val]

    

    model = LogisticRegression(C = 0.1, class_weight = 'balanced', penalty ='l1', solver='liblinear')

    model.fit(x_train,y_train)

    

    preds = model.predict(x_val)

scores = cross_val_score(model, X, y, cv=s_fold, n_jobs=1)
fpr, tpr, thresholds = metrics.roc_curve(y_val, preds)

auc = metrics.auc(fpr, tpr)



plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc)

plt.legend()

plt.title('ROC curve')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.grid(True)
sub_preds = model.predict(test2)
submission = pd.DataFrame({

    'id': test['id'],

    'target': sub_preds

})

submission.to_csv("submission_os.csv", index=False)
from sklearn.ensemble import RandomForestClassifier

Model=RandomForestClassifier(max_depth=10)

Model.fit(x_train,y_train)

y_pred=Model.predict(x_val)
fpr, tpr, thresholds = metrics.roc_curve(y_val, y_pred)

auc = metrics.auc(fpr, tpr)



plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc)

plt.legend()

plt.title('ROC curve')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.grid(True)
sub_preds = Model.predict(test2)
submission = pd.DataFrame({

    'id': test['id'],

    'target': sub_preds

})

submission.to_csv("submission_RF.csv", index=False)