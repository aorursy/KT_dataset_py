# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Import all required libraries for machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_json("../input/train.json")
test = pd.read_json("../input/test.json")
train.head(2)
# test.head(2)
print(train.shape)
print(test.shape)
train_df = pd.DataFrame()
test_df = pd.DataFrame()
for frame in train.audio_embedding:
    tmp = pd.DataFrame(np.array(frame).reshape(1,-1))
    train_df = train_df.append(tmp)
for frame in test.audio_embedding:
    tmp = pd.DataFrame(np.array(frame).reshape(1,-1))
    test_df = test_df.append(tmp)
for k in train_df:
    train_df[k].fillna(train_df[k].mean(),inplace=True)
for k in train_df:
    test_df[k].fillna(test_df[k].mean(),inplace=True)
X_train,X_test,y_train,y_test = train_test_split(train_df,train.is_turkey,test_size=.2)
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
y_pred
accuracy_score(y_test,y_pred)*100
conf_mat = confusion_matrix(y_test,y_pred)
conf_mat
label = ["Turkey","Not_Turkey"]
sns.heatmap(conf_mat,annot=True,fmt="d",xticklabels=label,yticklabels=label)
logit_roc_auc = roc_auc_score(y_test, lr.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, lr.predict_proba(X_test)[:,1])
plt.figure(figsize=(8,4))
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('LogisticRegression_ROC')
plt.show()
y_pred2 = lr.predict(test_df.iloc[:-1,])
y_pred2
y_pred2[y_pred2==0].size
y_pred2[y_pred2==1].size
submit = pd.read_csv("../input/sample_submission.csv")
submit.is_turkey[1:] = y_pred2
submit.to_csv('submission.csv', index=False)
pd.read_csv("submission.csv")
