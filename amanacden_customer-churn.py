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

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('/kaggle/input/mba-dsa-practice/train.csv/train.csv')

test = pd.read_csv('/kaggle/input/mba-dsa-practice/Test.csv')

#creating dataframe for the required output

submission = pd.DataFrame()

submission['ID'] = test['ID']

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train, train['target'], test_size=0.20, random_state=42, shuffle=True)  

enclist = ['job', 'marital', 'education', 'connect','landline', 'smart', 'last_month','poutcome']

from sklearn.preprocessing import LabelEncoder
train.groupby(['target']).size()
train
for c in enclist:

    enc = LabelEncoder()

    X_train[c] = enc.fit_transform(X_train[c])

    X_val[c] = enc.transform(X_val[c])



y_train = X_train[['target']]

X_train = X_train.drop(['target','ID'], axis=1)    



from lightgbm import LGBMClassifier

import lightgbm as lgb

classifier = LGBMClassifier(learning_rate=0.1,max_depth=6,scale_pos_weight = 7.6,random_state = 0)

classifier.fit(X_train, y_train)

y_train.groupby(['target']).size()

y_val = X_val[['target']]

X_val = X_val.drop(['target','ID'], axis=1)
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 12,max_depth=6, criterion = 'entropy', class_weight = 'balanced_subsample',random_state = 0)

classifier.fit(X_train, y_train)

predictions = classifier.predict_proba(X_val)[:,1]
y1 = pd.DataFrame(predictions,columns=['prob'])

y1['hard'] = np.where(y1['prob']<0.64,0,1)

#print(y1.groupby(['hard'])['prob'].count())

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_val, y1['hard'])

#print(cm)

lgb_real_auc = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[0,1]+cm[1,0])

print("CM Accuracy : "+str(lgb_real_auc))

lgb_pred = pd.DataFrame(predictions,columns=['self_pay_status'])
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

y_predict = classifier.predict_proba(X_val)[:,1]

fpr, tpr, thresholds = roc_curve( y_val, y_predict)

auc_score = auc(fpr, tpr)

print("AUC : "+str(auc_score))

plt.figure(figsize=(16, 6))

lw = 2

plt.plot(fpr, tpr, lw=lw, label='ROC curve (area = %0.5f)' % auc_score)  

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend(loc="lower right")

plt.show()
for c in enclist:

    enc = LabelEncoder()

    train[c] = enc.fit_transform(train[c])

    test[c] = enc.transform(test[c])



y = train[['target']]

train = train.drop(['target','ID'], axis=1)

test = test.drop(['ID'], axis=1)    
from lightgbm import LGBMClassifier

import lightgbm as lgb

classifier = LGBMClassifier(learning_rate=0.1,max_depth=6,scale_pos_weight = 7.6,random_state = 0)

classifier.fit(train, y)
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 12,max_depth=6, criterion = 'entropy', class_weight = 'balanced_subsample',random_state = 0)

classifier.fit(train, y)
predictions = classifier.predict_proba(test)[:,1]
y2 = pd.DataFrame(predictions,columns=['prob'])

submission['target'] = np.where(y2['prob']<0.75,0,1)
submission
submission.groupby(['target']).size()
submission.to_csv('checkman.csv',index=False)