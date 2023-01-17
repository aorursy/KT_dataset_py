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
import numpy as np

import pandas as pd

df_train=pd.read_csv('/kaggle/input/1056lab-fraud-detection-in-credit-card/train.csv',index_col=0)

df_test=pd.read_csv('/kaggle/input/1056lab-fraud-detection-in-credit-card/test.csv',index_col=0)
X=df_train.drop('Class',axis=1).values

y=df_train['Class'].values
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=0)

len(X_train),len(X_valid),len(y_train),len(y_valid)
from sklearn.ensemble import RandomForestClassifier

gscv=RandomForestClassifier(criterion='entropy',max_depth=5,n_estimators=102,random_state=0,n_jobs=-1)

gscv.fit(X,y)
gscv.score(X,y)
#gscv.best_params_
#gscv.cv_results_['mean_test_score']
#gscv.cv_results_['params']
Z=df_test.values

p=gscv.predict_proba(Z)
from sklearn.metrics import accuracy_score, confusion_matrix

y_pred = gscv.predict(X_valid)

tn, fp, fn, tp = confusion_matrix(y_valid, y_pred).ravel()

(tn, fp, fn, tp)
print('precision : %.4f'%(tp / (tp + fp))) #正と予測したデータのうち，実際に正であるものの割合

print('recall : %.4f'%(tp / (tp + fn))) #実際に正であるもののうち，正であると予測されたものの割合
from imblearn.under_sampling import RandomUnderSampler

# 正例の数を保存

positive_count_train = y_train.sum()

#print('positive count:{}'.format(positive_count_train))

# 正例が10％になるまで負例をダウンサンプリング

rus = RandomUnderSampler(ratio={0:positive_count_train*9, 1:positive_count_train}, random_state=0)

# 学習用データに反映

X_train_resampled, y_train_resampled = rus.fit_sample(X_train, y_train)
#gscv.fit(X_train_resampled, y_train_resampled)
y_pred = gscv.predict(X_valid)

print('Confusion matrix(test):\n{}'.format(confusion_matrix(y_valid, y_pred)))

print('Accuracy(test) : %.5f' %accuracy_score(y_valid, y_pred))
tn, fp, fn, tp = confusion_matrix(y_valid, y_pred).ravel()

print('precision : %.4f'%(tp / (tp + fp)))

print('recall : %.4f'%(tp / (tp + fn)))
from imblearn.over_sampling import RandomOverSampler



# 正例を10％まであげる

ros = RandomOverSampler(ratio = {0:X_train.shape[0], 1:X_train.shape[0]//9}, random_state = 0)



# 学習用データに反映

X_train_resampled, y_train_resampled = ros.fit_sample(X_train_resampled, y_train_resampled)
gscv.fit(X_train_resampled, y_train_resampled)
y_pred = gscv.predict(X_valid)

print('Confusion matrix(test):\n{}'.format(confusion_matrix(y_valid, y_pred)))

print('Accuracy(test) : %.5f' %accuracy_score(y_valid, y_pred))
tn, fp, fn, tp = confusion_matrix(y_valid, y_pred).ravel()

print('precision : %.4f'%(tp / (tp + fp)))

print('recall : %.4f'%(tp / (tp + fn)))
y_pred=gscv.predict_proba(Z)

df_submit=pd.read_csv('/kaggle/input/1056lab-fraud-detection-in-credit-card/sampleSubmission.csv',index_col=0)

df_submit['Class']=y_pred[:,1]

df_submit.to_csv('submission4.csv')
from imblearn.ensemble import BalancedBaggingClassifier

rf_model = RandomForestClassifier(n_jobs=-1,criterion='entropy',max_depth=5,n_estimators=102,random_state=0)

usbc = BalancedBaggingClassifier(base_estimator=rf_model, n_jobs=-1, n_estimators=10, ratio='not minority')

usbc.fit(X, y)
y_pred = usbc.predict(X_valid)

print('Confusion matrix(test):\n{}'.format(confusion_matrix(y_valid, y_pred)))

print('Accuracy(test) : %.5f' %accuracy_score(y_valid, y_pred))
tn, fp, fn, tp = confusion_matrix(y_valid, y_pred).ravel()

print('precision : %.4f'%(tp / (tp + fp)))

print('recall : %.4f'%(tp / (tp + fn)))
y_pred=usbc.predict_proba(Z)

df_submit=pd.read_csv('/kaggle/input/1056lab-fraud-detection-in-credit-card/sampleSubmission.csv',index_col=0)

df_submit['Class']=y_pred[:,1]

df_submit.to_csv('submission5.csv')