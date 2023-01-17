# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings 

warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_features=pd.read_csv("/kaggle/input/lish-moa/train_features.csv")

train_features.head()
train_features.shape
test_features=pd.read_csv("/kaggle/input/lish-moa/test_features.csv")

test_features.head()
#The dataset has categorical data, but the model needs something numerical. So let's apply one hot encoding here.

train_features.cp_type.unique()
train_features['cp_type'] =train_features.cp_type.map(lambda x:0 if x == 'trt_cp' else 1)

test_features['cp_type'] =test_features.cp_type.map(lambda x:0 if x == 'trt_cp' else 1)
train_features.cp_dose.unique()
train_features['cp_dose'] =train_features.cp_dose.map(lambda x:0 if x =='D1' else 1)

test_features['cp_dose'] =test_features.cp_dose.map(lambda x:0 if x =='D1' else 1)
replace_values = {24:1, 48:2, 72: 3}

train_features['cp_time'] =train_features['cp_time'].map(replace_values)

test_features['cp_time'] =test_features['cp_time'].map(replace_values)
#check cleaned table

train_features_new=train_features.iloc[:, 1:]

train_features_new.head()
test_features.shape
train_targets_scored=pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")

train_targets_scored.head()
train_targets_scored.shape
#remove sig_id for computing in a later stage

train_targets_new=train_targets_scored.iloc[:, 1:]
#calculate sum per column and find the top 100s. (The top 100 most significant factors?)

train_targets_scored = train_targets_scored.set_index('sig_id')

train_targets_scored.sum().nlargest(70)
train_targets_scored.sum().nsmallest(20)
train_targets_scored.sum(axis=1).nsmallest(200)
train_targets_scored.sum().nlargest(20).plot.bar(figsize=(18,15))
test_features.head()
train_features.shape
test_features.shape
train_targets_scored.shape
test_features_new=test_features.iloc[:, 1:]

test_features_new.head()
#normalize dataset ((x-min)/(max-min))



normalized_train_features=(train_features_new-train_features_new.min())/(train_features_new.max()-train_features_new.min())

normalized_test_features=(test_features_new-test_features_new.min())/(test_features_new.max()-test_features_new.min())
from sklearn.decomposition import PCA

#PCA will hold 80% of the variance and the number of components required to capture 80% variance will be used

pca = PCA(0.8)

pca.fit(normalized_train_features)



PCA(copy=True, iterated_power='auto', n_components=0.8, random_state=42,

  svd_solver='auto', tol=0.0, whiten=False)

print(pca.n_components_)

X = pca.transform(normalized_train_features.values)

X_test = pca.transform(normalized_test_features.values)

y = train_targets_new.values

# Extremely sloooooow!



from skmultilearn.model_selection import IterativeStratification

from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import LogisticRegression

import sklearn.metrics as metrics

from sklearn.metrics import precision_score,recall_score, confusion_matrix, classification_report, accuracy_score, f1_score, log_loss



#from sklearn.model_selection import train_test_split

y = train_targets_new.values



cnt=0

accu_losses=[]

k_fold = IterativeStratification(n_splits=5, order=1)

for train_index, val_index in k_fold.split(X, y):



    X_train, X_val = X[train_index], X[val_index]

    y_train, y_val = y[train_index], y[val_index]

    

    clf =OneVsRestClassifier(LogisticRegression(solver='lbfgs',penalty='l2'), n_jobs=-1)



    clf.fit(X_train, y_train)  

 

    # Making a prediction on the test set 



    pred_train =clf.predict_proba(X_train)

    pred_val = clf.predict_proba(X_val)

    pred_test = clf.predict_proba(X_test)

  

   

    # Evaluating the model

       

    # Evaluating the model

    loss = log_loss(np.ravel(y_val), np.ravel(pred_val))

    print ("Fold", cnt, "loss value is:",loss)

    accu_losses.append(loss)

    cnt+=1

print('mean of loss', np.mean(accu_losses))
print('mean of loss', np.mean(accu_losses))
samp = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')
samp.iloc[:,1:] = pred_test

samp.to_csv('submission.csv',index=False)