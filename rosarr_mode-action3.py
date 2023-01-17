# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os

print(os.listdir("../input"))

from joblib import dump, load
import collections
import random
import time

from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
file_path = "../input/lish-moa/"

#data to fit and evaluate the model
train_data = pd.read_csv(file_path+'train_features.csv')
df_train=train_data.copy()
target_data  = pd.read_csv(file_path+'train_targets_scored.csv')
df_target=target_data.copy()

#collect data for test submission
test_data = pd.read_csv(file_path+'test_features.csv')
df_test=test_data.copy()
sample = pd.read_csv(file_path+'sample_submission.csv')

print(train_data.shape)
train_data.head()


df_train.loc[df_train['cp_type']=='ctl_vehicle','cp_time']= 0
df_train.loc[df_train['cp_type']=='ctl_vehicle','cp_dose']= 0
df_test.loc[df_train['cp_type']=='ctl_vehicle','cp_time']= 0
df_test.loc[df_train['cp_type']=='ctl_vehicle','cp_dose']= 0

df_train.loc[:, 'cp_dose']=df_train.loc[:, 'cp_dose'].map({'D1': 1, 'D2': 2,0:0})
df_train.loc[:, 'cp_type']=df_train.loc[:, 'cp_type'].map({'trt_cp': 1, 'ctl_vehicle': 0})
df_train.loc[:,'cp_time']=df_train.loc[:,'cp_time'].map({0:0,24:1,48:2,72:3})

df_test.loc[:,'cp_dose']=df_test.loc[:,'cp_dose'].map({'D1': 1, 'D2': 2,0:0})
df_test.loc[:, 'cp_type']=df_test.loc[:,'cp_type'].map({'trt_cp': 1, 'ctl_vehicle': 0})
df_test.loc[:,'cp_time']=df_test.loc[:,'cp_time'].map({0:0,24:1,48:2,72:3})
df_train.head()
df_test.head()
#features cp_time, cp_dose and g-, c-
features=list(df_train.columns)[2:]
targets=list(target_data.columns)[1:]
for target in targets:
    count=collections.Counter(target_data.loc[:,target])
    if count[1]<5:
        print (target, count)

#remove two few positives (erbb2_inhibitor,atp-sensitive_potassium_channel_antagonist).

targets.remove('erbb2_inhibitor')
targets.remove('atp-sensitive_potassium_channel_antagonist')

def PCA_model1(X, components_number):
    #Apply PCA and return the sum explained variance 
    c=components_number
    pca = PCA(n_components=c).fit(X)
    sum_vexp_var=sum(pca.explained_variance_ratio_)
    return sum_vexp_var

def PCA_model2(X, components_number):
    #Apply PCA and return pca fit estimator and the reduce data in form of dataframe 
    c=components_number
    pca = PCA(n_components=c).fit(X)    
    reduced_data=pca.transform(X)
    df_reduced_data=pd.DataFrame((reduced_data),columns=range(0,c))    
    return pca, df_reduced_data
    
X=df_train[features]
y=np.array(df_target[targets])
X_sample=df_test[features]
scaler=StandardScaler().fit(X)
X=scaler.transform(X)
X_sample=scaler.transform(X_sample)
#Try a range of components
components_number=[2,30,40,50,100,200]
for c in components_number:
    print ('PCA with {} components:'.format(c))    
    sum_vexp_var= PCA_model1(X, c)
    print( 'sum explained_variance_ratio for {}:'.format(c),sum_vexp_var)
    print('')   
#Calculate reduced data with 40 components
pca,df_reduced_data= PCA_model2(X,40)
X=df_reduced_data
y=np.array(df_target[targets])
X_sample=pca.transform(X_sample)
start=time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
clf=SVC(probability=True)
clf_fit=OneVsRestClassifier(clf).fit(X_train, y_train)
y_pred=clf_fit.predict(X_test)
y_pred_prob=clf_fit.predict_proba(X_test)
sample_pred_prob=clf_fit.predict_proba(X_sample)
end=time.time()
print('time:',end-start)
print("accuracy: %0.2f " % (accuracy_score(y_test,y_pred)))
print('')
print("log_los: %0.3f " % (log_loss(y_test.ravel(),y_pred_prob.ravel())))
#save model
filename='model3_pca_svc4.joblib'
dump(clf_fit, filename)
#Introduce results of predictions in this dataframe, put everything to 0.
#this way removed targets would be put to 0
sample_null=sample.loc[:,list(target_data.columns)[1:]].transform(lambda x:x*0)
print(sample_null.shape)
sample_null.head()
sample_null.loc[:,targets] = sample_pred_prob

sample_null.head()
#check if data with cp_vehicle are predicted to be 0
sample_vehicle=sample_null.loc[df_test['cp_type']==0]
#print(list(sample_null.loc[df_test['cp_type']==0].max()))
sample_vehicle.describe()
#change controls to 0. Some targets were not predicted to be 0
sample_null.loc[df_test['cp_type'] == 0, sample_null.columns] = 0
sample_null.head()
sample.iloc[:,1:]=sample_null
sample.to_csv('submission.csv', index=False)