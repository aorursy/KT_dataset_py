

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

credit_data = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv",parse_dates=True)
credit_data.head()
import matplotlib.pyplot as plt
import seaborn as sb
class_count=pd.value_counts(credit_data['Class'],sort=True).sort_index()
print(class_count)
plt.xlabel('class label')
plt.ylabel('frequency')
class_count.plot(kind='bar')
plt.legend()

class_count.plot(kind='pie')
from sklearn.preprocessing import scale
print(credit_data['Amount'])
credit_data['norm_Amount']= scale(credit_data['Amount'])
print(credit_data['norm_Amount'])


credit_data.drop(['Amount','Time'],axis=1)
y= credit_data['Class']
x=credit_data.loc[:,credit_data.columns != 'Class']
#print(x)
from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
X= pd.concat([train_x,train_y],axis=1)

fraud_trans = X.loc[X['Class']==1,:]
print(len(fraud_trans))
true_trans = X.loc[X['Class']!=1,:]
print(len(true_trans))
from sklearn.utils import resample
#perform downsampling
downsample=resample(true_trans,replace=False,n_samples=len(fraud_trans),random_state=0)
#concat downsampled majority class and minority class
X_downsample = pd.concat([downsample,fraud_trans])
#print(X_downsample)
print(pd.value_counts(X_downsample['Class']==1))
X_train_downsample,X_valid_downsample,y_train_downsample,y_valid_downsample = train_test_split(X_downsample.loc[:,X_downsample.columns != 'Class'] ,X_downsample['Class'],test_size=0.3,random_state=1)

print(X_downsample.loc[:,X_downsample.columns != 'Class'].head())
print(X_downsample['Class'].head())
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
def error_metric(my_pred,model,y_valid):
    recall=recall_score(y_valid,my_pred)
    f1=f1_score(y_valid,my_pred)
    print("recall value on validation dataset",recall)
    print("f1 value on validation dataset",f1)
    
    #check the error metrics on our test data
    
    pred= model.predict(test_x)
    recall=recall_score(test_y,pred)
    f1=f1_score(test_y,pred)
    print("recall value on test dataset",recall)
    print("f1 value on test dataset",f1)
    
    print("confusion matrix for our test dataset"+"\n", confusion_matrix(test_y,pred))
def rfc(X_train,X_valid,y_train,y_valid):
    model = RandomForestClassifier(n_estimators=10,criterion="entropy",random_state=1)
    model.fit(X_train,y_train)
    my_pred=model.predict(X_valid)
    error_metric(my_pred,model,y_valid)
def xgb(X_train,X_valid,y_train,y_valid):
    model = XGBClassifier(n_estimator=500)
    model.fit(X_train,y_train)
    my_pred=model.predict(X_valid)
    error_metric(my_pred,model,y_valid)
rfc(X_train_downsample,X_valid_downsample,y_train_downsample,y_valid_downsample)
xgb(X_train_downsample,X_valid_downsample,y_train_downsample,y_valid_downsample)
X_train_upsample,X_valid_upsample,y_train_upsample,y_valid_upsample = train_test_split(train_x,train_y,test_size=0.3,random_state=1)
X= pd.concat([X_train_upsample,y_train_upsample],axis=1)
fraud_trans = X.loc[X['Class']==1,:]
print(len(fraud_trans))
true_trans = X.loc[X['Class']!=1,:]
print(len(true_trans))
from sklearn.utils import resample
#perform downsampling
upsample=resample(fraud_trans,replace=True,n_samples=len(true_trans),random_state=0)
#concat downsampled majority class and minority class
X_upsample = pd.concat([upsample,true_trans])
#print(X_downsample)
print(pd.value_counts(X_upsample['Class']==1))


y_train_upsample= X_upsample['Class']
X_train_upsample = X_upsample.loc[:,X_upsample.columns != 'Class']
rfc(X_train_upsample,X_valid_upsample,y_train_upsample,y_valid_upsample)
xgb(X_train_upsample,X_valid_upsample,y_train_upsample,y_valid_upsample)