import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input/train_amex"))

# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train_amex/train.csv',)
hist=pd.read_csv('../input/train_amex/historical_user_logs.csv')
test=pd.read_csv('../input/test_lnmuiyp/test.csv')
HistJoined_train = train.join(hist, on='user_id',how='left',rsuffix='_hist')
HistJoined_test = test.join(hist, on='user_id',how='left',rsuffix='_hist')
#hist.head()
#train.head()

del train
del test

HistJoined_train.head()
print('train.shape: %s\n test.shape: %s'%(HistJoined_train.shape,HistJoined_test.shape))
#HistJoined['product_category_2'].value_counts()
HistJoined_train.skew()
#train['is_click'].value_counts()
#train.info()
#train['DateTime']
#train.isnull().sum()
#HistJoined_train.isnull().mean()

HistJoined_train.info()







HistJoined_train.columns
colsToEncode=['product','gender','product_hist','action']
#colsToEncode
for cols in colsToEncode:
    HistJoined_train[cols] = pd.factorize(HistJoined_train[cols])[0]
    HistJoined_test[cols] = pd.factorize(HistJoined_test[cols])[0]    
HistJoined_train['is_click'].skew()
HistJoined_train.isnull().mean()
NullCols=['product_category_2','user_group_id','age_level','user_depth','city_development_index']
HistJoined_train[NullCols].head()

#HistJoined_train.dropna(subset=['age_level','user_group_id','user_depth'],axis=0,inplace=True)
#HistJoined_test.dropna(subset=['age_level','user_group_id','user_depth'],axis=0,inplace=True)


#values = {'city_development_index':-1,'product_category_2':-1}
HistJoined_train.fillna(method='ffill',inplace=True)
HistJoined_test.fillna(method='ffill',inplace=True)

values = {'city_development_index':-1,'product_category_2':-1}
HistJoined_train.fillna(value=values,inplace=True)
HistJoined_test.fillna(value=values,inplace=True)
HistJoined_train.isnull().sum()
HistJoined_test.isnull().sum()

import datetime
HistJoined_train['DateTime'] = [datetime.datetime.strptime(d,'%Y-%m-%d %H:%M') for d in HistJoined_train['DateTime']]
HistJoined_train['DateTime_hist'] = [datetime.datetime.strptime(d,'%Y-%m-%d %H:%M') for d in HistJoined_train['DateTime_hist']]

HistJoined_test['DateTime'] = [datetime.datetime.strptime(d,'%Y-%m-%d %H:%M') for d in HistJoined_test['DateTime']]
HistJoined_test['DateTime_hist'] = [datetime.datetime.strptime(d,'%Y-%m-%d %H:%M') for d in HistJoined_test['DateTime_hist']]

HistJoined_train['hour_of_day']=[d.hour for d in HistJoined_train['DateTime']]
HistJoined_train['hour_of_day_hist']=[d.hour for d in HistJoined_train['DateTime_hist']]

HistJoined_test['hour_of_day']=[d.hour for d in HistJoined_test['DateTime']]
HistJoined_test['hour_of_day_hist']=[d.hour for d in HistJoined_test['DateTime_hist']]
#Parts_of_day
#Morning,afternoon,evening,night,late-night.

HistJoined_train['Parts_of_day'] = pd.cut(x=HistJoined_train['hour_of_day'],bins=5,labels=[0,1,2,3,4]).astype('int64')
HistJoined_test['Parts_of_day'] = pd.cut(x=HistJoined_test['hour_of_day'],bins=5,labels=[0,1,2,3,4]).astype('int64')

HistJoined_train['Parts_of_day_hist'] = pd.cut(x=HistJoined_train['hour_of_day_hist'],bins=5,labels=[0,1,2,3,4]).astype('int64')
HistJoined_test['Parts_of_day_hist'] = pd.cut(x=HistJoined_test['hour_of_day_hist'],bins=5,labels=[0,1,2,3,4]).astype('int64')

#HistJoined_train.groupby('Parts_of_day')['hour_of_day'].value_counts()
#HistJoined_train.groupby('user_id')['webpage_id'].value_counts()

HistJoined_train['No_of_Webpage_Hits'] = HistJoined_train.groupby('user_id',)['webpage_id'].transform('count')
HistJoined_test['No_of_Webpage_Hits'] = HistJoined_test.groupby('user_id',)['webpage_id'].transform('count')

#HistJoined_train['No_of_Webpage_Hits'].value_counts()
HistJoined_train['No_of_Product_Hits_hist'] = HistJoined_train.groupby('user_id_hist',)['product_hist'].transform('count')
HistJoined_test['No_of_Product_Hits_hist'] = HistJoined_test.groupby('user_id_hist',)['product_hist'].transform('count')

#HistJoined_train.groupby(['user_id'])['DateTime'].diff()
HistJoined_train.info()
#Function to optimize the memory by downgrading the datatype to optimal length
def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)
dtype_list=[]
for col in HistJoined_train.columns:
    dtype_list.append(HistJoined_train[col].dtypes)
dtype_list=list(set(dtype_list))
print("Total Datatypes present: %s "%dtype_list)
# Analysing for Train dataset
for dtype in dtype_list:
    
    if 'int' in str(dtype):
        print("Analyse %s"%str(dtype))
        df_int=HistJoined_train.select_dtypes(include=[str(dtype)])
        converted_int = df_int.apply(pd.to_numeric,downcast='unsigned')
    
        print(mem_usage(df_int))
        print(mem_usage(converted_int))
        
    elif 'float' in str(dtype):
        print("Analyse %s"%str(dtype))
        df_float = HistJoined_train.select_dtypes(include=[str(dtype)])
        converted_float = df_float.apply(pd.to_numeric,downcast='float')
        
        print(mem_usage(df_float))
        print(mem_usage(converted_float))
print("Memory Usage of Original dataset: %s"%mem_usage(HistJoined_train))
HistJoined_train[converted_int.columns] = converted_int
HistJoined_train[converted_float.columns] = converted_float
print("Memory Usage of Optimized dataset: %s"%mem_usage(HistJoined_train))
# Analysing for Test dataset
for dtype in dtype_list:
    
    if 'int' in str(dtype):
        print("Analyse %s"%str(dtype))
        df_int=HistJoined_test.select_dtypes(include=[str(dtype)])
        converted_int = df_int.apply(pd.to_numeric,downcast='unsigned')
    
        print(mem_usage(df_int))
        print(mem_usage(converted_int))
        
    elif 'float' in str(dtype):
        print("Analyse %s"%str(dtype))
        df_float = HistJoined_test.select_dtypes(include=[str(dtype)])
        converted_float = df_float.apply(pd.to_numeric,downcast='float')
        
        print(mem_usage(df_float))
        print(mem_usage(converted_float))
print("Memory Usage of Original dataset: %s"%mem_usage(HistJoined_test))
HistJoined_test[converted_int.columns] = converted_int
HistJoined_test[converted_float.columns] = converted_float

print("Memory Usage of Optimized dataset: %s"%mem_usage(HistJoined_test))
session_id_test = HistJoined_test.session_id
HistJoined_train.drop(['DateTime','DateTime_hist','session_id','user_id','product','campaign_id','webpage_id'],axis=1,inplace=True)
HistJoined_test.drop(['DateTime','DateTime_hist','session_id','user_id','product','campaign_id','webpage_id'],axis=1,inplace=True)

#HistJoined_train.drop(['DateTime','DateTime_hist'],axis=1,inplace=True)
#HistJoined_test.drop(['DateTime','DateTime_hist'],axis=1,inplace=True)

HistJoined_train.head()
type(session_id_test)
fig,ax = plt.subplots(figsize=(15,15))
ax = sns.heatmap(data=HistJoined_train.corr(method='pearson'),annot=True,cmap='YlGnBu',linewidths=0.05)

HistJoined_train.isnull().sum()
from sklearn.model_selection import train_test_split
X=HistJoined_train.drop(['is_click'],axis=1)
y=HistJoined_train['is_click']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22,shuffle=True,stratify=y)

print('X_train.shape %s, X_test.shape %s\ny_train.shape %s, y_test.shape %s'%(X_train.shape,X_test.shape,y_train.shape,y_test.shape))
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit only to the training data
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.fit_transform(X_test)
from sklearn import model_selection
#performance metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
#Hyperparameter tuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
#Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(class_weight='balanced',penalty='l2',solver='saga',max_iter=100,C=1)

LR.fit(scaled_X_train,y_train)
y_pred_lr = LR.predict(scaled_X_test)
print('Accuracy: %.4f' %accuracy_score(y_pred=y_pred_lr,y_true=y_test))
print('Confusion Matrix: \n%s'%confusion_matrix(y_pred=y_pred_lr,y_true=y_test))
print('Classification report: \n %s'%classification_report(y_pred=y_pred_lr,y_true=y_test))
print('AUC score: %.5f'%roc_auc_score(y_test,y_pred_lr))


from imblearn.over_sampling import SMOTE

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(scaled_X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))

### Oversampled data
rfc_over = RandomForestClassifier(n_estimators=50,class_weight='balanced',criterion='entropy',max_depth=500,)

rfc_over.fit(X_train_res,y_train_res)
y_pred_rf = rfc_over.predict(scaled_X_test)

print('Accuracy: %.4f' %accuracy_score(y_pred=y_pred_rf,y_true=y_test))
print('Confusion Matrix: \n%s'%confusion_matrix(y_pred=y_pred_rf,y_true=y_test))
print('Classification report: \n %s'%classification_report(y_pred=y_pred_rf,y_true=y_test))
print('AUC score: %.5f'%roc_auc_score(y_test,y_pred_rf))
'''class_wt = dict({0:1.39,1:80})
rfc = RandomForestClassifier(max_features='log2',n_estimators=10,class_weight='balanced',criterion='entropy',max_depth=500)

rfc.fit(scaled_X_train,y_train)
y_pred_rf = rfc.predict(scaled_X_test)


print('Accuracy: %.4f' %accuracy_score(y_pred=y_pred_rf,y_true=y_test))
print('Confusion Matrix: \n%s'%confusion_matrix(y_pred=y_pred_rf,y_true=y_test))
print('Classification report: \n %s'%classification_report(y_pred=y_pred_rf,y_true=y_test))
print('AUC score: %.5f'%roc_auc_score(y_test,y_pred_rf))
print(rfc.feature_importances_)
'''
#Predicting the test dataset values
#Scaling the test dataset
scaled_test = scaler.fit_transform(HistJoined_test)
final_pred_rfc = rfc_over.predict(scaled_test)
submission_rf = pd.DataFrame(data=session_id_test)
submission_rf['is_click'] = pd.DataFrame({'is_click':final_pred_rfc})
submission_rf.head()
submission_rf.to_csv('submission_rfc_over.csv',index=False)
#Predicting the test dataset values
#Scaling the test dataset
final_pred_lr = LR.predict(scaled_test)
submission_lr = pd.DataFrame(data=session_id_test)
submission_lr['is_click'] = pd.DataFrame({'is_click':final_pred_lr})
submission_lr.head()

submission_lr.to_csv('submission_lr.csv',index=False)