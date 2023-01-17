
from sklearn.ensemble import RandomForestClassifier
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
import os
print(os.listdir("../input"))

%%time
event_type=pd.read_csv("../input/event_type.csv",error_bad_lines=False)
train = pd.read_csv("../input/train.csv")
severity_type = pd.read_csv("../input/severity_type.csv")
log_feature = pd.read_csv("../input/log_feature.csv")
test = pd.read_csv("../input/test.csv")
resource_type = pd.read_csv("../input/resource_type.csv",error_bad_lines=False)
sample_submission = pd.read_csv("../input/sample_submission.csv")
print("test",test.shape)
print("train",train.shape)
print('test',test.head())
print('train',train.head(4))
print('sample_submission',sample_submission.head())
print('event_type',event_type.shape,event_type.head(2))
print('severity_type',severity_type.shape,severity_type.head(2))
print('log_feature',log_feature.shape,log_feature.head(2))
print('resource_type',resource_type.shape,resource_type.head(2))
train.head()
event_type['id']=pd.to_numeric(event_type['id'],errors='coerce')
#converting object datatype into numeric
event_type.dtypes
train_eve_mer=train.merge(event_type.drop_duplicates(subset=['id']),on='id')
train_merge2=train_eve_mer.merge(severity_type.drop_duplicates(subset=['id']),how='left',on='id')
train_merge3=pd.merge(train_merge2,log_feature.drop_duplicates(subset=['id']),how='left',on='id')
train_merge4=pd.merge(train_merge3,resource_type.drop_duplicates(subset=['id']),how='left',on='id')
train_merge4.head()
train_merge4.dtypes
train_merge4.isnull().sum()
cat_col=list(set(train_merge4.columns)-set(train_merge4._get_numeric_data().columns))
for i in range(len(cat_col)):
    train_merge4[cat_col[i]]=train_merge4[cat_col[i]].astype('category')

    
le=preprocessing.LabelEncoder()
for i in range(len(cat_col)):
    train_merge4[cat_col[i]]=le.fit_transform(train_merge4[cat_col[i]])
train_merge4.drop(['id'],axis=1,inplace=True)
target=train_merge4[['fault_severity']]
train_merge4.drop(['fault_severity'],axis=1,inplace=True)
rfc=RandomForestClassifier()
rfc.fit(train_merge4,target)

train_merge4.head()
test.head()
test.head()
test.shape
test_merge1=pd.merge(test,event_type.drop_duplicates(subset=['id']),how='left',on='id')
test_merge2=pd.merge(test_merge1,severity_type.drop_duplicates(subset=['id']),how='left',on='id')
test_merge3=pd.merge(test_merge2,log_feature.drop_duplicates(subset=['id']),how='left',on='id')
test_merge4=pd.merge(test_merge3,resource_type.drop_duplicates(subset=['id']),how='left',on='id')
test_merge4.head()
cat_col
test_merge4.dtypes

for i in range(len(cat_col)):
    test_merge4[cat_col[i]]=test_merge4[cat_col[i]].astype('category')

le2=preprocessing.LabelEncoder()
for i in range(len(cat_col)):
    test_merge4[cat_col[i]]=le2.fit_transform(test_merge4[cat_col[i]])
test_merge4.dtypes
test_merge4.drop(['id'],axis=1,inplace=True)
train_merge4.columns
test_merge4.columns
predict_test=rfc.predict_proba(test_merge4)
pred_df=pd.DataFrame(predict_test,columns=['predict_0', 'predict_1', 'predict_2'])
submission=pd.concat([test[['id']],pred_df],axis=1)
submission.to_csv('sub.csv',index=False,header=True)


