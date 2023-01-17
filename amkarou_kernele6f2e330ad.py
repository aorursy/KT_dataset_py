# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



train= pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#
event_type = pd.read_csv('../input/event_type.csv',error_bad_lines=False)
log_feature= pd.read_csv('../input/log_feature.csv',error_bad_lines=False)
resource_type = pd.read_csv('../input/resource_type.csv',error_bad_lines=False)
severity_type = pd.read_csv('../input/severity_type.csv',error_bad_lines=False)
test.shape
print(train.head(5))
print(test.head(5))
train['source'] = 'train'
test['source'] = 'test'
data = pd.concat([train,test], ignore_index=True)
data.tail()
#Our target/output feature
data['fault_severity'].value_counts()
#Autres fichiers excel?
print(event_type.shape)
print (log_feature.shape)
print (resource_type.shape)
print(severity_type.shape)
print(event_type.head(5))
print(log_feature.head(5))
print(resource_type.head(5))
print(severity_type.head(5))

#Manipuler Fichier event_type
event_type.describe()
#drop the last line which has abstracts variables
event_type.drop(event_type.index[31170], inplace=True)
event_type.tail()
event_type['id']=event_type['id'].astype('int')
event_type = event_type.merge(data, on='id')
event_type.head()
event_type_unq = pd.DataFrame(event_type['event_type'].value_counts())
event_type_unq.head()
#Determine % of training samples:
event_type_unq['PercTrain'] = event_type.pivot_table(values='source',index='event_type',aggfunc=lambda x: sum(x=='train')/float(len(x)))
event_type_unq.head()
#Determine the mode of each:
df_event=event_type.loc[event_type['source']=='train']
print(df_event)
event_type_unq['Mode_Severity'] = df_event.pivot_table(values='fault_severity',index='event_type', aggfunc=lambda x: x.mode().iat[0])

event_type_unq.iloc[-15:]
event_type_unq['preprocess'] = event_type_unq.index.values
unchange = 33
event_type_unq['preprocess'].iloc[unchange:] = event_type_unq['Mode_Severity'].iloc[unchange:].apply(lambda x: 'Remove' if pd.isnull(x) else 'event_type others_%d'%int(x))

print(event_type_unq['preprocess'].value_counts())
print(event_type_unq)
event_type = event_type.merge(event_type_unq[['preprocess']], left_on='event_type',right_index=True)
print(event_type.head())
event_type['preprocess'].value_counts()
event_type_merge = event_type.pivot_table(values='event_type',index='id',columns='preprocess',aggfunc=lambda x: len(x), fill_value=0)
event_type_merge.shape
data = data.merge(event_type_merge, left_on='id', right_index=True)
data.head()
#On passe mtn au fichier log features:
log_feature['log_feature'].value_counts().head()
log_feature = log_feature.merge(data[['id','fault_severity','source']], on='id')
log_feature.head()
log_feature_unq = pd.DataFrame(log_feature['log_feature'].value_counts())
log_feature_unq['PercTrain'] = log_feature.pivot_table(values='source',index='log_feature',aggfunc=lambda x: sum(x=='train')/float(len(x)))
log_feature_unq.head()
df_logfeatures=log_feature.loc[log_feature['source']=='train']
log_feature_unq['Mode_Severity'] = df_logfeatures.pivot_table(values='fault_severity',index='log_feature', aggfunc=lambda x: x.mode().iat[0])
len(log_feature_unq)
log_feature_unq.ix[100:130]
log_feature_unq['preprocess'] = log_feature_unq.index.values
log_feature_unq['preprocess'].loc[log_feature_unq['PercTrain']==1] = np.nan
top_unchange = 128
log_feature_unq['preprocess'].iloc[top_unchange:] = log_feature_unq['Mode_Severity'].iloc[top_unchange:].apply(lambda x: 'Remove' if pd.isnull(x) else 'feature others_%d'%int(x))
print(log_feature_unq['preprocess'].value_counts())
print(log_feature_unq)

log_feature = log_feature.merge(log_feature_unq[['preprocess']], left_on='log_feature',right_index=True)
print(event_type.head())
log_feature['preprocess'].value_counts()
log_feature_merge = log_feature.pivot_table(values='volume',index='id',columns='preprocess',aggfunc=np.sum, fill_value=0)
log_feature_merge.shape
data = data.merge(log_feature_merge, left_on='id', right_index=True)
data.head()
#On passe au fichier Resource Type:
resource_type['resource_type'].value_counts()
resource_type = resource_type.merge(data[['id','fault_severity','source']], on='id')
resource_type_unq = pd.DataFrame(resource_type['resource_type'].value_counts())
resource_type_unq.head()
resource_type_unq['PercTrain'] = resource_type.pivot_table(values='source',index='resource_type',aggfunc=lambda x: sum(x=='train')/float(len(x)))
resource_type_unq.head()
#Determine the mode of each:
df_resource=resource_type.loc[resource_type['source']=='train']
resource_type_unq['Mode_Severity'] = df_resource.pivot_table(values='fault_severity',index='resource_type', aggfunc=lambda x: x.mode().iat[0])
resource_type_unq

resource_type_merge = resource_type.pivot_table(values='source',index='id',columns='resource_type',aggfunc=lambda x: len(x), fill_value=0)
data = data.merge(resource_type_merge, left_on='id', right_index=True)
data.head()
#Enfin on a le fichier excel Severity Type:
severity_type['severity_type'].value_counts()
severity_type = severity_type.merge(data[['id','fault_severity','source']], on='id')
severity_type.head()


severity_type_unq = pd.DataFrame(severity_type['severity_type'].value_counts())
severity_type_unq.head()
severity_type_unq['PercTrain'] = severity_type.pivot_table(values='source',index='severity_type',aggfunc=lambda x: sum(x=='train')/float(len(x)))
severity_type_unq.head()
#Determine the mode of each:
severity_type_unq['Mode_Severity'] = severity_type.loc[severity_type['source']=='train'].pivot_table(values='fault_severity',index='severity_type', aggfunc=lambda x: x.mode().iat[0])
severity_type_unq
severity_type_merge = severity_type.pivot_table(values='source',index='id',columns='severity_type',aggfunc=lambda x: len(x), fill_value=0)
event_type_merge.head()
data = data.merge(severity_type_merge, left_on='id', right_index=True)
data.head(), data.shape
#On va ajouter une variable 'location_count' 
location_count = data['location'].value_counts()
data['location_count'] = data['location'].apply(lambda x: location_count[x])

#Feature Count:
featvar = [x for x in data.columns if 'feature ' in x]
data['feature_count'] = data[featvar].apply(np.sum, axis=1)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['location'] = le.fit_transform(data['location'])
[x for x in data.columns if 'Remove' in x]
data.drop(['Remove_x','Remove_y'],axis=1,inplace=True)

training_set = data.loc[data['source']=='train']
test_set = data.loc[data['source']=='test']
training_set.drop('source',axis=1,inplace=True)
test_set.drop(['source','fault_severity'],axis=1,inplace=True)
X = training_set.iloc[:,1:]
Y = training_set.iloc[:,0]
test_set.shape
test_set.tail()
# diviser les données en train and test sets
from sklearn.model_selection import train_test_split
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
import xgboost as xgb
dtrain = xgb.DMatrix(data=X_train,label=y_train)
params = {'learning_rate':0.1,
        'n_estimators':500,
        'max_depth':10,
        'min_child_weight':30,
        'gamma':0,
        'subsample':0.8,
        'colsample_bytree':0.8,
        'objective': 'multi:softprob',
        'num_class':3,
        'eval_metric':['mlogloss'],
        'missing':-1,
        'nthread':4}

num_round = 10000
bst = xgb.train(params, dtrain, num_round)
n_classes = 3
dtest = xgb.DMatrix(data=X_test,label=y_test)
preds = bst.predict(dtest).reshape(-1, n_classes)
import numpy as np
best_preds = np.asarray([np.argmax(line) for line in preds])
#Calcul de précision
from sklearn.metrics import precision_score
print(precision_score(y_test, best_preds, average='macro'))

test_set.size
#Prediction:
testpred = bst.predict(xgb.DMatrix(test_set)).reshape(-1, n_classes)
submission = pd.read_csv('../input/sample_submission.csv')
cols = ['predict_0', 'predict_1', 'predict_2']

submission[cols] = testpred
submission.tail()
submission.to_csv('submissionv2.csv', index=False, header=True, mode='a')