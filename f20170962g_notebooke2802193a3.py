import pandas as pd
from IPython.display import display, HTML
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)






train = pd.read_csv('../input/minor-project-2020/train.csv')
test = pd.read_csv('../input/minor-project-2020/test.csv')

label = train['target']
train2 = train.drop(columns = ['target', 'id'])
display(train2.head(10))
params = {}
params['learning_rate']= 0.0001
params['boosting_type']='gbdt'
params['objective']='binary'
params['metric']='roc'
params['sub_feature']=0.5
params['num_leaves']= 10
params['min_data']=50
params['max_depth']=10
params['num_boost_round'] = 1000
params['scale_pos_weight'] = len(train)/len(train[train.target == 1]) - 1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb





x_train, x_test, y_train, y_test = train_test_split(train2,label,test_size = 0.25, random_state = 0)
wt = []
for i in y_train:
    if i == '1':
        wt.append(10000)
    else:
        wt.append(1)

sc=StandardScaler()
#x_train = sc.fit_transform(x_train)
#x_test = sc.transform(x_test)
d_train = lgb.Dataset(x_train, label = y_train)
clf= lgb.train(params, d_train)

#convert into binary values
co = 0
cz = 0
for i in range(len(y_pred)):
    if (y_pred[i] >= 0.052):
        y_pred[i] = 1
        co += 1
    else:
        y_pred[i] =0
        cz += 1

print(co, cz)
len(y_pred)        
            
    
    
from sklearn.metrics import roc_auc_score
accuracy = roc_auc_score(y_pred, y_test)
print(accuracy)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#y_pred = clf.predict(test)

#convert into binary values

params = {"objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        'lambda_l1': 8.363957625616022e-05, 'lambda_l2': 5.1576139520241994e-05, 'num_leaves': 24, 'feature_fraction': 0.858469319864406, 'min_data': 46, 'max_depth': 94, 'num_boost_round': 86, 'learning_rate': 0.0007961842606486752, 'min_child_samples': 37, 'scale_pos_weight': 386.944723690289}
x_train, x_test, y_train, y_test = train_test_split(train2,label,test_size = 0.25, random_state = 0)
d_train = lgb.Dataset(x_train, label = y_train)
gbm = lgb.train(params, d_train)

display(test.head())
preds = gbm.predict(test.drop(columns = ['id', 'target']))
y_pred = preds


co = 0
cz = 0
#for i in range(len(y_pred)):
#    if (y_pred[i] >= 0.01):
#            y_pred[i] = 1
##            co += 1
#    else:
#            y_pred[i] = 0
#            cz += 1
            
        
        
print(co, cz, len(y_pred), len(test))
df = pd.DataFrame(data = y_pred)
df['target'] = df[0]
df.drop(columns = [0], inplace = True)
display(df.head(10))
test['target'] = df.target
print(test.head(10))
print(len(test))
test[['id', 'target']].to_csv('./output.csv' ,index = False)