## This is a study of the data which found in Kaggle, Jan flight data in 2019 and 2020
## Random Forest model used to try to predict the target flight will be delayed or not
## By using the random forest also can find out the feature importance.

## Finally, the random forest can achieve in 0.94 precision and 0.50 recall test data, 
## F1 around 0.66
## Some importance found, for example, the previous flight depature time is most
## important feature, and the current fligh time block is also important feature.


import pandas as pd
import os, re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime
import copy

rootpath = '/kaggle/input/flight-delay-prediction'
datapath = os.path.join(rootpath)
re_get_year = re.compile('Jan_(\d{4})')
datafilelist = [ (re_get_year.findall(i)[0], i , pd.read_csv(os.path.join(datapath, i)).loc[:, :"DISTANCE"]  ) for i in os.listdir(datapath)]
def is_delay(df):
    if ((df['ARR_DEL15'] == 1) or (df['DEP_DEL15'] == 1)) and ((df['CANCELLED']==0) & (df['DIVERTED']==0)):
        return 1
    else:
        return 0
def get_statistic(df):
    cancel_num = df.loc[df['CANCELLED']==1].shape[0]
    divert_num = df.loc[df['DIVERTED']==1].shape[0]
    total_num = df.loc[:, :].shape[0]
    norm_num = total_num - cancel_num - divert_num
    return { 
             'cancelled' : [cancel_num , cancel_num*1./total_num], 
             'diverted' : [divert_num , divert_num*1./total_num],
             'norm_num' : [norm_num , norm_num*1./total_num]
            }
def generate_value(unique_df, columns_name=None): 
    col_name = 'default' if columns_name is None else columns_name
    tmp = {
           'orig_v' : [], 
           'map_v' : []
      }
    for k,i in enumerate(unique_df):
        tmp['orig_v'].append(i)
        tmp['map_v'].append(k)
    df_tmp = pd.DataFrame(tmp)
    df_tmp.columns = [col_name , col_name +'_map'] 
    return df_tmp


def precision_recall(model, X, y_true):
    test = pd.DataFrame({ 'predict':list(clf.predict(X)) , 'true' : list(y_true) } )
    tp = test[ ((test['predict'] == 1) | (test['predict'] == 2)) & ((test['true']==1) | (test['true']==2) ) ].shape[0]
    fp = test[ ((test['predict'] == 1) | (test['predict'] == 2)) & (test['true']==0)].shape[0]
    tn = test[ (test['predict'] == 0) & (test['true']==0)].shape[0]
    fn = test[ (test['predict'] == 0) & ((test['true']==1) | (test['true']==2) )].shape[0]
    print(tp, fp, tn , fn)
    precision = tp*1. / (tp+fp)
    recall = tp*1. / (tp+fn)
    acc = (tp+tn)*1. /(tp+tn+fn+fp)
    f1 = 2.*(precision * recall)/(precision + recall)
    return tp, fp, tn, fn , precision, recall, f1,acc

def metric_calc(model, X_train, Y_train, X_test, Y_test): 
    _,_,_,_, p_train, r_train, f_train, a_train = precision_recall(clf, X_train, Y_train)
    _,_,_,_, p_test, r_test, f_test, a_test = precision_recall(clf, X_test, Y_test)
    importance = list(model.feature_importances_)
    importance = list(zip(list(X_train.columns), importance))
    return p_train, r_train, f_train, a_train ,  p_test, r_test, f_test, a_test, importance
datafilelist[0][2]['is_delay']  = datafilelist[0][2].apply(lambda x: is_delay(x), axis=1)
datafilelist[1][2]['is_delay']  = datafilelist[1][2].apply(lambda x: is_delay(x), axis=1)
datafilelist[0][2]['Year'] = int(datafilelist[0][0])
datafilelist[1][2]['Year'] = int(datafilelist[1][0])
data = pd.concat( [datafilelist[0][2] , datafilelist[1][2]])


op_map = generate_value(data['OP_CARRIER'].unique(), columns_name='op_carrier')
tail_num_map = generate_value(data['TAIL_NUM'].unique(), columns_name='tail_num')
origin_map = generate_value(data['ORIGIN'].unique(), columns_name='origin')
dest_map = generate_value(data['DEST'].unique(), columns_name='dest')
dep_time_blk_map = generate_value(data['DEP_TIME_BLK'].unique(), columns_name='dep_time_blk')



data_TMP = data.loc[: , [ 'Year' , 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'OP_CARRIER', 'TAIL_NUM', 'ORIGIN', 'DEST', 'DEP_TIME_BLK', 'DISTANCE','DEP_TIME', 'ARR_TIME', 'is_delay', 'DEP_DEL15', 'ARR_DEL15']]  
data_TMP = data_TMP.merge(op_map, how='left', left_on='OP_CARRIER' , right_on='op_carrier' ) 
data_TMP = data_TMP.merge(tail_num_map, how='left' , left_on='TAIL_NUM' , right_on = 'tail_num')
data_TMP = data_TMP.merge(origin_map, how='left' , left_on='ORIGIN' , right_on = 'origin')
data_TMP = data_TMP.merge(dest_map, how='left' , left_on='DEST' , right_on = 'dest')
data_TMP = data_TMP.merge(dep_time_blk_map, how='left' , left_on='DEP_TIME_BLK' , right_on = 'dep_time_blk')



dep_delay_predict_vector = ['Year' , 'DAY_OF_MONTH' , 'DAY_OF_WEEK' , 'op_carrier_map' , 'origin_map' , 'dest_map' , 'dep_time_blk_map' , 'DEP_TIME', 'DISTANCE', 'DEP_DEL15' ] 
dep_delay_predict_vector_rename = ['year' , 'daymonth' , 'dayweek' , 'opcarrier', 'origin' , 'dest' , 'deptimeblk', 'dep_time' ,'distance' , 'depdelay' ] 
delay_predict_data = data_TMP.loc[ ~data_TMP['DEP_DEL15'].isnull(), dep_delay_predict_vector]
delay_predict_data.columns = dep_delay_predict_vector_rename

delay_predict_data = delay_predict_data.sort_values( ['origin', 'year', 'daymonth', 'dep_time'] ).reset_index().drop('index', axis=1)
delay_predict_data2 = copy.deepcopy(delay_predict_data)
delay_predict_data2=delay_predict_data2.groupby("origin").shift(1)
delay_predict_data2.columns = ['p_' + i for i in delay_predict_data2.columns] 
delay_predict_data2 = delay_predict_data2.merge(delay_predict_data, how='left' , left_index=True, right_index=True)

X =  delay_predict_data2.loc[:, :'distance']
Y = delay_predict_data2.loc[:, 'depdelay'] 

# set all nul value into 0
X.loc[X.p_year.isnull(), 'p_year'] = 0
X.loc[X.p_daymonth.isnull(), 'p_daymonth'] = 0 
X.loc[X.p_dayweek.isnull(), 'p_dayweek'] = 0
X.loc[X.p_opcarrier.isnull(), 'p_opcarrier'] = 0
X.loc[X.p_dest.isnull(), 'p_dest'] = 0
X.loc[X.p_deptimeblk.isnull(), 'p_deptimeblk'] = 0
X.loc[X.p_dep_time.isnull(), 'p_dep_time'] = 0
X.loc[X.p_distance.isnull(), 'p_distance'] = 0
X.loc[X.p_depdelay.isnull(), 'p_depdelay'] = 0
X = X.drop(['p_year' , 'p_daymonth' , 'p_dayweek' , 'p_dest' , 'p_distance', 'dep_time'] , axis=1)

print('Number of feature : {}\nNumber of data : {}'.format(X.shape[1], X.shape[0]))
## Proccsing the split the data into testset(30%) and trainset (70%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=25)


## Processing multiple parameters testing and generate a dictinary to record the result
result = { 
            'idx' : [],
            'est' :[],
            'max_depth' :[], 
            'criterion' : [], 
            'class_weight' : [],
            'train_precision' : [],
            'train_recall' : [],
            'train_f1' : [],
            'train_acc' : [],
            'test_precision' : [],
            'test_recall' : [],
            'test_f1' : [],
            'test_acc' : [],
            'importance': [], 
            'time' : [] 
         } 

num_of_loop = { 
                'estimator' : [50] , 
                'max_depth' : [100],
                'criterion' : ['entropy'], 
                'class_weight' : [{0:1, 1:10}] 
              } 


config_list = []

for est in num_of_loop['estimator']: 
    for max_depth in num_of_loop['max_depth']:
        for criterion in num_of_loop['criterion']:
            for class_weight in num_of_loop['class_weight']:
                config_list.append( [est, max_depth, criterion, class_weight] )
                
config_list = list(enumerate(config_list))
start_index = 0
config_list = config_list[start_index:]


t=  tqdm(config_list, desc='Bar desc' , leave = True, position=0)
for config in t:
    est = config[1][0]
    max_depth = config[1][1]
    criterion = config[1][2]
    class_weight = config[1][3]
    
    clf = RandomForestClassifier(n_estimators=est, 
                                 max_depth=max_depth, 
                                 criterion=criterion, 
                                 random_state=14, 
                                 class_weight=class_weight,
                                 n_jobs=3)
    time_check = datetime.now()
    clf.fit(X_train, Y_train)
    p_train, r_train, f_train, a_train ,  p_test, r_test, f_test, a_test, importance = metric_calc(clf, X_train, Y_train, X_test, Y_test) 
    time_check = datetime.now() - time_check
    result['idx'].append(config[0])
    result['est'].append(est)
    result['max_depth'].append(max_depth)
    result['criterion'].append(criterion)
    result['class_weight'].append(class_weight)
    result['train_precision'].append(p_train)
    result['train_recall'].append(r_train)
    result['train_f1'].append(f_train)
    result['train_acc'].append(a_train)
    result['test_precision'].append(p_test)
    result['test_recall'].append(r_test)
    result['test_f1'].append(f_test)
    result['test_acc'].append(a_test)
    result['importance'].append(importance)
    result['time'].append(time_check) 
    t.set_description('idx:{}, time:{}\nestimator:{} , depth:{} , criterion:{}\np_train:{}, r_train:{}\np_test:{},r_test:{}\n'.format(config[0], time_check, est, max_depth, criterion, p_train, r_train, p_test, r_test) )
resultsample1_seed25 = pd.DataFrame(result)
feature_importance = resultsample1_seed25['importance'].values[0]
_tmp = {'feature' : [] , 'importance' : []}
for i in feature_importance: 
    _tmp['feature'].append(i[0])
    _tmp['importance'].append(i[1])
feature_importance = pd.DataFrame(_tmp)
feature_importance.sort_values('importance' , ascending=False, inplace=True)

import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(10, 8))
plt.title('Feature vs Importance')
plt.bar(feature_importance.feature, feature_importance.importance)
plt.xticks(rotation=90)
plt.ylabel('Importance(sklearn)', size=15)
plt.xlabel('Feature', size=15)
plt.show()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module): 
    def __init__(self, feature_size, hidden_dim=50, dropout = 0.2 ,leakslope=0.3): 
        super(Net, self).__init__()
        self.leakslope = leakslope
        self.fc1 = nn.Linear(feature_size, hidden_dim)
        self.dropout1 = nn.Dropout(p=dropout)
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//3)
        self.dropout2 = nn.Dropout(p=dropout)
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim//3)
        self.fc3 = nn.Linear(hidden_dim//3, feature_size)
        self.dropout3 = nn.Dropout(p=dropout)
        self.batchnorm3 = nn.BatchNorm1d(feature_size)
        self.fc4 = nn.Linear(feature_size, hidden_dim//5)
        self.dropout4 = nn.Dropout(p=dropout)
        self.batchnorm4 = nn.BatchNorm1d(hidden_dim//5)
        self.fc_final = nn.Linear(hidden_dim//5, 1)
   
    def activation(self, x):
        act = torch.nn.LeakyReLU(self.leakslope)
        return act(x)
    
    
    def forward(self, _input): 
        residual = _input
        x = self.fc1(_input)
        x = self.activation(x)
        x = self.batchnorm1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.batchnorm2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.batchnorm3(x)
        x = self.dropout3(x)
        x += residual
        x = self.fc4(x)
        x = self.activation(x)
        x = self.batchnorm4(x)
        x = self.dropout4(x)
        x = self.fc_final(x)
        x = torch.sigmoid(x)
        return x
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=15)
_X_train= X_train.to_numpy()
_X_max = _X_train.max(axis=0)
_X_min = _X_train.min(axis=0)
_X_train = (_X_train - _X_min) / (_X_max - _X_min)
_X_test= X_test.to_numpy()
_X_max = _X_test.max(axis=0)
_X_min = _X_train.min(axis=0)
_X_test = (_X_test - _X_min) / (_X_max - _X_min)
train_X = torch.tensor(_X_train).float()
test_X = torch.tensor(_X_test).float()
test_Y = torch.tensor(Y_test.to_numpy()).float()
train_Y = torch.tensor(Y_train.to_numpy()).float()
# If can run around 150 epoch, the precision can achieve 90% and recall have achieve 0.5

hidden_dim = 250
dropout = 0.001
leakslope = 0.1
batch_size = 1000
epoch = 10 
lr = 1e-1
momentum = (0.99, 0.99)
shuffle = True


tmp_Y = train_Y.reshape((-1,1))
dataset = torch.cat((train_X, tmp_Y) , 1)
net = Net(feature_size=12, hidden_dim=hidden_dim, dropout = dropout, leakslope=leakslope)


from torch.utils.data import DataLoader
if torch.cuda.is_available():
    net = net.to('cuda')

loss_track = []  
recall_track = [] 
precision_track = []
optimizer = optim.Adam(net.parameters(), lr=lr, betas=momentum)
criterion = nn.BCELoss()

if torch.cuda.is_available:
    criterion.to('cuda')

for j in tqdm(range(epoch)):
    dataloader = DataLoader(dataset, batch_size=batch_size ,shuffle=shuffle)
    net.train()
    for i in dataloader:
        t_X = i[:,:-1]
        t_Y = i[:,-1].reshape(-1, 1)
            
        if torch.cuda.is_available():
            t_X = t_X.to('cuda')
            t_Y = t_Y.to('cuda')
        output = net(t_X)
        optimizer.zero_grad()
        loss = criterion(output, t_Y)
        loss.backward()
        optimizer.step()
        loss_track.append(loss.to('cpu').tolist())
    
    net.eval()
    predict_test =  net.forward(train_X).reshape(-1).to('cpu').tolist()
    result = pd.DataFrame({ 'predict':predict_test , 'target': train_Y.reshape(-1)  } )
    result['type'] = result.apply( lambda x: 'tn' if x['predict'] < 0.5 and x['target'] == 0 else 'fn' if x['predict'] < 0.5 and x['target'] == 1 else 'tp'if x['predict'] > 0.5 and x['target'] == 1 else 'fp',axis=1)
    check = result
    result = result.groupby('type').count()['predict']
    try:
        recall = result['tp'] *1.  / (result['tp']    + result['fn'])
    except: 
        recall = 0
    
    try:
        precision = result['tp'] *1.  / (result['tp']    + result['fp'])
    except: 
        precision = 0 
        
    recall_track.append(recall)
    precision_track.append(precision)
net.eval()
predict_test =  net.forward(train_X).reshape(-1).to('cpu').tolist()
import matplotlib.pyplot as plt 
%matplotlib inline
plt.hist(predict_test)
result = pd.DataFrame({ 'predict':predict_test , 'target': train_Y.reshape(-1).tolist()  } )
result['final_predict'] = result.apply( lambda x: 1 if x['predict'] > 0.5 else 0 , axis=1) 
tp = result[ (result['final_predict'] == 1) & (result['target'] == 1)].count()
pp = result[ (result['final_predict'] == 1) ].count()
ap = result[ (result['target'] == 1) ].count()

final_loss = loss.item()
precision = (tp*1./pp)['target']
recall = (tp*1./ap)['target']
accuracy = (result[result['target'] == result['final_predict'] ].count() *1.  / result.shape[0])['target']

print('precision : {} , recall: {} , accuracy:{} , loss:{}' .format(precision, recall, accuracy, final_loss))
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(loss_track)
plt.show()

plt.plot(precision_track)
plt.show()


plt.plot(recall_track)
plt.show()
import numpy as np
y = net.forward(test_X)
y = y.tolist()
y = np.array(y).reshape(-1).tolist()
tt_Y = test_Y.tolist()
result = pd.DataFrame({ 'predict':y , 'target': test_Y.reshape(-1).tolist()  } )
result['final_predict'] = result.apply( lambda x: 1 if x['predict'] > 0.5 else 0 , axis=1) 
tp = result[ (result['final_predict'] == 1) & (result['target'] == 1)].count()
pp = result[ (result['final_predict'] == 1) ].count()
ap = result[ (result['target'] == 1) ].count()
print('precision')
print(tp*1./pp)
print('\n\nrecall')
print(tp*1./ap)
