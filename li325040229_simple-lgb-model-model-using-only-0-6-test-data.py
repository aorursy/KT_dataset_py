import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# from keras.models import Sequential

# from keras.layers import Dense,Dropout,Conv2D,InputLayer,TimeDistributed,MaxPooling2D,Flatten

# from keras.layers import LSTM



from numpy import array

#from keras.models import Sequential, load_model

from tqdm import tqdm

import seaborn as sns



import lightgbm as lgb

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold, KFold

train_df = pd.read_csv('../input/conways-reverse-game-of-life-2020/train.csv')

test_df = pd.read_csv("../input/conways-reverse-game-of-life-2020/test.csv")

print(train_df.shape)

print(test_df.shape)
train_df.head()
sample_size = 300

train = train_df.drop(["id",'delta'], axis=1)  #Let's drop delta first, And we will use it in the later version.

train_x = train.iloc[:sample_size,625:].astype('int32') #Only use part of data for this demo. For quick update and modify.

train_y = train.iloc[:sample_size,:625].astype('int32')

test = test_df.drop(["id",'delta'], axis=1)

test_x = test.iloc[:,:]
test_x.head()
view_size = 1

size = 25



own_cell_pos = [[i//25,i%25] for i in range(625)]

own_cell_pos_rec = [[pos[0] + size, pos[1] + size] for pos in own_cell_pos]



#Get train obs

train_sample = np.array(train_x.iloc[:sample_size,:]).reshape(sample_size,25,25)

tile_obs_layer = np.tile(train_sample, [1, 3, 3])

own_cell_obs_layer = [np.sum(np.array([tile_obs_layer[i,pos[0] - view_size: pos[0] + view_size + 1, pos[1] - view_size: pos[1] + view_size + 1] 

                                       for pos in own_cell_pos_rec]).reshape(625,9),axis=1) for i in range(sample_size)]



deltas = train_df['delta'][:sample_size].values

deltas_ = np.array([[delta]*625 for delta in deltas]).reshape(sample_size,625,1)



origin_loc =  train_sample.reshape(sample_size,625,1)

own_cell_obs_layer =  np.array(own_cell_obs_layer).reshape(sample_size,625,1)



train_df = np.concatenate((own_cell_obs_layer,deltas_),axis=2).astype('int32')

train_df = np.concatenate((train_df,origin_loc),axis=2).astype('int32')

print('train_set shape',train_df.shape)

train_target = np.array(train_y)





#Get test obs

test_set = np.array(test_x.iloc[:,:]).reshape(test_x.shape[0],25,25)

origin_loc =  test_set.reshape(test_x.shape[0],625,1)



tile_obs_layer = np.tile(test_set, [1, 3, 3])

own_cell_obs_layer = [tile_obs_layer[:,pos[0] - view_size: pos[0] + view_size + 1, pos[1] - view_size: pos[1] + view_size + 1] for pos in own_cell_pos_rec]

own_cell_obs_layer = [np.sum(np.array([tile_obs_layer[i,pos[0] - view_size: pos[0] + view_size + 1, pos[1] - view_size: pos[1] + view_size + 1] 

                                       for pos in own_cell_pos_rec]).reshape(625,9),axis=1) for i in range(test_x.shape[0])]

deltas = test_df['delta'][:].values

deltas_ = np.array([[delta]*625 for delta in deltas]).reshape(test_x.shape[0],625,1)



own_cell_obs_layer =np.array(own_cell_obs_layer).reshape(test_x.shape[0],625,1)

test_df = np.concatenate((own_cell_obs_layer,deltas_),axis=2).astype('int32')

test_df = np.concatenate((test_df,origin_loc),axis=2).astype('int32')



print('test_set shape',test_df.shape)
train_df.shape
del own_cell_obs_layer,deltas_
print(train_target.shape)
num_folds = 5

features = ['round_num','delta','origin']

print('Features:',features)

folds = KFold(n_splits=num_folds, random_state=2020)

oof = np.zeros(sample_size*625)

getVal = np.zeros(sample_size*625)
train_df = pd.DataFrame(train_df.reshape(sample_size*625,3)).astype('int32')

train_df.columns = features



test_df = pd.DataFrame(test_df.reshape(test_df.shape[0]*625,3)).astype('int32')

test_df.columns = features



target_df = pd.DataFrame(train_target.reshape(sample_size*625,1)).astype('int32')

target_df.columns = ['start']



predictions = np.zeros(test_df.shape[0])
train_df.head()
param = {

    'bagging_freq': 5,

    'bagging_fraction': 0.335,

    'boost_from_average':'false',

    'boost': 'gbdt',

    'feature_fraction': 0.041,

    'learning_rate': 0.0083,

    'max_depth': -1,

    'metric':{'binary_logloss', 'mae'},

    'min_data_in_leaf': 80,

    'min_sum_hessian_in_leaf': 10.0,

    'num_leaves': 13,

    'num_threads': 8,

    'tree_learner': 'serial',

    'objective': 'binary', 

    'verbosity': -1

}
# cat_feature = ['delta']

# train_df[cat_feature[0]] = train_df[cat_feature[0]].astype('category')

# test_df[cat_feature[0]] = test_df[cat_feature[0]].astype('category')
# features = ['locate_1','locate_3','locate_4','locate_5','locate_7','delta']
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target_df.values)):

    

    X_train, y_train = train_df.iloc[trn_idx][features].astype('int32'), target_df.iloc[trn_idx].astype('int32')

    X_valid, y_valid = train_df.iloc[val_idx][features], target_df.iloc[val_idx].astype('int32')

    

    X_tr, y_tr = X_train.values,[i[0] for i in y_train.values]

    X_tr = pd.DataFrame(X_tr)

    

    X_tr.columns = X_train.columns

    #cat

    #X_tr[cat_feature[0]] = X_tr[cat_feature[0]].astype('category')

    print("Fold idx:{}".format(fold_ + 1))

    trn_data = lgb.Dataset(X_tr, label=y_tr)

    val_data = lgb.Dataset(X_valid, label=y_valid)

    

    clf = lgb.train(param, trn_data, 5000, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 100)

    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)

    getVal[val_idx]+= clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration) / folds.n_splits

    

    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits
# print()

print(np.median(predictions))

print(np.percentile(predictions,94))
threshold = np.percentile(predictions,94)

result = [1 if pred > threshold else 0 for pred in predictions]

print(predictions[:10])

submit = pd.read_csv("../input/conways-reverse-game-of-life-2020/sample_submission.csv")



ids = submit.iloc[:,0].values

ids = ids.reshape(ids.shape[0],1)



sub = np.array(result).reshape(test_x.shape[0],test_x.shape[1])

sub = np.hstack((ids,sub))



submission = pd.DataFrame(sub)

submission.columns = submit.columns[:]

submission.index = submit.index

plt.imshow(submission.iloc[0,1:].values.reshape(25,25))
plt.imshow(test_x.iloc[0,:].values.reshape(25,25))
print("\n >> CV score: {:<8.5f}".format(roc_auc_score(target_df, oof)))
submission.to_csv('submission.csv',index=False)