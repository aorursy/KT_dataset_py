import sys

!cp ../input/rapids/rapids.0.15.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null

sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
import sys

import numpy as np

import pandas as pd

import gc

from sklearn.metrics import f1_score,roc_auc_score

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import mean_absolute_error

import cudf

from cuml.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt

from tqdm import tqdm
train_df = pd.read_csv('../input/conways-reverse-game-of-life-2020/train.csv')

test_df = pd.read_csv("../input/conways-reverse-game-of-life-2020/test.csv")

print(train_df.shape)

print(test_df.shape)
sample_size = 50000

train = train_df.drop(["id",'delta'], axis=1)  #Let's drop delta first, And we will use it in the later version.

train_x = train.iloc[:sample_size,625:].astype('int32') #Only use part of data for this demo. For quick update and modify.

train_y = train.iloc[:sample_size,:625].astype('int32')

test = test_df.drop(["id",'delta'], axis=1)

test_x = test.iloc[:,:]
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



del own_cell_obs_layer,deltas_
train_target = train_target.reshape(sample_size,625,1)
#features = ['locate_'+str(c) for c in range(9)] +['delta']+['start']

features = ['round_num','delta','origin','start']



all_train = np.concatenate((train_df,train_target),axis=2)



train_df = pd.DataFrame(all_train.reshape(sample_size*625,4)).astype('float32')

train_df.columns = features



test_df = pd.DataFrame(test_df.reshape(test_df.shape[0]*625,3)).astype('float32')

test_df.columns = features[:-1]



# target_df = pd.DataFrame(train_target.reshape(sample_size*625,1)).astype('float32')

# target_df.columns = ['stop']



predictions = np.zeros(test_df.shape[0])
train_df.head()
test_df.head()
Target = 'start'
features = ['round_num','delta','origin']
NUM_FOLDS = 5

skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)



test_df = cudf.from_pandas(test_df)



oof_preds = np.zeros(sample_size*625)

y_test = np.zeros(test_df.shape[0])



for fold, (train_ind, val_ind) in enumerate(skf.split(train_df.values, train_df['delta'].values)):

    

    tr_df, val_df = train_df.iloc[train_ind], train_df.iloc[val_ind]

    print('Fold', fold )



    tr_df = cudf.from_pandas( tr_df )

    val_df   = cudf.from_pandas( val_df )



    model = RandomForestRegressor(

            n_estimators=35,

            rows_sample = 0.35,

            max_depth=18,

            max_features="auto",        

            split_algo=0,

            bootstrap=False, #Don't use repeated rows, this is important to set to False to improve accuracy

        ).fit( tr_df[features], tr_df[Target])

        

    pred = model.predict( val_df[features] ).to_array()

    oof_preds[val_ind] = pred

        

    y_test += model.predict( test_df[features] ).to_array() / NUM_FOLDS

    del model; _=gc.collect()

    

#y_test = np.round( y_test )
# for i in tqdm(range(95,100)):

#     threshold = np.percentile(oof_preds,i)

#     mae_loss = mean_absolute_error(train_df[Target].values, np.array([1 if pred > threshold else 0 for pred in oof_preds]))

#     if mae_loss<min_mae:

#         min_mae = mae_loss

#         min_threshold = threshold

#         print(i)

# print('min_mae:',min_mae,'min_threshold:',min_threshold)
predictions = y_test
threshold = np.percentile(predictions,94)



#result =[1 if pred >  threshold[i%625] else 0 for i,pred in enumerate(predictions)]

result = [1 if pred > threshold else 0 for pred in predictions]

print(predictions[:10])

submit = pd.read_csv("../input/conways-reverse-game-of-life-2020/sample_submission.csv")



ids = submit.iloc[:,0].values

ids = ids.reshape(ids.shape[0],1)



sub = np.array(predictions).reshape(test_x.shape[0],test_x.shape[1])

sub = np.hstack((ids,sub))



submission = pd.DataFrame(sub)

submission.columns = submit.columns[:]

submission.index = submit.index

submission.to_csv('submission.csv',index=False)
plt.imshow(submission.iloc[3,1:].values.reshape(25,25)) #predict
plt.imshow(test_x.iloc[3,:].values.reshape(25,25)) #origin
print(submission.iloc[2,1:].values.sum()/625)

print(test_x.iloc[2,:].values.sum()/625)
submission.to_csv('submission.csv',index=False)