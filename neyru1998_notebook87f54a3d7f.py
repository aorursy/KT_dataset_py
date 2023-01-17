import pandas as  pd 
import numpy as np 
import os 
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from joblib import Parallel, delayed
import lightgbm as lgbm
# ! pip install lightgbm 
# ! pip install tqdm 
test.head(10)
def get_df_chunks(df, nb_chunks=None, chunksize=None ) :
    if bool(nb_chunks) == bool(chunksize):
        raise ValueError("You must provide only one argument: nb_chunks or chunksize")
    nb_rows = len(df)
    if nb_chunks:
        chunksize = nb_rows // nb_chunks + (nb_rows % nb_chunks > 0)
    for i in range(0, nb_rows, chunksize):
        yield df[i : i + chunksize]
        
def map_amino_acid(x):
    return np.array([emb_features_dict[e] for e in x ])
def get_features(x): 
    output=pd.Series()
    output["mean"]=np.mean(x)
    output["min"]=np.min(x)       
    output["max"]=np.max(x)     
    output["std"]=np.std(x)      
    output["var"]=np.var(x) 
    output["ptp"]=np.ptp(x) 
    axis=-1
    output["min_ax_"+str(axis)]=np.mean(np.min(x,axis=axis))       
    output["max_ax_"+str(axis)]=np.mean(np.max(x,axis=axis))     
    output["std_ax_"+str(axis)]=np.mean(np.std(x,axis=axis))      
    output["var_ax_"+str(axis)]=np.mean(np.var(x,axis=axis)) 
    output["ptp_ax_"+str(axis)]= output["max_ax_"+str(axis)]-output["min_ax_"+str(axis)]
        
    return output

def get_features_df_train(chunk):
    features=chunk.Sequence.apply(map_amino_acid)
    features=features.apply(get_features)
    features["ID"]=chunk["ID"]
    features["target"]=chunk["target"] 
    return features

def get_features_df_test(chunk):
    features=chunk.Sequence.apply(map_amino_acid)
    features=features.apply(get_features)
    features["ID"]=chunk["ID"]
    return features

emb_features = pd.read_csv("../input/enzymes/amino_acid_embeddings.csv")
emb_features.head()
emb_features_dict = {k:np.mean(v) for k,v in zip(emb_features.Amino_Acid,emb_features.drop("Amino_Acid",1).values)}
emb_features_dict
del emb_features
train = pd.read_csv("../input/trainn/train.csv")
test = pd.read_csv("../input/enzymes/test.csv")
chunksize = 2000
n_jobs = 20 # nbr of CPUs to use (if you increase it you may face memoery issue so try to decrease the chunksize)
df_iter = get_df_chunks(train, chunksize=chunksize)
train_features = Parallel(n_jobs=n_jobs)(delayed(get_features_df_train)(chunk) for chunk in tqdm(df_iter))
df_iter = get_df_chunks(test, chunksize=chunksize)
test_features = Parallel(n_jobs=n_jobs)(delayed(get_features_df_test)(chunk) for chunk in tqdm(df_iter))
test_features = pd.concat(test_features)
train_features = pd.concat(train_features)
#test_features.to_csv("./proc_data/test_features.csv",index=False)
#train_features.to_csv("./proc_data/train_features.csv",index=False)
train,val = train_test_split(train_features,test_size=0.1,random_state=1994,stratify = train['target'])
params={'bagging_fraction': 1,
 'bagging_freq': 1,
 'boosting_type': 'gbdt',
 'feature_fraction': 0.8,
 'learning_rate': 0.01,
 'max_depth': 7,
 'metric': 'multi_logloss',
 'min_data_in_leaf': 33,
 'num_leaves': 44,
 'num_threads': 8,
 'objective': 'multiclass',
 "num_class":train.target.nunique(),
 'seed': 2020,
 'tree_learner': 'serial'}
num_rounds=1000000
early_stoping_rounds=50
verbose_eval=50
features=train.drop(["ID","target"],1).columns.tolist()
def lgbm_model(train,validation,test_data,features,target_name,params):
    dtrain = lgbm.Dataset(train[features],train[target_name])
    dval = lgbm.Dataset(validation[features],validation[target_name])
    lgbm_model= lgbm.train(params=params,
                train_set=dtrain,
                num_boost_round=num_rounds,
                valid_sets=[dtrain,dval],
                 verbose_eval=verbose_eval,
                 early_stopping_rounds=early_stoping_rounds)
    best_iteration = lgbm_model.best_iteration
    validation_prediction=lgbm_model.predict(validation[features], num_iteration=best_iteration)
    test_prediction=lgbm_model.predict(test_data[features], num_iteration=best_iteration)
    return test_prediction ,validation_prediction
test_pred,val_pred = lgbm_model(train, val, test_features, features, "target", params)
sub = test[["ID"]].copy()
for i in range(test_pred.shape[1]):
    sub["target_{}".format(i)]=test_pred[:,i]
sub.to_csv("StarterNotebookML_sub.csv",index=False)