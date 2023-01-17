import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import matplotlib.pyplot as plt
import xgboost as xgb
print('Reading datasets')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print('Merging test and train')
test['target'] = np.nan
train = train.append(test).reset_index() # merge train and test
del test
print('Done, shape=',np.shape(train))
def rank_gauss(x, title=None):
    #Trying to implement rankGauss in python, here are my steps
    # 1) Get the index of the series
    # 2) sort the series
    # 3) standardize the series between -1 and 1
    # 4) apply erfinv to the standardized series
    # 5) create a new series using the index
    # Am i missing something ??
    # I subtract mean afterwards. And do not touch 1/0 (binary columns). 
    # The basic idea of this "RankGauss" was to apply rank trafo and them shape them like gaussians. 
    # Thats the basic idea. You can try your own variation of this.
    
    if(title!=None):
        fig, axs = plt.subplots(3, 3)
        fig.suptitle(title)
        axs[0][0].hist(x)

    from scipy.special import erfinv
    N = x.shape[0]
    temp = x.argsort()
    if(title!=None):
        print('1)', max(temp), min(temp))
        axs[0][1].hist(temp)
    rank_x = temp.argsort() / N
    if(title!=None):
        print('2)', max(rank_x), min(rank_x))
        axs[0][2].hist(rank_x)
    rank_x -= rank_x.mean()
    if(title!=None): 
        print('3)', max(rank_x), min(rank_x))
        axs[1][0].hist(rank_x)
    rank_x *= 2
    if(title!=None):
        print('4)', max(rank_x), min(rank_x))
        axs[1][1].hist(rank_x)
    efi_x = erfinv(rank_x)
    if(title!=None): 
        print('5)', max(efi_x), min(efi_x))
        axs[1][2].hist(efi_x)
    efi_x -= efi_x.mean()
    if(title!=None):
        print('6)', max(efi_x), min(efi_x))
        axs[2][0].hist(efi_x)
        plt.show()

    return efi_x

for i in train.columns:
    if i.endswith('cat'): # could be train[i].dtype == 'object' + labelencode, or maybe one hot encode...
        print('Categorical: ',i)
        train[i] = rank_gauss(train[i].values, i) # display rank gauss tranformation
    elif i.endswith('bin'):
        print('Binary: ',i)
    else:
        print('Numeric: ',i)
# TODO, use incremental learning with XGB and execute denoising autoencoder

from math import ceil
class DAESequence:
    def __init__(self, df, batch_size=128, random_cols=.15, random_rows=1, use_cache=False, use_lock=False, verbose=True):
        self.df = df.values.copy()     # ndarray baby
        self.batch_size = int(batch_size)
        self.len_data = df.shape[0]
        self.len_input_columns = df.shape[1]
        if(random_cols <= 0):
            self.random_cols = 0
        elif(random_cols >= 1):
            self.random_cols = self.len_input_columns
        else:
            self.random_cols = int(random_cols*self.len_input_columns)
        if(self.random_cols > self.len_input_columns):
            self.random_cols = self.len_input_columns
        self.random_rows = random_rows
        self.cache = None
        self.use_cache = use_cache
        self.use_lock = use_lock
        self.verbose = verbose
        
        self.lock = ReadWriteLock()
        self.on_epoch_end()

    def on_epoch_end(self):
        if(not self.use_cache):
            return
        if(self.use_lock):
            self.lock.acquire_write()
        if(self.verbose):
            print("Doing Cache")
        self.cache = {}
        for i in range(0, self.__len__()):
            self.cache[i] = self.__getitem__(i, True)
        if(self.use_lock):
            self.lock.release_write()
        gc.collect()
        if(self.verbose):
            print("Done")

    def __len__(self):
        return int(ceil(self.len_data / float(self.batch_size)))

    def __getitem__(self, idx, doing_cache=False):
        if(not doing_cache and self.cache is not None and not (self.random_cols <=0 or self.random_rows<=0)):
            if(idx in self.cache.keys()):
                if(self.use_lock):
                    self.lock.acquire_read()
                ret0, ret1 = self.cache[idx][0], self.cache[idx][1]
                if(self.use_lock):
                    self.lock.release_read()
                if (not doing_cache and self.verbose):
                    print('DAESequence Cache ', idx)
                return ret0, ret1
        idx_end = min(idx + self.batch_size, self.len_data)
        cur_len = idx_end - idx
        rows_to_sample = int(self.random_rows * cur_len)
        input_x = self.df[idx: idx_end]
        if (self.random_cols <= 0 or self.random_rows <= 0 or rows_to_sample<=0):
            return input_x, input_x # not dae
        # here start the magic
        random_rows = np.random.randint(low=0, high=self.len_data-rows_to_sample, size=rows_to_sample)
        random_rows[random_rows>idx] += cur_len # just to don't select twice the current rows
        cols_to_shuffle = np.random.randint(low=0, high=self.len_input_columns, size=self.random_cols)
        noise_x = input_x.copy()
        noise_x[0:rows_to_sample, cols_to_shuffle] = self.df[random_rows[:,None], cols_to_shuffle]
        if(not doing_cache and self.verbose):
            print('DAESequence ', idx)
        return noise_x, input_x

print("Create Model")
dae_data = train[train.columns.drop(['id','target'])] # only get "X" vector

# reduce data size, we are in kaggle =)
dae_data = dae_data[0:1000]
dae_data = dae_data.drop('index', axis=1)
len_input_columns, len_data = dae_data.shape[1], dae_data.shape[0]

eta = 0.1
max_depth = 6
subsample = 0.9
colsample_bytree = 0.85
min_child_weight = 55
num_boost_round = 500

params = {"objective": "reg:linear",
          "booster": "gbtree",
          "eta": eta,
          "max_depth": int(max_depth),
          "subsample": subsample,
          "colsample_bytree": colsample_bytree,
          "min_child_weight": min_child_weight,
          "silent": 1
          }
#FIRST IDEA, AE with stacking use the output (feature_Results) as input to next model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

train_rows = dae_data.shape[0]
feature_results = []
global_min_length, global_max_length = None, None
for target_g in dae_data.columns.tolist():
    features = dae_data.columns.drop(target_g).tolist()
    target_list = [target_g]
    train_fea = np.array(dae_data[features])
    for target in target_list:
        train_label = dae_data[target]
        kfold = KFold(n_splits=5, random_state=218, shuffle=True)
        kf = kfold.split(dae_data)
        cv_train = np.zeros(shape=(dae_data.shape[0], 1))
        min_length, max_length = None, None
        for i, (train_fold, validate) in enumerate(kf):
            X_train, X_validate, label_train, label_validate = train_fea[train_fold, :], train_fea[validate, :], train_label[train_fold], train_label[validate]
            dtrain = xgb.DMatrix(X_train, label_train)
            dvalid = xgb.DMatrix(X_validate, label_validate)
            watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
            bst = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=50, verbose_eval=False)
            min_length, max_length = min_length if min_length is not None and min_length<bst.best_ntree_limit else bst.best_ntree_limit, \
                                     max_length if max_length is not None and max_length>bst.best_ntree_limit else bst.best_ntree_limit
            cv_train[validate, 0] += bst.predict(xgb.DMatrix(X_validate), ntree_limit=bst.best_ntree_limit)
        global_min_length, global_max_length = \
            global_min_length if global_min_length is not None and global_min_length<min_length else min_length, \
            global_max_length if global_max_length is not None and global_max_length>max_length else max_length
        print(target, 'mse: ', mean_squared_error(train_label, cv_train), ' min/max best_ntree_limit: ',min_length, max_length)
        feature_results.append(cv_train)

feature_results = np.hstack(feature_results)
print("AE MSE from train data: ", mean_squared_error(dae_data, feature_results),
     'min/max best_ntree_limit: ',global_min_length, global_max_length)

#SECOND IDEA, AE with CV to get highest tree size, and train a regressor with this size 
# and predict 3 'layers', each one with one size (trees/4 * 1,2,3...)
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
print("CV Part")
tree_sizes={}
for target_g in dae_data.columns.tolist():
    features = dae_data.columns.drop(target_g).tolist()
    dtrain = xgb.DMatrix(dae_data[features].values, dae_data[target_g].values)
    bst = xgb.cv(params, dtrain, num_boost_round, early_stopping_rounds=50, verbose_eval=False, nfold=5)
    tree_sizes[target_g] = bst.shape[0]
    print(target_g, tree_sizes[target_g])
print(tree_sizes, min(tree_sizes.values()), max(tree_sizes.values()))
max_trees = max(tree_sizes.values())
print("Tree size: ",max_trees)
# we could get max size, fit models (overfit obvious), half size and 1/3 size
feature_results2 = []
features = []
print("CREATE FEATURE PART ", len(dae_data.columns.tolist()))
for target_g in dae_data.columns.tolist():
    features = dae_data.columns.drop(target_g).tolist()
    dtrain = xgb.DMatrix(dae_data[features].values, dae_data[target_g].values)
    size = tree_sizes[target_g] # max_trees
    bst = xgb.train(params, dtrain, num_boost_round=size)
    pred = bst.predict(dtrain)
    feature_results2.append(pred)
    print('creating features: ',target_g, size, ' error:', mean_squared_error(dae_data[target_g], pred))
    if(int(size/4)<=0):
        features.append(bst.predict(dtrain, ntree_limit =size) )
    else:
        features.append(bst.predict(dtrain, ntree_limit =int(size/4)*1) )
        features.append(bst.predict(dtrain, ntree_limit =int(size/4)*2) )
        features.append(bst.predict(dtrain, ntree_limit =int(size/4)*3) )
print(len(feature_results2))
feature_results2 = np.hstack(feature_results2).reshape(-1,dae_data.shape[1])
features = np.hstack(features)


# this error is a bit high, maybe we should do max_tree per feature? or stack too?
print("AE MSE from train data: ", mean_squared_error(dae_data, feature_results2))

print("Features = ",features.shape)

plt.hist(dae_data)
plt.show()
plt.hist(feature_results, bins=100)
plt.show()
plt.hist(feature_results2, bins=100)
plt.show()
plt.hist(features)
plt.show()

