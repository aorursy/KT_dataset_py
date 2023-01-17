import gc

import numpy as np

import pandas as pd

import lightgbm as lgb

from scipy.stats import rankdata

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold
col_names = ['var_' + str(i) for i in range(200)]

col_types = [np.float32 for i in range(200)]



types = dict(zip(col_names, col_types))



types['target'] = np.uint8

types['ID_code'] = str
path_train='../input/santander-customer-transaction-prediction/train.csv'

path_test='../input/santander-customer-transaction-prediction/test.csv'



train = pd.read_csv(path_train, dtype=types)

test = pd.read_csv(path_test, dtype=types)
y = train.pop('target')

train_ids = train.pop('ID_code')

test_ids = test.pop('ID_code')
print("Memory usage for the train: {0:.2f}MB".

      format(train.memory_usage().sum() / (1024**2)))
print("Memory usage the test: {0:.2f}MB".

      format(test.memory_usage().sum() / (1024**2)))

path_load_fakes = '../input/list-of-fake-samples-and-public-private-lb-split/'



#Synthetic data indecies:

path_synth = path_load_fakes + 'synthetic_samples_indexes.npy'

synth_idx = np.load(path_synth)



#Public LB data indecies:

path_public_LB = path_load_fakes + 'public_LB.npy'

public_idx = np.array(list(np.load(path_public_LB).tolist()))



#Private LB data indecies:

path_private_LB = path_load_fakes + 'private_LB.npy'

private_idx = np.array(list(np.load(path_private_LB).tolist()))



#Indecies for public and private data combined:

true_idx=np.append(private_idx, public_idx)
test_synth = test.iloc[synth_idx.tolist(), :]

test_true = test.iloc[true_idx.tolist(), :]



print("\nThe shape of the synthetic test data:")

print(test_synth.shape)
print("\nThe shape of the true test data:")

print(test_true.shape)
len_train = len(train)

len_test_true = len(test_true)

len_test_synth = len(test_synth)



merged = pd.concat([train, test_true])



print("\nThe shape of the merged data:")

print(merged.shape)



del test, train, test_true

gc.collect()
original_features=merged.columns



for col in original_features:

    

    unq_val, inv, cnts = np.unique(merged[col].values, 

                                   return_inverse=True, 

                                   return_counts=True)

    

    merged[col+'_counts']=cnts[inv]

    

    #Populate the count column of the sythetic test with zeroes.

    test_synth[col+'_counts']=0

    

train = merged.iloc[:len_train, :]

    

test_true=merged.iloc[len_train:, :]



test = pd.concat([test_true, test_synth]).sort_index()



del merged, test_true, test_synth



gc.collect()



print("\nThe shape of the processed train data:")

print(train.shape)
print("\nThe shape of the processed test data:")

print(test.shape)
def augment(x,y,t=2):

        

    xs,xn = [],[]

    for i in range(t):

        mask = y>0

        x1 = x[mask].copy()

        

        x1_var=x1[:, 0:200]

        x1_counts=x1[:, 200:]

        

        del x1

        

        ids = np.arange(x1_var.shape[0])

        for c in range(x1_var.shape[1]):

            np.random.shuffle(ids)

            x1_var[:,c] = x1_var[ids][:,c]

            x1_counts[:,c] = x1_counts[ids][:,c]

            x1=np.column_stack((x1_var, x1_counts))

        xs.append(x1)



    for i in range(t//2):

        mask = y==0

        x1 = x[mask].copy()

        

        x1_var=x1[:, 0:200]

        x1_counts=x1[:, 200:]

        

        del x1

        

        ids = np.arange(x1_var.shape[0])

        for c in range(x1_var.shape[1]):

            np.random.shuffle(ids)

            x1_var[:,c] = x1_var[ids][:,c]

            x1_counts[:,c] = x1_counts[ids][:,c]

            x1=np.column_stack((x1_var, x1_counts))

        xn.append(x1)



    xs = np.vstack(xs)

    xn = np.vstack(xn)

    ys = np.ones(xs.shape[0])

    yn = np.zeros(xn.shape[0])

    x = np.vstack([x,xs,xn])

    y = np.concatenate([y,ys,yn])

    return x,y
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



test_preds = np.zeros((len(test), 1))

oof_preds = np.zeros((len(train), 1))



roc_cv =[]



params = {

    "objective" : "binary",

    "metric" : "auc",

    "boosting": 'gbdt',

    "max_bin": 255, 

    "max_depth" : -1,

    "num_leaves" : 3,

    "learning_rate" : 0.1,

    "min_data_in_leaf": 150,

    "min_sum_hessian_in_leaf": 25,

    "tree_learner": "serial",

    "boost_from_average": "false",

    "lambda_l1" : 1, 

    "verbosity" : 1,

    "min_gain_to_split": 0.3,

    }
for fold_, (trn_, val_) in enumerate(folds.split(y, y), 1):

    

    print("\nStarting fold {}".format(fold_))

    

    trn_x, trn_y = train.iloc[trn_, :], y.iloc[trn_].values.ravel()

    val_x, val_y = train.iloc[val_, :], y.iloc[val_].values.ravel()    

            

    features=trn_x.columns

        

    trn_aug_x, trn_aug_y = augment(trn_x.values, trn_y, 9)

    trn_aug_x = pd.DataFrame(trn_aug_x)

    trn_aug_x.columns = features

    

    #Compute the row sums of the counts (may give us a slight boost):

    count_feats=['var_'+str(i)+'_counts' for i in range(200)]

    trn_aug_x['tot']=trn_aug_x[count_feats].sum(axis=1)

    val_x['tot']=val_x[count_feats].sum(axis=1)

    test['tot']=test[count_feats].sum(axis=1)

        

    print("The shapes of trn_x and trn_y after augmentation are {} and {}, respectively".\

          format(trn_x.shape, trn_y.shape))

        

    print("The shapes of trn_aug_x and trn_aug_y after augmentation are {} and {}, respectively".\

          format(trn_aug_x.shape, trn_aug_y.shape))

        

    print("The shapes of val_x and val_y after augmentation are {} and {}, respectively".\

          format(val_x.shape, val_y.shape))

        

    print("The shape of test is {}".format(test.shape))

            

    print("\nConverting the data to lgbm format")

    trn_data = lgb.Dataset(trn_aug_x, label=trn_aug_y)

    val_data = lgb.Dataset(val_x, label=val_y)

    evals_result = {}

            

    print("Training the classifier")

    clf = lgb.train(params,

                    trn_data,

                    num_boost_round = 1000000,

                    valid_sets = [val_data],

                    early_stopping_rounds=3000,

                    verbose_eval = 3000,

                    evals_result=evals_result,

                   )

          

    print("\nMaking predictions for the validation data")

    val_pred = clf.predict(val_x, num_iteration=clf.best_iteration)

    

    oof_preds[val_, :] = val_pred.reshape((-1, 1))

    

    print("Computing the AUC score")

    roc_auc_fold=roc_auc_score(val_y, val_pred)

    roc_cv.append(roc_auc_fold)

        

    print("AUC = {}".format(roc_auc_fold))

    

    print("Making predictions for the test data")

    test_fold_pred = clf.predict(test, num_iteration=clf.best_iteration)



    test_preds += test_fold_pred.reshape((-1, 1))

    

test_preds /= 5

#Optional

#test_preds = rankdata(test_preds)/len(test_preds)
roc_score = round(sum(roc_cv)/len(roc_cv), 5)

print("\nAverage of the folds' AUCs = {}".format(roc_score))
roc_score_1 = round(roc_auc_score(y, oof_preds.ravel()), 5)

print("Combined folds' AUC = {}".format(roc_score_1))
st_dev = round(np.array(roc_cv).std(), 5)

print("The standard deviation = {}".format(st_dev))
sub = pd.DataFrame({'ID_code' : test_ids, 

                    'target' : test_preds.astype(np.float16).ravel()})



sub.to_csv('SUBMISSION.CSV', index=False)



print("All done!")