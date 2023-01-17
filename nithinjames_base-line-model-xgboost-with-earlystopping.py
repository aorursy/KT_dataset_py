import pandas as pd

import numpy as np

np.random.seed(1133)

import itertools



import xgboost as xgb

from sklearn.cross_validation import train_test_split

from sklearn.metrics import r2_score
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
#copy the target variable and drop the coloumn

target = train_data['y'].copy()

train_data.drop(['y'],inplace=True,axis=1)



#save test_id and drop ID from both train and test



test_id = test_data.ID.values.copy()

train_data.drop(['ID'],inplace=True,axis=1)

test_data.drop(['ID'],inplace=True,axis=1)
# remove constant columns,there are 12 features like dat

remove_const = []

for col in train_data.columns:

    if train_data[col].dtype !='object':

        

        if train_data[col].std() == 0:

            remove_const.append(col)
#remove those constant coloumns





train_data.drop(remove_const, axis=1, inplace=True)

test_data.drop(remove_const, axis=1, inplace=True)
#from an old script in santander competition

def remove_feat_identicals(data_frame):

    # Find features having the same values in the same order and

    # remove one of those redundant features.

    print("")

    print("identical features...")

    n_features = data_frame.shape[1]

    # Find the names of identical features by going through all the

    # combinations of features (each pair is compared only once).

    feat_delete = []

    for feat_1, feat_2 in itertools.combinations(

            iterable=data_frame.columns, r=2):

        if np.array_equal(data_frame[feat_1], data_frame[feat_2]):

            feat_delete.append(feat_2)

    feat_names_delete = np.unique(feat_delete)

    n_features_deleted = len(feat_names_delete)

    print("  - Delete %s / %s features (~= %.1f %%)" % (

        n_features_deleted, n_features,

        100.0 * (np.float(n_features_deleted) / n_features)))

    return feat_names_delete
#get the features that occuring in the same order

feature_to_delete = remove_feat_identicals(train_data)
#delete the features

train_data.drop(feature_to_delete, axis=1, inplace=True)

test_data.drop(feature_to_delete, axis=1, inplace=True)
#convert categorical values to one-hot encoding(since we are using xgboost,this may give better score)



train_dummies = pd.get_dummies(train_data)

test_dummies = pd.get_dummies(test_data)
def diff_list(first, second):

    second = set(second)

    return [item for item in first if item not in second]
#There are some coloumn that exist in train but not in test(vice-versa)

#find them and drop 

train_dummies.drop(diff_list(train_dummies.columns,test_dummies.columns),inplace=True,axis=1)

test_dummies.drop(diff_list(test_dummies.columns,train_dummies.columns),inplace=True,axis=1)
#Split the dataset to train and test

train_X,test_X,train_y,test_y = train_test_split(train_dummies,target,test_size=0.2,random_state=142)
#Find the mean of target variable to set as the base score for xgboost

y_mean = np.mean(target)




xgb_params = {

    'eta': 0.02,

    'max_depth': 4,

    'subsample': 0.95,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'base_score': y_mean,

    'min_child_weight' : 1

}

dtrain = xgb.DMatrix(train_X, train_y)

dtest = xgb.DMatrix(test_X,test_y)

evallist = [(dtrain,'train'),(dtest,'test')]
def xgb_r2_score(preds, dtrain):

    labels = dtrain.get_label()

    return 'r2', r2_score(labels, preds)
model = xgb.train(dict(xgb_params, silent=0),dtrain=dtrain,num_boost_round=1000,evals=evallist,

                  feval=xgb_r2_score,early_stopping_rounds=10,maximize=True)
#Make prediction and save results

xg_check = xgb.DMatrix(test_dummies)

test_pred = model.predict(xg_check)


# make predictions and save results

output = pd.DataFrame({'ID':test_id, 'y': test_pred})

output.to_csv('xgboost-categorical_sub.csv', index=False)