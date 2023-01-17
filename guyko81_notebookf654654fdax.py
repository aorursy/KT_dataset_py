# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb # XGBoost implementation



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



# read data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



features = [x for x in train.columns if x not in ['id','loss']]

#print(features)



cat_features = [x for x in train.select_dtypes(include=['object']).columns if x not in ['id','loss']]

num_features = [x for x in train.select_dtypes(exclude=['object']).columns if x not in ['id','loss']]

#print(cat_features)

#print(num_features)



train['log_loss'] = np.log(train['loss'])



ntrain = train.shape[0]

ntest = test.shape[0]

train_test = pd.concat((train[features], test[features])).reset_index(drop=True)

for c in range(len(cat_features)):

    train_test[cat_features[c]] = train_test[cat_features[c]].astype('category').cat.codes



train_x = train_test.iloc[:ntrain,:]

test_x = train_test.iloc[ntrain:,:]
train_x = train[features]

test_x = test[features]

for c in range(len(cat_features)):

    a = pd.DataFrame(train['log_loss'].groupby([train[cat_features[c]]]).mean())

    a[cat_features[c]] = a.index

    a.sort(['log_loss'], ascending=[1], inplace=True)

    a['A'] = list(range(len(a.index))) 

    a['A'] += 1000 #for cases where test and train have different category value set

    train_x[cat_features[c]] = pd.merge(left=train_x, right=a, how='left', on=cat_features[c])['A']

    test_x[cat_features[c]] = pd.merge(left=test_x, right=a, how='left', on=cat_features[c])['A']



train_x.head(n=20)

#train_x['cat1'] = pd.merge(left=train_x, right=a, how='left', on='cat1')['log_loss']

#train_x.head(n=20)



#from sklearn.tree import DecisionTreeRegressor

#train_x2 = train_test.iloc[:ntrain,:]

#test_x2 = train_test.iloc[ntrain:,:]

#for c in range(len(cat_features)):

#    new_features=[x for x in features if x != cat_features[c]]

#    for r in range(0,train_x[cat_features[c]].max()+1):

#        if train_x2.loc[train_x[cat_features[c]]==r,cat_features[c]].shape[0] > 0:

#            #see train['cat89'].unique() and test['cat89'].unique() - we never see 'F' in train

#            train_x_tree = train_x.loc[train_x[cat_features[c]]==r,new_features]

#            test_x_tree = test_x.loc[test_x[cat_features[c]]==r,new_features]

#            

#            num_of_split = np.minimum(20,np.int(np.ceil(train_x_tree.shape[0]/10)))

#            dtree = DecisionTreeRegressor(random_state=0, max_depth=1, min_samples_leaf=num_of_split )

#

#            regr = dtree.fit(train_x_tree, train.loc[train_x[cat_features[c]]==r,'log_loss'])

#

#            train_x2.loc[train_x[cat_features[c]]==r,cat_features[c]] = regr.predict(train_x_tree)

#            try:

#                test_x2.loc[test_x[cat_features[c]]==r,cat_features[c]] = regr.predict(test_x_tree)

#                #see train['cat89'].unique() and test['cat89'].unique() - we don't have 'I' in test

#            except:

#                pass

        
xgdmat = xgb.DMatrix(train_x, train['log_loss']) # Create our DMatrix to make XGBoost more efficient



params = {'eta': 0.01, 'seed':0, 'subsample': 0.5, 'colsample_bytree': 0.5, 

             'objective': 'reg:linear', 'max_depth':6, 'min_child_weight':3} 



# Grid Search CV optimized settings

num_rounds = 1000

bst = xgb.train(params, xgdmat, num_boost_round = num_rounds)
test_xgb = xgb.DMatrix(test_x)

submission = pd.read_csv("../input/sample_submission.csv")

submission.iloc[:, 1] = np.exp(bst.predict(test_xgb))

submission.to_csv('xgb.sick_categorical_encoding.csv', index=None)