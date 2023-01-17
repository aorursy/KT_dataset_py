# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import pandas as pd

import numpy as np

import xgboost as xgb

from sklearn import ensemble,metrics,cross_validation



data = pd.read_csv('../input/train.csv')

test_ = pd.read_csv('../input/test.csv')



data['val']='train'

test_['val']='test'



all_data=pd.concat([data,test_])



cat_cols=[i for i in all_data.columns if 'cat' in i]

new_data=pd.get_dummies(all_data,columns=cat_cols,drop_first=True)

X=[i for i in new_data.columns if i not in ['id','target','val']]

Y=['target']

train_data=new_data[new_data['val']=='train']

test_data=new_data[new_data['val']=='test']

del all_data

del data

del new_data

print ('Data process Complete')


def gini(list_of_values):

    sorted_list = sorted(list(list_of_values))

    height, area = 0, 0

    for value in sorted_list:

        height += value

        area += height - value / 2.

    fair_area = height * len(list_of_values) / 2

    return (fair_area - area) / fair_area

  

def normalized_gini(y_pred, y):

    normalized_gini = gini(y_pred)/gini(y)

    return normalized_gini



def gini_xgb(preds, dtrain):

    labels = dtrain.get_label()

    gini_score = gini_normalized(labels, preds)

    return [('gini', gini_score)]



K = 5

kf = cross_validation.KFold(train_data.shape[0],n_folds= K, random_state = 3228, shuffle = True)
xgb_preds=[]

for train_index, test_index in kf:

    print ('A')

    train_X, valid_X = train_data[X].loc[train_index], train_data[X].loc[test_index]

    train_y, valid_y = train_data[Y].loc[train_index], train_data[Y].loc[test_index]



    # params configuration also from anokas' kernel

    xgb_params = {

        'eta': 0.02,

        'max_depth': 6,

        'subsample': 0.9,

        'objective': 'binary:logistic',

        'silent': 1,

        'colsample_bytree': 0.9

    }



    d_train = xgb.DMatrix(train_X, train_y)

    d_valid = xgb.DMatrix(valid_X, valid_y)

    d_test = xgb.DMatrix(test_data[X])

    

    model = xgb.train(xgb_params, d_train, num_boost_round = 400)

                        

    xgb_pred = model.predict(d_test)

    xgb_preds.append(list(xgb_pred))
preds=[]

for i in range(len(xgb_preds[0])):

    sum=0

    for j in range(K):

        sum+=xgb_preds[j][i]

    preds.append(sum / K)



output = pd.DataFrame({'id': test_['id'], 'target': preds})

output.to_csv("sol_sub.csv".format(K), index=False)   

print ('process done')