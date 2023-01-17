# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn import preprocessing

from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor 

from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import Imputer

from sklearn.model_selection import KFold, cross_val_score, train_test_split

import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.preprocessing import LabelEncoder



import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn 



def display_feature(data, feature):

    fig, ax = plt.subplots()

    ax.scatter(x = data[feature], y = data['SalePrice'])

    plt.ylabel('SalePrice', fontsize=13)

    plt.xlabel(feature, fontsize=13)

    plt.show()





def allenamento(model):

    kf = KFold(5, shuffle=True, random_state=42).get_n_splits(train_data)

    res= np.sqrt(-cross_val_score(model, train_data, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(res.mean())



def scoreing(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def imputed_values_plus(train_data, test_data): 

    imputed_train_plus = train_data.copy()

    imputed_test_plus = test_data.copy()



    cols_with_missing = (col for col in train_data.columns 

                                    if train_data[col].isnull().any())

    for col in cols_with_missing:

        imputed_train_plus[col + '_was_missing'] = imputed_train_plus[col].isnull()

        imputed_test_plus[col + '_was_missing'] = imputed_test_plus[col].isnull()



    imputer = Imputer()

    imputed_train_plus = imputer.fit_transform(imputed_train_plus)

    imputed_test_plus = imputer.transform(imputed_test_plus)

    

    train_data = imputed_train_plus

    test_data = imputed_test_plus

    return(imputed_train_plus, imputed_test_plus)



def gboost_model(train_data, test_data, y_train):

    GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

    my_pipeline = make_pipeline(Imputer(), GBoost )

    all = allenamento(my_pipeline)

    print("allenamento" + str(all) ) 

    my_pipeline.fit(train_data, y_train)

    my_pipeline_train_pred = my_pipeline.predict(train_data) 

    print("score gboost: ", scoreing(y_train, my_pipeline_train_pred))

    my_pipeline_test_pred = np.expm1( my_pipeline.predict(test_data)  ) 

    return (my_pipeline_test_pred)



def test_max_leaf(max_leaf_nodes):

    r_forest = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    my_pipeline = make_pipeline(Imputer(), r_forest)

    my_pipeline.fit(train_data, y_train)

    my_pipeline_train_pred = my_pipeline.predict(train_data) 

    score = scoreing(y_train, my_pipeline_train_pred)

    return(score)



def random_forest_model():

    # looking for the best number of nodes for prediction with RandomForestRegressor model 

    candidate_max_leaf_nodes = [20, 50, 70, 90, 100,120, 150, 200 ,250, 300 ]

    sol = []

    for lf in candidate_max_leaf_nodes: 

        sol.append( test_max_leaf(lf) )

        

    min_index = sol.index(min(sol))

    # print(min_index , min(sol) )

    max_leaf_nodes = candidate_max_leaf_nodes[min_index]

    

    r_forest = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=9)

    my_pipeline = make_pipeline(Imputer(), r_forest)

    my_pipeline.fit(train_data, y_train)

    my_pipeline_train_pred = my_pipeline.predict(train_data) 

    print("score random", scoreing(y_train, my_pipeline_train_pred))               

    my_pipeline_test_pred = np.expm1( my_pipeline.predict(test_data)  )

    return (my_pipeline_test_pred)



def XGB_model(train_data, test_data, y_train):

    xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=4, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1)

    # my_pipeline = make_pipeline(Imputer(), xgb )

    all = allenamento(xgb)

    print("allenamento " + str(all) ) 

    xgb.fit(train_data, y_train)

    xgb_train_pred = xgb.predict(train_data) 

    print("score xgb", scoreing(y_train, xgb_train_pred))

    xgb_test_pred = np.expm1( xgb.predict(test_data)  ) 

    return (xgb_test_pred)

    



if __name__ == "__main__":

    

    # path for the data

    train_path = '../input/train.csv'

    test_path = '../input/test.csv'

            

    # load data

    train_data = pd.read_csv(str(train_path))

    test_data = pd.read_csv(str(test_path))

    

    # exclude what is not a string and id 

    # train_data = train_data.select_dtypes(exclude=['object'])

    # test_data = test_data.select_dtypes(exclude=['object'])

    

    # save id for submission

    train_id = train_data['Id'] 

    test_id = test_data['Id']  

    

    #display_feature(train_data, 'GrLivArea')

    display_feature(train_data, 'WoodDeckSF')

    train_data.drop( train_data[(train_data['GrLivArea']>4000) & (train_data['SalePrice']<300000)].index, inplace=True)

    train_data.drop( train_data[(train_data['WoodDeckSF']<400) & (train_data['SalePrice']>600000)].index, inplace=True)

    # train_data = train_data.drop( train_data[(train_data['TotRmsAbvGrd']==10) & (train_data['SalePrice']>=700000)].index)

    # train_data.drop( train_data[train_data['SalePrice']<300000].index, inplace = True)

    display_feature(train_data, 'WoodDeckSF')

    

    train_data.drop('Id', axis = 1, inplace = True)

    test_data.drop('Id', axis = 1, inplace = True)

    

    

    train_data["SalePrice"] = np.log1p(train_data["SalePrice"])

    y_train = train_data.SalePrice.values

    

    ntrain = train_data.shape[0]

    ntest = test_data.shape[0]

    train_data.drop(['SalePrice'], axis=1, inplace=True)

    

    merged_data = pd.concat((train_data, test_data)).reset_index(drop=True)



    for feature in merged_data:

        if merged_data[feature].dtype != "object":

                merged_data[feature] = np.log1p(merged_data[feature])

     

    # try encode string 

    categorical_data = merged_data.select_dtypes(include=['object'])

    for c in categorical_data: 

        lbl = LabelEncoder()

        lbl.fit( list(merged_data[c].values) )

        lbl.transform( list(merged_data[c].values) )

    merged_data = pd.get_dummies(merged_data)

     

    

    train_data = merged_data[:ntrain]

    test_data = merged_data[ntrain:]

    

    # cols_with_missing = [col for col in train_data.columns if train_data[col].isnull().any()]

    # reduced_X = train_data.drop(cols_with_missing, axis=1)

    train_data, test_data = imputed_values_plus(train_data, test_data)

    predictions_xgb = XGB_model(train_data, test_data, y_train) 

    predictions_gboost = gboost_model(train_data, test_data, y_train)

    # predictions_rf = random_forest_model() 

    

    

    predictions =  predictions_gboost*0.5 + predictions_xgb*0.5

    print("OK")

    

    # steps to submit and create csv

    sub = pd.DataFrame()

    sub['Id'] = test_id

    sub['SalePrice'] = predictions

    sub.to_csv('submission.csv',index=False)

    '''

    y = train_data.SalePrice

    X = train_data.drop('SalePrice', axis=1)

    print( XGB_model_train(X,y) ) 

    print( get_mae(train_path) )  

    '''