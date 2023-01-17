# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from lightgbm import LGBMClassifier

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from sklearn.metrics import log_loss, accuracy_score

import xgboost as xgb

from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")

import logging

from heamy.dataset import Dataset

from heamy.estimator import Classifier

from heamy.pipeline import ModelsPipeline

CACHE=False

NFOLDS = 5

DATA_DIR = "../input/learn-together"

SUBMISSION_FILE = "{0}/sample_submission.csv".format(DATA_DIR)
def addFeatures(df):

    #horizontal and vertical distance to hydrology can be easily combined

    cols = ['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology']

    df['distance_to_hydrology'] = df[cols].apply(np.linalg.norm, axis=1)

    

    #adding a few combinations of distance features to help enhance the classification

    cols = ['Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points',

            'Horizontal_Distance_To_Hydrology']

    df['distance_mean'] = df[cols].mean(axis=1)

    df['distance_sum'] = df[cols].sum(axis=1)

    df['distance_dif_road_fire'] = df[cols[0]] - df[cols[1]]

    df['distance_dif_hydro_road'] = df[cols[2]] - df[cols[0]]

    df['distance_dif_hydro_fire'] = df[cols[2]] - df[cols[1]]

    

    #taking some factors influencing the amount of radiation

    df['cosine_of_slope'] = np.cos(np.radians(df['Slope']) )

    #X['Diff_azimuth_aspect_9am'] = np.cos(np.radians(123.29-X['Aspect']))

    #X['Diff_azimuth_aspect_12noon'] = np.cos(np.radians(181.65-X['Aspect']))

    #X['Diff_azimuth_aspect_3pm'] = np.cos(np.radians(238.56-X['Aspect']))



    #sum of Hillshades

    shades = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']

    #df['Sum_of_shades'] = df[shades].sum(1)

    weights = pd.Series([0.299, 0.587, 0.114], index=cols)

    df['hillshade'] = (df[shades]*weights).sum(1)



    df['elevation_vdh'] = df['Elevation'] - df['Vertical_Distance_To_Hydrology']

    

    #binned features

    bin_defs = [

        # col name, bin size, new name

        ('Elevation', 200, 'Binned_Elevation'), 

        ('Aspect', 45, 'Binned_Aspect'),

        ('Slope', 6, 'Binned_Slope'),

    ]

    

    for col_name, bin_size, new_name in bin_defs:

        df[new_name] = np.floor(df[col_name]/bin_size)

        

    print('Total number of features : %d' % (df.shape)[1])

    return df
weights = [7.593211270832795, 1, 1.939598531912596, 1.1429202760461918, 1.1032958863198228, 1.0883962288167983,

           2.5458413312744073, 1.3142614728641315, 2.8449343430207925, 0.8136208789805158, 1.0709147648909365, 

           0.7707629737823462, 1.0, 0.8104528513180582, 1.0, 1.0749417474588394, 0.7654163512155585, 0.8030187040170972,

           1.0329576637993996, 7.614421045866052, 1, 0.9476259741687187, 800.0, 533.3204003431088, 879.7676928167089, 

           670.0124183890873, 618.7672505753314, 552.7902772091511, 765.8603672921006, 800.0, 726.0527101376886, 

           597.4990430522444, 670.7967759239023, 482.17013705648037, 541.3176529658227, 800, 800.0, 697.5757747281089, 

           631.1298454088087, 642.4394575971372, 629.6475870178654, 779.8391046580264, 800.0, 281.70458567820396, 

           664.715674176997, 1127.2316339753593, 704.1050089801397, 791.4569525873807, 386.18744192290285, 

           666.9332363939681, 502.1957048164419, 681.803592224423, 601.6970391056689, 768.0795748801415, 800.0, 

           741.1224887205307, 732.1737832122601, 800, 800, 954.5115560927762, 638.6112784779291, 800, 798.1741071391841, 

           800.0, 800.0, 662.1283969196991]
def preprocessData():

    

    train = pd.read_csv('../input/learn-together/train.csv', index_col='Id')

    test = pd.read_csv('../input/learn-together/test.csv', index_col='Id')

    y_train = train['Cover_Type']

    

    classes = train.Cover_Type.unique()

    num_classes = len(classes)

    print("There are %i classes: %s " % (num_classes, classes))

    train.drop(['Cover_Type'], axis=1, inplace=True)



    train = addFeatures(train)    

    test = addFeatures(test)



    dtrn_first_ten = train.loc[:,:'Horizontal_Distance_To_Fire_Points']

    dtrn_wa_st = train.loc[:,'Wilderness_Area1':'Soil_Type40']

    dtrn_added_features = train.loc[:,'distance_to_hydrology':]

    dtrn_ = pd.concat([dtrn_first_ten,dtrn_added_features,dtrn_wa_st],axis=1)



    dtst_first_ten = test.loc[:,:'Horizontal_Distance_To_Fire_Points']

    dtst_wa_st = test.loc[:,'Wilderness_Area1':'Soil_Type40']

    dtst_added_features = test.loc[:,'distance_to_hydrology':]

    dtst_ = pd.concat([dtst_first_ten,dtst_added_features,dtst_wa_st],axis=1)

    

    for i in range(len(train.columns)):

        c = train.columns[i]

        train[c] *= weights[i]

        test[c] *= weights[i]

    

    train = dtrn_.values

    test = dtst_.values

    y = y_train.ravel()-1

   

    return {'X_train': train, 'X_test': test, 'y_train': y}  #dtrn_, dtst_, y_train
dataset = Dataset(preprocessor=preprocessData, use_cache=True)
knn_param = {

    'n_neighbors':1,

    'p':1

}



rf_param = {    

    'n_estimators':181, 

    'max_features':'sqrt', 

    'bootstrap':False,

    'max_depth':60,

    'min_samples_split':2,

    'min_samples_leaf':1,

    'random_state':1

    }



et_param = {

    'n_estimators':500,

    'max_features':66,

    'min_samples_split':5,

    'min_samples_leaf':1,

    'random_state':1

    }



lgb_param = {

    'objective':'multiclass',

    'num_class':7,

    'learning_rate':0.2,

    'num_leaves':149,

    'random_state':1

}



lr_param = {

    'multi_class':'multinomial', 

    'solver':'newton-cg', 

    'random_state':1

}



xg_params = {

        'seed': 0,

        'colsample_bytree': 0.7,

        'silent': 1,

        'subsample': 0.7,

        'learning_rate': 0.1,

        'objective': 'multi:softprob',   

        'num_class': 7,

        'max_depth': 4,

        'min_child_weight': 1,

        'eval_metric': 'mlogloss',

        'nrounds': 200

    }
knn = Classifier(dataset=dataset, estimator = KNeighborsClassifier, use_cache=CACHE, parameters=knn_param,name='knn')

rf = Classifier(dataset=dataset, estimator = RandomForestClassifier, use_cache=CACHE, parameters=rf_param,name='rf')

et = Classifier(dataset=dataset, estimator=ExtraTreesClassifier, use_cache=CACHE, parameters=et_param,name='et')

lgb = Classifier(dataset=dataset, estimator=LGBMClassifier, use_cache=CACHE, parameters=lgb_param,name='lgb')

lr = Classifier(dataset=dataset, estimator=LogisticRegression, use_cache=CACHE, parameters=lr_param,name='lr')

xgf = Classifier(dataset=dataset, estimator=XGBClassifier, use_cache=CACHE, parameters=xg_params,name='xgf')

pipeline = ModelsPipeline(knn, rf, et, lgb, lr, xgf) 

stack_ds = pipeline.stack(k=NFOLDS,seed=1)
# Train LogisticRegression on stacked data (second stage)

lr = LogisticRegression

lr_params = {'C': 5, 'random_state' : 1, 'solver' : 'liblinear', 'multi_class' : 'ovr',}

stacker = Classifier(dataset=stack_ds, estimator=lr, use_cache=False, parameters=lr_params)
results = stacker.validate(k=NFOLDS,scorer=log_loss)
dtrain = xgb.DMatrix(stack_ds.X_train, label=stack_ds.y_train)

dtest = xgb.DMatrix(stack_ds.X_test)



xgb_params = {

    'seed': 0,

    'colsample_bytree': 0.8,

    'silent': 1,

    'subsample': 0.6,

    'learning_rate': 0.05,

    'objective': 'multi:softprob',

    'num_class': 7,        

    'max_depth': 6,

    'num_parallel_tree': 1,

    'min_child_weight': 1,

    'eval_metric': 'mlogloss',

}



res = xgb.cv(xgb_params, dtrain, num_boost_round=1000, 

             nfold=NFOLDS, seed=1, stratified=True,

             early_stopping_rounds=20, verbose_eval=5, show_stdv=True)



best_nrounds = res.shape[0] - 1

cv_mean = res.iloc[-1, 2]

cv_std = res.iloc[-1, 3]



print('Ensemble-CV: {0}+{1}, best nrounds = {2}'.format(cv_mean, cv_std, best_nrounds))
model = xgb.train(xgb_params, dtrain, best_nrounds)

xpreds = model.predict(dtest)

predictions = np.round(np.argmax(xpreds, axis=1)).astype(int) + 1
submission = pd.read_csv(SUBMISSION_FILE)

submission['Cover_Type'] = predictions

submission.to_csv('submission.csv', index=None)