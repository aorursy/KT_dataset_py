# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import pickle

import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
faeture_list = ['image_name','target','sex','age_approx','anatom_site_general_challenge']



siim20_csv = pd.read_csv('../input/jpeg-melanoma-384x384/train.csv',usecols = faeture_list)

siim19_csv = pd.read_csv('../input/jpeg-isic2019-384x384/train.csv',usecols = faeture_list)
siim19_csv.head()
siim19_csv['year'] = '2019' 

siim20_csv['year'] = '2020'



siim_all = pd.concat([siim19_csv,siim20_csv],ignore_index = True)



train = siim_all
train_new = train.dropna()

train_new.info()
#SEED value

SEED_VALUE = 2244
train_new.target.value_counts()
print(50090/4921)
from sklearn.preprocessing import LabelEncoder

sex_enc = LabelEncoder()

anatom_enc = LabelEncoder()
train_new['sex_enc']   = sex_enc.fit_transform(train_new.sex.astype('str'))

train_new['anatom_enc']= anatom_enc.fit_transform(train_new.anatom_site_general_challenge.astype('str'))
train_new.age_approx.hist(bins=8, alpha=0.5)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()



train_new['age_approx_scaled'] = scaler.fit_transform(train_new[['age_approx']])
pickle.dump(sex_enc, open('sex_encoder', 'wb'))

pickle.dump(anatom_enc, open('anatom_encoder', 'wb'))

pickle.dump(scaler, open('age_encoder', 'wb'))
train_new.info()
X_data = train_new[['sex_enc','age_approx_scaled','anatom_enc','target']]

#y_data = train_new[['target']]
from sklearn.model_selection import StratifiedKFold



df = X_data

nfolds = 5



splits = StratifiedKFold(n_splits=nfolds, random_state=2020, shuffle=True)

splits = list(splits.split(df,df.target))



folds_splits = np.zeros(len(df)).astype(np.int)

for i in range(nfolds): folds_splits[splits[i][1]] = i



df['split'] = folds_splits

df.head()
def run_train():

    for fold_number in range(nfolds):

        print('Training started for Fold :' + str(fold_number))

    

        train_df = df[(df.split != fold_number)]

        valid_df = df[(df.split == fold_number)]

        

        #print(train_df.shape + valid_df.shape )

    

        train_model(train_df, valid_df, fold_number)
import pickle

import xgboost as xgb

from sklearn.metrics import cohen_kappa_score,classification_report

from sklearn.metrics import roc_auc_score



xgb_model = xgb.XGBClassifier(n_estimators=500,learning_rate=0.01,objective='binary:logistic',

                                  max_depth=5, eval_metric = 'auc', scale_pos_weight=10 )



def train_model(train_df, valid_df, fold_number):

    

    X_train = train_df[['sex_enc','age_approx_scaled','anatom_enc']]

    y_train = train_df.target

    

 #   xgb_model = xgb.XGBClassifier(n_estimators=500,learning_rate=0.01,objective='binary:logistic',

 #                                 max_depth=5, eval_metric = 'auc', scale_pos_weight=10 )

  

    #({'eta': 0.01, 'max_depth': 5, 'n_estimators': 500}, 0.7401964387321014)

    xgb_model.fit(X_train, y_train, verbose=True)

    

    X_test = valid_df[['sex_enc','age_approx_scaled','anatom_enc']]

    y_test = valid_df.target

    predictions = xgb_model.predict(X_test)

    

    #print(predictions)

    #print(cohen_kappa_score(y_test, predictions,weights='quadratic'))

    #model_score = cohen_kappa_score(y_test, predictions,weights='quadratic')

    #print(model_score)

    #model_name = 'xgboost-' + str(fold_number) + '-' + str(model_score) + '.dat'

    #pickle.dump(xgb_model, open(model_name, 'wb'))

    print(roc_auc_score(y_test, predictions))

    print('---------------------------------------------------------------')
run_train()
from sklearn.model_selection import GridSearchCV



params = {

     "objective"    : 'binary:logistic',

     "num_classes"  : 2

     #"min_child_weight" : [ 1, 3, 5, 7 ],

     #"gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],

     #"colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]

     }



parameters = {

     "eta"          : [ 0.001, 0.002, 0.01 ] ,

     "max_depth"    : [ 3 ,5, 7],

     "n_estimators" : [200,300,500,600]

     #"min_child_weight" : [ 1, 3, 5, 7 ],

     #"gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],

     #"colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]

     }



clf = xgb.XGBClassifier(**params)



grid = GridSearchCV(clf,

                    parameters, n_jobs=4,

                    scoring="roc_auc",

                    cv=5)



X_train = train_new[['sex_enc','age_approx_scaled','anatom_enc']]

y_train = train_new.target



# uncomment to find best parameters

#grid.fit(X_train, y_train, verbose=True)



#Print best parameters

#grid.best_params_, grid.best_score_
test_data = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
test_data.anatom_site_general_challenge.value_counts()
anatom_type = ['torso', 'lower extremity', 'upper extremity', 'head/neck', 'palms/soles', 'oral/genital']
from tqdm import tqdm



pt_id = []

for i in tqdm(range(test_data.shape[0])):

    row = test_data.loc[i]

    #print(row)

    if row.anatom_site_general_challenge not in anatom_type:

        pt_id.append(row.patient_id)
len(pt_id)
#grouped = test_data.groupby(test_data.patient_id)

#for name,group in grouped:

#    if name in pt_id:

#        print(name)

#        print (group.anatom_site_general_challenge)
# Repalcing all null values by torso as it is most occuring

test_data.anatom_site_general_challenge.fillna('torso',inplace=True)
# Transforming data ,using same transformer created for train

test_data['sex_enc']   = sex_enc.transform(test_data.sex.astype('str'))

test_data['anatom_enc']= anatom_enc.transform(test_data.anatom_site_general_challenge.astype('str'))

test_data['age_approx_scaled'] = scaler.transform(test_data[['age_approx']])
pred_xgb = xgb_model.predict(test_data[['sex_enc','age_approx_scaled', 'anatom_enc']])
sub = pd.DataFrame({'image_name':test_data.image_name.values, 'target':pred_xgb})

sub.to_csv('submission_xgb.csv',index = False)

sub.head()