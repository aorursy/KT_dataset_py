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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
input_path = '/kaggle/input/porto-seguro-safe-driver-prediction/'

df = pd.read_csv(input_path+'train.csv')

df_test = pd.read_csv(input_path+'test.csv')
id_test = df_test['id']
df.shape
target = df['target']
target.value_counts()
df.drop(columns=['target'],inplace=True)
import pickle

df_metadata = pickle.load(open('/kaggle/input/pssdpickledfmetedatapickle/df_metedata_pickle','rb'))
df_metadata
def preprocessing(df):

    df.replace(to_replace=-1,value=np.nan,inplace=True)

    for col in df.columns:

        #Dropping Insignificant Columns

        if df_metadata.loc[col,'Dropped']:

            df.drop(columns=[col],inplace=True)

            continue

        #Filling Missing Values

        df[col].fillna(df_metadata.loc[col,'Missing'],inplace=True)

        #Changing the datatype of columns

        if (df_metadata.loc[col,'DTypes'] == 'Categorical') or (df_metadata.loc[col,'DTypes'] == 'Ordinal'):

            df[col] = df[col].astype('category')
df.shape
preprocessing(df)

preprocessing(df_test)
df.isnull().sum()
df_test.isnull().sum()
def outlier_processing(df,df_test):

    for col in df.columns:

        if df[col].dtype.name != 'category':

            first_quartile, third_quartile = np.percentile(df[col],[25,75])

            first_percetnile, ninetynine_percentile = np.percentile(df[col],[1,99])

            IQR = third_quartile - first_quartile

            lower_bound = first_quartile - (1.5*IQR)

            upper_bound = third_quartile + (1.5*IQR)

            df[col].loc[df[col]>upper_bound] = ninetynine_percentile

            df_test[col].loc[df_test[col]>upper_bound] = ninetynine_percentile

            df[col].loc[df[col]<lower_bound] = first_percetnile

            df_test[col].loc[df_test[col]<lower_bound] = first_percetnile

        
#outlier_processing(df,df_test)
ordinal_columns = [col for col in df.columns if df_metadata.loc[col,'DTypes'] == 'Ordinal' and df[col].nunique() > 2]
categorical_columns_great_2 = [col for col in df.columns if df_metadata.loc[col,'DTypes'] == 'Categorical' and df[col].nunique() > 2]
from sklearn.preprocessing import LabelEncoder

for col in ordinal_columns:

    label_encode = LabelEncoder()

    df[col+'label'] = label_encode.fit_transform(df[col])

    df_test[col+'label'] = label_encode.transform(df_test[col])

    df.drop(columns=[col],inplace=True)

    df_test.drop(columns=[col],inplace=True)
df = pd.get_dummies(df,prefix=col,columns=categorical_columns_great_2,drop_first=True)

df_test = pd.get_dummies(df_test,columns=categorical_columns_great_2,prefix=col,drop_first=True)
df.shape
df_test.shape
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_train_scale = scaler.fit_transform(df)
df_test_scale = scaler.transform(df_test)
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
def gini(actual,pred,cmpcol = 0,sortcol = 1):

    assert( len(actual) == len(pred) )

    All = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)

    All = All[ np.lexsort((All[:,2], -1*All[:,1])) ]

    totAllosses = All[:,0].sum()

    giniSum = All[:,0].cumsum().sum() / totAllosses

    giniSum -= (len(actual) + 1) / 2.

    return giniSum / len(actual)



def gini_normalized(a, p):

    return gini(a, p) / gini(a, a)



def gini_xgb(preds, dtrain):

    labels = dtrain.get_label()

    gini_score = gini_normalized(labels, preds)

    return 'gini', gini_score
import xgboost as xgb
params = {

    'min_child_weight': 10.0,

    'objective': 'binary:logistic',

    'max_depth': 7,

    'max_delta_step': 1.8,

    'colsample_bytree': 0.4,

    'subsample': 0.8,

    'eta': 0.025,

    'gamma': 0.65,

    'num_boost_round' : 1000

    }
model_ensemble = []

for i,(train_index, valid_index) in enumerate(skf.split(df_train_scale,target)):

    print('[FOLD %d/%d]'%(i+1,5))

    X_train,X_valid = df_train_scale[train_index],df_train_scale[valid_index]

    y_train,y_valid = target.loc[train_index],target.loc[valid_index]

    #Convert Data in to XGBoost format

    df_train_xgb = xgb.DMatrix(X_train,y_train)

    df_valid_xgb = xgb.DMatrix(X_valid,y_valid)

    valid_list = [(df_train_xgb, 'train'), (df_valid_xgb, 'valid')]

    xgb_model = xgb.train(params, df_train_xgb, 3000, valid_list, feval=gini_xgb, maximize=True, early_stopping_rounds=70,verbose_eval=100)

    model_ensemble.append(xgb_model)
predict_proba = 0

df_test_xgb = xgb.DMatrix(df_test_scale)

for i, model in enumerate(model_ensemble):

    print('[FOLD %d/%d Prediciton:]'%(i+1,5))

    predictions = xgb_model.predict(df_test_xgb)

    predict_proba += predictions

predict_proba = predict_proba/5
submit = pd.DataFrame({'id':id_test,'target':predict_proba})

submit.to_csv('xgb_porto.csv',index=False) 

submit.head()