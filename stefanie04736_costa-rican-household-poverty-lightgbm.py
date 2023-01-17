# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import get_scorer
import sklearn

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df.head()
categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
categorical_columns
def replace_yes_no(df):
    df[['dependency', 'edjefe', 'edjefa']] = df[['dependency','edjefe', 'edjefa']].replace({'yes':1, 'no':1}).astype(float)
    return df

df = replace_yes_no(df)
df_test = replace_yes_no(df_test)
def tablets(df):
    df['v18q1'][df['v18q1'].isnull()] = 0
    df = df.drop(columns = 'v18q')
    return df

df = tablets(df)
df_test = tablets(df_test)
def escolari(df):
    escolari_mean = df.groupby(['idhogar'], as_index = False)['escolari'].mean().rename(columns = {'mean': 'escolari_mean'})
    escolari_mean.columns = ['idhogar', 'escolari_mean']

    escolari_max = df.groupby(['idhogar'], as_index = False)['escolari'].max().rename(columns = {'max': 'escolari_max'})
    escolari_max.columns = ['idhogar', 'escolari_max']

    df = df.merge(escolari_mean, how = 'left', on = 'idhogar')
    df = df.merge(escolari_max, how = 'left', on = 'idhogar')
    
    return df

df = escolari(df)
df_test = escolari(df_test)
def water_provision(df):
    df['water_prov'] = 0
    df.loc[df['abastaguadentro'] == 1, 'water_prov'] = 2
    df.loc[df['abastaguafuera'] == 1, 'water_prov'] = 1
    df.loc[df['abastaguano'] == 1, 'water_prov'] = 0
    df = df.drop(columns = ['abastaguadentro','abastaguafuera', 'abastaguano'])
    return df

df = water_provision(df)
df_test = water_provision(df_test)
def walls_roof_floor(df):
    df['walls'] = 0
    df.loc[df['epared1'] == 1, 'walls'] = 1
    df.loc[df['epared2'] == 1, 'walls'] = 2
    df.loc[df['epared3'] == 1, 'walls'] = 3
    
    df['roof'] = 0
    df.loc[df['etecho1'] == 1, 'roof'] = 1
    df.loc[df['etecho2'] == 1, 'roof'] = 2
    df.loc[df['etecho3'] == 1, 'roof'] = 3
        
    df['floor'] = 0
    df.loc[df['eviv1'] == 1, 'floor'] = 1
    df.loc[df['eviv2'] == 1, 'floor'] = 2
    df.loc[df['eviv3'] == 1, 'floor'] = 3

    df = df.drop(columns = ['epared1','epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3'])
    
    return df

df = walls_roof_floor(df)
df_test = walls_roof_floor(df_test)
def education_level(df):
    df['education'] = 0
    df.loc[df['instlevel1'] == 1, 'education'] = 1
    df.loc[df['instlevel2'] == 1, 'education'] = 2
    df.loc[df['instlevel3'] == 1, 'education'] = 3
    df.loc[df['instlevel4'] == 1, 'education'] = 4
    df.loc[df['instlevel5'] == 1, 'education'] = 5
    df.loc[df['instlevel6'] == 1, 'education'] = 6
    df.loc[df['instlevel7'] == 1, 'education'] = 7
    df.loc[df['instlevel8'] == 1, 'education'] = 8
    df.loc[df['instlevel9'] == 1, 'education'] = 9

    df = df.drop(columns = ['instlevel1','instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9'])
    
    return df

df = education_level(df)
df_test = education_level(df_test)
def tipovivi(df):
    df['tipovivi'] = 0
    df.loc[df['tipovivi1'] == 1, 'tipovivi'] = 1
    df.loc[df['tipovivi2'] == 1, 'tipovivi'] = 2
    df.loc[df['tipovivi3'] == 1, 'tipovivi'] = 3
    df.loc[df['tipovivi4'] == 1, 'tipovivi'] = 4
    df.loc[df['tipovivi5'] == 1, 'tipovivi'] = 5
    
    df = df.drop(columns = ['tipovivi1','tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5'])
    
    return df

df = tipovivi(df)
df_test = tipovivi(df_test)
def rubbish(df):
    df['rubbish'] = 0
    df.loc[df['elimbasu1'] == 1, 'rubbish'] = 1
    df.loc[df['elimbasu2'] == 1, 'rubbish'] = 2
    df.loc[df['elimbasu3'] == 1, 'rubbish'] = 3
    df.loc[df['elimbasu4'] == 1, 'rubbish'] = 4
    df.loc[df['elimbasu5'] == 1, 'rubbish'] = 5
    df.loc[df['elimbasu6'] == 1, 'rubbish'] = 0
    
    df = df.drop(columns = ['elimbasu1','elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6'])
    
    return df

df = rubbish(df)
df_test = rubbish(df_test)
def energy(df):
    df['energy'] = 0
    df.loc[df['energcocinar1'] == 1, 'energy'] = 1
    df.loc[df['energcocinar2'] == 1, 'energy'] = 2
    df.loc[df['energcocinar3'] == 1, 'energy'] = 3
    df.loc[df['energcocinar4'] == 1, 'energy'] = 4
    
    df = df.drop(columns = ['energcocinar1','energcocinar2', 'energcocinar3', 'energcocinar4'])
    
    return df

df = energy(df)
df_test = energy(df_test)
def toilet(df):
    df['toilet'] = 0
    df.loc[df['sanitario1'] == 1, 'toilet'] = 1
    df.loc[df['sanitario5'] == 1, 'toilet'] = 2
    df.loc[df['sanitario6'] == 1, 'toilet'] = 3
    df.loc[df['sanitario3'] == 1, 'toilet'] = 4
    df.loc[df['sanitario2'] == 1, 'toilet'] = 5
       
    df = df.drop(columns = ['sanitario1','sanitario2', 'sanitario3', 'sanitario5', 'sanitario6'])
    
    return df

df = toilet(df)
df_test = toilet(df_test)
def new_variables(df):
    df['rent_by_hhsize'] = df['v2a1'] / df['hhsize'] # rent by household size
    df['rent_by_people'] = df['v2a1'] / df['r4t3'] # rent by people in household
    df['rent_by_rooms'] = df['v2a1'] / df['rooms'] # rent by number of rooms
    df['rent_by_living'] = df['v2a1'] / df['tamviv'] # rent by number of persons living in the household
    df['rent_by_minor'] = df['v2a1'] / df['hogar_nin']
    df['rent_by_adult'] = df['v2a1'] / df['hogar_adul']
    df['children_by_adults'] = df['hogar_nin'] / df['hogar_adul']
    df['house_quali'] = df['walls'] + df['roof'] + df['floor']
    df['tablets_by_adults'] = df['v18q1'] / df['hogar_adul'] # number of tablets per adults
    df['ratio_nin'] = df['hogar_nin'] / df['hogar_adul'] # ratio children to adults
    return df

df = new_variables(df)
df_test = new_variables(df_test)
df.head(15)
# Use all columns as features except Ids and Target
feats = [f for f in df.columns if f not in ['Id','Target','idhogar']]

# 10 folds
folds = StratifiedKFold(n_splits= 10, shuffle=True, random_state=1054)

# matrix for predictions
preds = np.zeros((df_test.shape[0], 4))

# iterate through folds
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df[feats], df['Target'])):
    print('Fold ', n_fold)
    train_x, train_y = df.iloc[train_idx], df['Target'].iloc[train_idx]
    valid_x, valid_y = df.iloc[valid_idx], df['Target'].iloc[valid_idx]
    
    # eliminate unnecessary features
    train_x = train_x[feats]
    valid_x = valid_x[feats]
    
    # create and fit model
    gbm = lgb.LGBMClassifier(n_jobs=4, random_state=0, class_weight='balanced', num_leaves = 100, learning_rate = 0.1, early_stopping_rounds = 200)
    gbm.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
                    verbose= 100, eval_metric = 'multi_error')
    
    # mean of predictions for argmax later on 
    preds += gbm.predict_proba(df_test[feats]) / folds.n_splits
# predicted class is the one with the highest prediction value
pred_maj = np.argmax(preds, axis = 1) + 1
df_test['Target'] = pred_maj.astype(int)
df_test[['Id', 'Target']].to_csv('submission_180831_lgbm.csv', index= False)
df_test['Target'].value_counts()