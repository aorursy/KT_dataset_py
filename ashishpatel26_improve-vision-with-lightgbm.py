# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 300)
%matplotlib inline

sns.set(style='white', context='notebook', palette='deep')
mycols = ["#66c2ff", "#5cd6d6", "#00cc99", "#85e085", "#ffd966", "#ffb366", "#ffb3b3", "#dab3ff", "#c2c2d6"]
sns.set_palette(palette = mycols, n_colors = 4)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from contextlib import contextmanager
import time

from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import warnings
import gc
warnings.simplefilter(action='ignore', category=FutureWarning)
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
def load_data():
    train_set = pd.read_csv('../input/costa-rican-household-poverty-prediction/train.csv')
    test_set = pd.read_csv('../input/costa-rican-household-poverty-prediction/test.csv')
    print(f'train set has {train_set.shape[0]} rows, and {train_set.shape[1]} features')
    print(f'test set has {test_set.shape[0]} rows, and {test_set.shape[1]} features')
    #Let's take a look at target
    target = train_set['Target']
    target.value_counts(normalize=True)
    return (train_set,test_set,target)
def handle_missing_value(train_set,test_set):
    '''
    Handle the missing value of train and test set and return the 
    '''
    # Train set Missing value
    data_na = train_set.isnull().sum().values / train_set.shape[0] *100
    df_na = pd.DataFrame(data_na, index=train_set.columns, columns=['Count'])
    df_na = df_na.sort_values(by=['Count'], ascending=False)
    missing_value_count = df_na[df_na['Count']>0].shape[0]
    print(f'We got {missing_value_count} rows which have missing value in train set ')
    df_na.head(6)
    # Test set Missing value
#     data_na1 = test_set.isnull().sum().values / test_set.shape[0] *100
#     df_na1 = pd.DataFrame(data_na1, index=test_set.columns, columns=['Count'])
#     df_na1 = df_na1.sort_values(by=['Count'], ascending=False)
#     missing_value_count1 = df_na1[df_na1['Count']>0].shape[0]
#     print(f'We got {missing_value_count} rows which have missing value in test set ')
#     df_na1.head(6)
    data_na = train_set.isnull().sum().values / train_set.shape[0] *100
    df_na = pd.DataFrame(data_na, index=train_set.columns, columns=['Count'])
    df_na = df_na.sort_values(by=['Count'], ascending=False)

    missing_value_count = df_na[df_na['Count']>0].shape[0]

    print(f'We got {missing_value_count} rows which have missing value in test set ')
    df_na.head(6)

    #Fill na
    def repalce_v18q1(x):
        if x['v18q'] == 0:
            return x['v18q']
        else:
            return x['v18q1']

    train_set['v18q1'] = train_set.apply(lambda x : repalce_v18q1(x),axis=1)
    test_set['v18q1'] = test_set.apply(lambda x : repalce_v18q1(x),axis=1)

    train_set['v2a1'] = train_set['v2a1'].fillna(value=train_set['tipovivi3'])
    test_set['v2a1'] = test_set['v2a1'].fillna(value=test_set['tipovivi3'])
    
    return (train_set,test_set)
def feature_engineering(train_set,test_set):
    '''
    Feature Engineering and Feature generation
    '''
    cols = ['edjefe', 'edjefa']
    train_set[cols] = train_set[cols].replace({'no': 0, 'yes':1}).astype(float)
    test_set[cols] = test_set[cols].replace({'no': 0, 'yes':1}).astype(float)
    
    train_set['roof_waste_material'] = np.nan
    test_set['roof_waste_material'] = np.nan
    train_set['electricity_other'] = np.nan
    test_set['electricity_other'] = np.nan

    def fill_roof_exception(x):
        if (x['techozinc'] == 0) and (x['techoentrepiso'] == 0) and (x['techocane'] == 0) and (x['techootro'] == 0):
            return 1
        else:
            return 0

    def fill_no_electricity(x):
        if (x['public'] == 0) and (x['planpri'] == 0) and (x['noelec'] == 0) and (x['coopele'] == 0):
            return 1
        else:
            return 0

    train_set['roof_waste_material'] = train_set.apply(lambda x : fill_roof_exception(x),axis=1)
    test_set['roof_waste_material'] = test_set.apply(lambda x : fill_roof_exception(x),axis=1)
    train_set['electricity_other'] = train_set.apply(lambda x : fill_no_electricity(x),axis=1)
    test_set['electricity_other'] = test_set.apply(lambda x : fill_no_electricity(x),axis=1)
    
    train_set['adult'] = train_set['hogar_adul'] - train_set['hogar_mayor']
    train_set['dependency_count'] = train_set['hogar_nin'] + train_set['hogar_mayor']
    train_set['dependency'] = train_set['dependency_count'] / train_set['adult']
    train_set['child_percent'] = train_set['hogar_nin']/train_set['hogar_total']
    train_set['elder_percent'] = train_set['hogar_mayor']/train_set['hogar_total']
    train_set['adult_percent'] = train_set['hogar_adul']/train_set['hogar_total']
    test_set['adult'] = test_set['hogar_adul'] - test_set['hogar_mayor']
    test_set['dependency_count'] = test_set['hogar_nin'] + test_set['hogar_mayor']
    test_set['dependency'] = test_set['dependency_count'] / test_set['adult']
    test_set['child_percent'] = test_set['hogar_nin']/test_set['hogar_total']
    test_set['elder_percent'] = test_set['hogar_mayor']/test_set['hogar_total']
    test_set['adult_percent'] = test_set['hogar_adul']/test_set['hogar_total']

    train_set['rent_per_adult'] = train_set['v2a1']/train_set['hogar_adul']
    train_set['rent_per_person'] = train_set['v2a1']/train_set['hhsize']
    test_set['rent_per_adult'] = test_set['v2a1']/test_set['hogar_adul']
    test_set['rent_per_person'] = test_set['v2a1']/test_set['hhsize']

    train_set['overcrowding_room_and_bedroom'] = (train_set['hacdor'] + train_set['hacapo'])/2
    test_set['overcrowding_room_and_bedroom'] = (test_set['hacdor'] + test_set['hacapo'])/2

    train_set['no_appliances'] = train_set['refrig'] + train_set['computer'] + train_set['television']
    test_set['no_appliances'] = test_set['refrig'] + test_set['computer'] + test_set['television']

    train_set['r4h1_percent_in_male'] = train_set['r4h1'] / train_set['r4h3']
    train_set['r4m1_percent_in_female'] = train_set['r4m1'] / train_set['r4m3']
    train_set['r4h1_percent_in_total'] = train_set['r4h1'] / train_set['hhsize']
    train_set['r4m1_percent_in_total'] = train_set['r4m1'] / train_set['hhsize']
    train_set['r4t1_percent_in_total'] = train_set['r4t1'] / train_set['hhsize']
    test_set['r4h1_percent_in_male'] = test_set['r4h1'] / test_set['r4h3']
    test_set['r4m1_percent_in_female'] = test_set['r4m1'] / test_set['r4m3']
    test_set['r4h1_percent_in_total'] = test_set['r4h1'] / test_set['hhsize']
    test_set['r4m1_percent_in_total'] = test_set['r4m1'] / test_set['hhsize']
    test_set['r4t1_percent_in_total'] = test_set['r4t1'] / test_set['hhsize']

    train_set['rent_per_room'] = train_set['v2a1']/train_set['rooms']
    train_set['bedroom_per_room'] = train_set['bedrooms']/train_set['rooms']
    train_set['elder_per_room'] = train_set['hogar_mayor']/train_set['rooms']
    train_set['adults_per_room'] = train_set['adult']/train_set['rooms']
    train_set['child_per_room'] = train_set['hogar_nin']/train_set['rooms']
    train_set['male_per_room'] = train_set['r4h3']/train_set['rooms']
    train_set['female_per_room'] = train_set['r4m3']/train_set['rooms']
    train_set['room_per_person_household'] = train_set['hhsize']/train_set['rooms']

    test_set['rent_per_room'] = test_set['v2a1']/test_set['rooms']
    test_set['bedroom_per_room'] = test_set['bedrooms']/test_set['rooms']
    test_set['elder_per_room'] = test_set['hogar_mayor']/test_set['rooms']
    test_set['adults_per_room'] = test_set['adult']/test_set['rooms']
    test_set['child_per_room'] = test_set['hogar_nin']/test_set['rooms']
    test_set['male_per_room'] = test_set['r4h3']/test_set['rooms']
    test_set['female_per_room'] = test_set['r4m3']/test_set['rooms']
    test_set['room_per_person_household'] = test_set['hhsize']/test_set['rooms']

    train_set['rent_per_bedroom'] = train_set['v2a1']/train_set['bedrooms']
    train_set['edler_per_bedroom'] = train_set['hogar_mayor']/train_set['bedrooms']
    train_set['adults_per_bedroom'] = train_set['adult']/train_set['bedrooms']
    train_set['child_per_bedroom'] = train_set['hogar_nin']/train_set['bedrooms']
    train_set['male_per_bedroom'] = train_set['r4h3']/train_set['bedrooms']
    train_set['female_per_bedroom'] = train_set['r4m3']/train_set['bedrooms']
    train_set['bedrooms_per_person_household'] = train_set['hhsize']/train_set['bedrooms']

    test_set['rent_per_bedroom'] = test_set['v2a1']/test_set['bedrooms']
    test_set['edler_per_bedroom'] = test_set['hogar_mayor']/test_set['bedrooms']
    test_set['adults_per_bedroom'] = test_set['adult']/test_set['bedrooms']
    test_set['child_per_bedroom'] = test_set['hogar_nin']/test_set['bedrooms']
    test_set['male_per_bedroom'] = test_set['r4h3']/test_set['bedrooms']
    test_set['female_per_bedroom'] = test_set['r4m3']/test_set['bedrooms']
    test_set['bedrooms_per_person_household'] = test_set['hhsize']/test_set['bedrooms']

    train_set['tablet_per_person_household'] = train_set['v18q1']/train_set['hhsize']
    train_set['phone_per_person_household'] = train_set['qmobilephone']/train_set['hhsize']
    test_set['tablet_per_person_household'] = test_set['v18q1']/test_set['hhsize']
    test_set['phone_per_person_household'] = test_set['qmobilephone']/test_set['hhsize']

    train_set['age_12_19'] = train_set['hogar_nin'] - train_set['r4t1']
    test_set['age_12_19'] = test_set['hogar_nin'] - test_set['r4t1']    

    train_set['escolari_age'] = train_set['escolari']/train_set['age']
    test_set['escolari_age'] = test_set['escolari']/test_set['age']

    train_set['rez_esc_escolari'] = train_set['rez_esc']/train_set['escolari']
    train_set['rez_esc_r4t1'] = train_set['rez_esc']/train_set['r4t1']
    train_set['rez_esc_r4t2'] = train_set['rez_esc']/train_set['r4t2']
    train_set['rez_esc_r4t3'] = train_set['rez_esc']/train_set['r4t3']
    train_set['rez_esc_age'] = train_set['rez_esc']/train_set['age']
    test_set['rez_esc_escolari'] = test_set['rez_esc']/test_set['escolari']
    test_set['rez_esc_r4t1'] = test_set['rez_esc']/test_set['r4t1']
    test_set['rez_esc_r4t2'] = test_set['rez_esc']/test_set['r4t2']
    test_set['rez_esc_r4t3'] = test_set['rez_esc']/test_set['r4t3']
    test_set['rez_esc_age'] = test_set['rez_esc']/test_set['age']
    
    train_set['dependency'] = train_set['dependency'].replace({np.inf: 0})
    test_set['dependency'] = test_set['dependency'].replace({np.inf: 0})

    print(f'train set has {train_set.shape[0]} rows, and {train_set.shape[1]} features')
    print(f'test set has {test_set.shape[0]} rows, and {test_set.shape[1]} features')
    
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    aggr_mean_list = ['rez_esc', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 'parentesco2',
                 'parentesco3', 'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9', 'parentesco10', 'parentesco11', 'parentesco12',
                 'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9',]

    other_list = ['escolari', 'age', 'escolari_age']

    for item in aggr_mean_list:
        group_train_mean = train_set[item].groupby(train_set['idhogar']).mean()
        group_test_mean = test_set[item].groupby(test_set['idhogar']).mean()
        new_col = item + '_aggr_mean'
        df_train[new_col] = group_train_mean
        df_test[new_col] = group_test_mean

    for item in other_list:
        for function in ['mean','std','min','max','sum']:
            group_train = train_set[item].groupby(train_set['idhogar']).agg(function)
            group_test = test_set[item].groupby(test_set['idhogar']).agg(function)
            new_col = item + '_' + function
            df_train[new_col] = group_train
            df_test[new_col] = group_test

    print(f'new aggregate train set has {df_train.shape[0]} rows, and {df_train.shape[1]} features')
    print(f'new aggregate test set has {df_test.shape[0]} rows, and {df_test.shape[1]} features')
    
    df_test = df_test.reset_index()
    df_train = df_train.reset_index()

    train_agg = pd.merge(train_set, df_train, on='idhogar')
    test = pd.merge(test_set, df_test, on='idhogar')

    #fill all na as 0
    train_agg.fillna(value=0, inplace=True)
    test.fillna(value=0, inplace=True)
    print(f'new train set has {train_agg.shape[0]} rows, and {train_agg.shape[1]} features')
    print(f'new test set has {test.shape[0]} rows, and {test.shape[1]} features')
    
    train = train_agg.query('parentesco1==1')
    
    submission = test[['Id']]

    #Remove useless feature to reduce dimension
    train.drop(columns=['idhogar','Id', 'tamhog', 'agesq', 'hogar_adul', 'SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned'], inplace=True)
    test.drop(columns=['idhogar','Id', 'tamhog', 'agesq', 'hogar_adul', 'SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned'], inplace=True)

    correlation = train.corr()
    correlation = correlation['Target'].sort_values(ascending=False)
    print(f'The most 20 positive feature: \n{correlation.head(20)}')
    print('*'*50)

    print(f'The most 20 negative feature: \n{correlation.tail(20)}')
    
    return (train,test,train_set,test_set)
def model_training(train,test):
    #parameter value is copied from 
    y = train['Target']
    train.drop(columns=['Target'], inplace=True)
    clf = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.1, objective='multiclass',
                                 random_state=None, silent=True, metric='None', 
                                 n_jobs=4, n_estimators=5500, class_weight='balanced',
                                 colsample_bytree =  0.89, min_child_samples = 90, num_leaves = 56, subsample = 0.96)
    
    kfold = 7
    kf = StratifiedKFold(n_splits=kfold, shuffle=True)

    predicts_result = []
    for train_index, test_index in kf.split(train, y):
        print("###")
        X_train, X_val = train.iloc[train_index], train.iloc[test_index]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)],early_stopping_rounds=100)
        predicts_result.append(clf.predict(test))
    
    return (predicts_result,clf)
       

def feature_plot(predicts_result,clf,train):
    indices = np.argsort(clf.feature_importances_)[::-1]
    indices = indices[:75]
    # Visualise these with a barplot
    plt.subplots(figsize=(20, 15))
    g = sns.barplot(y=train.columns[indices], x = clf.feature_importances_[indices], orient='h', palette = mycols)
    g.set_xlabel("Relative importance",fontsize=12)
    g.set_ylabel("Features",fontsize=12)
    g.tick_params(labelsize=9)
    g.set_title("LightGBM feature importance");
def main(debug = False):
    with timer("Dataset is Reading... "):
        train_set,test_set,target = load_data()
        #outlier in test set which rez_esc is 99.0
        test_set.loc[test_set['rez_esc'] == 99.0 , 'rez_esc'] = 5
        gc.collect()
    with timer("Handle Missing Value..."):
        train_set,test_set = handle_missing_value(train_set,test_set)
        gc.collect()
    with timer("Feature Engineering..."):
        train,test,train_set,test_set = feature_engineering(train_set,test_set)
        gc.collect()
    with timer("Automatic Model Tuning..."):
        predicts_result,clf = model_training(train,test)
        feature_plot(predicts_result,clf,train)
        gc.collect()
    with timer("Final Submission"):
        submission = pd.read_csv("../input/costa-rican-household-poverty-prediction/sample_submission.csv")
        submission['Target'] = np.array(predicts_result).mean(axis=0).round().astype(int)
        submission.to_csv('submission.csv', index = False)
if __name__ == "__main__":
    with timer("Full model run"):
        main(debug= False)
sub = pd.read_csv("../input/modeltune/Ashish4321.csv")
sub.to_csv("submission1.csv", index = False)
