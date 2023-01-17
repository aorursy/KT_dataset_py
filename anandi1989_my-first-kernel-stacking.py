import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 300)
%matplotlib inline

sns.set(style='white', context='notebook', palette='deep')

mycols = ["#66c2ff", "#5cd6d6", "#00cc99", "#85e085", "#ffd966", "#ffb366", "#ffb3b3", "#dab3ff", "#c2c2d6"]
sns.set_palette(palette = mycols, n_colors = 4)


from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

train_set = pd.read_csv('../input/train.csv')
test_set = pd.read_csv('../input/test.csv')

#print(f'train set has {train_set.shape[0]} rows, and {train_set.shape[1]} features')
#print(f'test set has {test_set.shape[0]} rows, and {test_set.shape[1]} features')
#Let's take a look at target
target = train_set['Target']
target.value_counts(normalize=True)
#outlier in test set which rez_esc is 99.0
test_set.loc[test_set['rez_esc'] == 99.0 , 'rez_esc'] = 5
data_na = train_set.isnull().sum().values / train_set.shape[0] *100
df_na = pd.DataFrame(data_na, index=train_set.columns, columns=['Count'])
df_na = df_na.sort_values(by=['Count'], ascending=False)

missing_value_count = df_na[df_na['Count']>0].shape[0]

#print(f'We got {missing_value_count} rows which have missing value in train set ')
df_na.head(6)
data_na = train_set.isnull().sum().values / train_set.shape[0] *100
df_na = pd.DataFrame(data_na, index=train_set.columns, columns=['Count'])
df_na = df_na.sort_values(by=['Count'], ascending=False)

missing_value_count = df_na[df_na['Count']>0].shape[0]

#print(f'We got {missing_value_count} rows which have missing value in test set ')
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

#print(f'train set has {train_set.shape[0]} rows, and {train_set.shape[1]} features')
#print(f'test set has {test_set.shape[0]} rows, and {test_set.shape[1]} features')
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

#print(f'new aggregate train set has {df_train.shape[0]} rows, and {df_train.shape[1]} features')
#print(f'new aggregate test set has {df_test.shape[0]} rows, and {df_test.shape[1]} features')
df_test = df_test.reset_index()
df_train = df_train.reset_index()

train_agg = pd.merge(train_set, df_train, on='idhogar')
test = pd.merge(test_set, df_test, on='idhogar')

#fill all na as 0
train_agg.fillna(value=0, inplace=True)
test.fillna(value=0, inplace=True)
#print(f'new train set has {train_agg.shape[0]} rows, and {train_agg.shape[1]} features')
#print(f'new test set has {test.shape[0]} rows, and {test.shape[1]} features')
#According to data descriptions,ONLY the heads of household are used in scoring. /
#All household members are included in test + the sample submission, but only heads of households are scored.
train = train_agg.query('parentesco1==1')
submission = test[['Id']]

#Remove useless feature to reduce dimension
train.drop(columns=['idhogar','Id', 'tamhog', 'agesq', 'hogar_adul', 'SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned'], inplace=True)
test.drop(columns=['idhogar','Id', 'tamhog', 'agesq', 'hogar_adul', 'SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned'], inplace=True)

correlation = train.corr()
correlation = correlation['Target'].sort_values(ascending=False)
#print(f'The most 20 positive feature: \n{correlation.head(20)}')
#print('*'*50)

#print(f'The most 20 negative feature: \n{correlation.tail(20)}')
y = train['Target']

train.drop(columns=['Target'], inplace=True)
class Ensemble(object):    
    def __init__(self, mode, n_splits, stacker_2, stacker_1, base_models):
        self.mode = mode
        self.n_splits = n_splits
        self.stacker_2 = stacker_2
        self.stacker_1 = stacker_1
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)


        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, 
                                                             random_state=2016).split(X, y))
        
        OOF_columns = []

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):                
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]

                print ("Fit %s_%d fold %d" % (str(clf).split("(")[0], i+1, j+1))
                clf.fit(X_train, y_train)

                S_train[test_idx, i] = clf.predict(X_holdout)  
                S_test_i[:, j] = clf.predict(T)
            S_test[:, i] = S_test_i.mean(axis=1)
            
            #print("  Base model_%d score: %.5f\n" % (i+1, roc_auc_score(y, S_train[:,i])))
        
            OOF_columns.append('Base model_'+str(i+1))
        OOF_S_train = pd.DataFrame(S_train, columns = OOF_columns)
        print('\n')
        print('Correlation between out-of-fold predictions from Base models:')
        print('\n')
        print(OOF_S_train.corr())
        print('\n')
            
        
        if self.mode==1:
            
            folds_2 = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                                                                   random_state=2016).split(S_train, y))
            
            OOF_columns = []

            S_train_2 = np.zeros((S_train.shape[0], len(self.stacker_1)))
            S_test_2 = np.zeros((S_test.shape[0], len(self.stacker_1)))
            
            for i, clf in enumerate(self.stacker_1):
            
                S_test_i_2 = np.zeros((S_test.shape[0], self.n_splits))

                for j, (train_idx, test_idx) in enumerate(folds_2):
                    X_train_2 = S_train[train_idx]
                    y_train_2 = y[train_idx]
                    X_holdout_2 = S_train[test_idx]

                    print ("Fit %s_%d fold %d" % (str(clf).split("(")[0], i+1, j+1))
                    clf.fit(X_train_2, y_train_2)
                                 
                    S_train_2[test_idx, i] = clf.predict(X_holdout_2)
                    S_test_i_2[:, j] = clf.predict(S_test)
                S_test_2[:, i] = S_test_i_2.mean(axis=1)
                
                #print("  1st level model_%d score: %.5f\n"%(i+1,roc_auc_score(y, S_train_2.mean(axis=1))))
                
                OOF_columns.append('1st level model_'+str(i+1))
            OOF_S_train = pd.DataFrame(S_train_2, columns = OOF_columns)
            print('\n')
            print('Correlation between out-of-fold predictions from 1st level models:')
            print('\n')
            print(OOF_S_train.corr())
            print('\n')


        if self.mode==2:
            
            WOC_columns = []
        
            S_train_2 = np.zeros((S_train.shape[0], len(self.stacker_1)))
            S_test_2 = np.zeros((S_test.shape[0], len(self.stacker_1)))
               
            for i, clf in enumerate(self.stacker_1):
            
                S_train_i_2= np.zeros((S_train.shape[0], S_train.shape[1]))
                S_test_i_2 = np.zeros((S_test.shape[0], S_train.shape[1]))
                                       
                for j in range(S_train.shape[1]):
                                
                    S_tr = S_train[:,np.arange(S_train.shape[1])!=j]
                    S_te = S_test[:,np.arange(S_test.shape[1])!=j]
                                               
                    print ("Fit %s_%d subset %d" % (str(clf).split("(")[0], i+1, j+1))
                    clf.fit(S_tr, y)

                    S_train_i_2[:, j] = clf.predict(S_tr)
                    S_test_i_2[:, j] = clf.predict(S_te)
                S_train_2[:, i] = S_train_i_2.mean(axis=1)    
                S_test_2[:, i] = S_test_i_2.mean(axis=1)
            
                #print("  1st level model_%d score: %.5f\n"%(i+1,roc_auc_score(y, S_train_2.mean(axis=1))))
                
                WOC_columns.append('1st level model_'+str(i+1))
            WOC_S_train = pd.DataFrame(S_train_2, columns = WOC_columns)
            print('\n')
            print('Correlation between without-one-column predictions from 1st level models:')
            print('\n')
            print(WOC_S_train.corr())
            print('\n')
            
            
        try:
            num_models = len(self.stacker_2)
            if self.stacker_2==(et_model):
                num_models=1
        except TypeError:
            num_models = len([self.stacker_2])
            
        if num_models==1:
                
            print ("Fit %s for final\n" % (str(self.stacker_2).split("(")[0]))
            self.stacker_2.fit(S_train_2, y)
            
            stack_res = self.stacker_2.predict(S_test_2)
        
            stack_score = self.stacker_2.predict(S_train_2)
            #print("2nd level model final score: %.5f" % (roc_auc_score(y, stack_score)))
                
        else:
            
            F_columns = []
            
            stack_score = np.zeros((S_train_2.shape[0], len(self.stacker_2)))
            res = np.zeros((S_test_2.shape[0], len(self.stacker_2)))
            
            for i, clf in enumerate(self.stacker_2):
                
                print ("Fit %s_%d" % (str(clf).split("(")[0], i+1))
                clf.fit(S_train_2, y)
                
                stack_score[:, i] = clf.predict(S_train_2)
                #print("  2nd level model_%d score: %.5f\n"%(i+1,roc_auc_score(y, stack_score[:, i])))
                
                res[:, i] = clf.predict(S_test_2)
                
                F_columns.append('2nd level model_'+str(i+1))
            F_S_train = pd.DataFrame(stack_score, columns = F_columns)
            print('\n')
            print('Correlation between final predictions from 2nd level models:')
            print('\n')
            print(F_S_train.corr())
            print('\n')
        
            stack_res = res.mean(axis=1)            
            #print("2nd level models final score: %.5f" % (roc_auc_score(y, stack_score.mean(axis=1))))

            
        return stack_res
# LightGBM params
# LightGBM params
lgb_params_1 = {
    'learning_rate': 0.01,
    'n_estimators': 200,
    'subsample': 0.8,
    'subsample_freq': 5,
    'colsample_bytree': 0.8,
    'max_bin': 10,
    'min_child_samples': 44,
    'seed': 99,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt'
}

lgb_params_2 = {
    'learning_rate': 0.02,
    'n_estimators': 300,
    'subsample': 0.7,
    'subsample_freq': 2,
    'colsample_bytree': 0.3,  
    'num_leaves': 16,
    'seed': 99
}

lgb_params_3 = {
    'learning_rate': 0.01,
    'n_estimators': 100,
    'subsample': 0.7,
    'subsample_freq': 3,
    'colsample_bytree': 0.85,  
    'num_leaves': 28,
    'max_bin': 10,
    'min_child_samples': 70,
    'seed': 99
}


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier 

# Base models
lgb_model_1 = LGBMClassifier(**lgb_params_1)

lgb_model_2 = LGBMClassifier(**lgb_params_2)

lgb_model_3 = LGBMClassifier(**lgb_params_3)
# Stacker models
log_model = LogisticRegression()

et_model = ExtraTreesClassifier(n_estimators=100, max_depth=6, min_samples_split=10, random_state=10)
xgb = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
mlp_model = MLPClassifier(max_iter=7, random_state=42)
RFC = RandomForestClassifier()
# Mode 2 run
stack = Ensemble(mode=2,
        n_splits=3,
        stacker_2 = (log_model),         
        stacker_1 = (xgb,et_model),
        base_models = (lgb_model_1,lgb_model_2,lgb_model_3))       
        
y_pred = stack.fit_predict(train, y, test)  
sample_submission = pd.read_csv("../input/sample_submission.csv")
sample_submission['Target'] = y_pred
sample_submission.to_csv('Stacking.csv', index=False)

