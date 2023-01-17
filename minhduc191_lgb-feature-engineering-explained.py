import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# load data
train = pd.read_csv('../input/train.csv')
print('Shape of train set {}'.format(train.shape))
test = pd.read_csv('../input/test.csv')
print('Shape of test set {}'.format(test.shape))

# for param tuning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import get_scorer
def onehot_encode(cat_feat, data):
    '''
    Encode given categorical feature and add names of new binary columns into the set of features
    :param cat_feat:
    :param data:
    :return:
    '''
    encoded = pd.get_dummies(data[cat_feat], prefix=cat_feat, dummy_na=True)
    res = pd.concat([data.drop(columns=[cat_feat]), encoded], axis='columns')
    return res

def add_quality(df, componente='pared', component='wall'):
    for i in [1,2,3]:
        i_quality = (df['e{}{}'.format(componente, i)] == 1)
        df.loc[i_quality, '{}_quality'.format(component)] = i
    return df

def to_english(df, sp_pre='pared', eng_pre='wall_', translate=None):
    '''
    rename certain columns in specified dataframe from Spanish 
    to English, given the translation
    '''
    for sp in translate.keys():
        spanish_name = sp_pre + '{}'.format(sp)
        english_name = eng_pre + '{}'.format(translate[sp])
        df.rename(columns={spanish_name: english_name}, inplace=True)
    
    return df
# join train and test
test['Target'] = np.nan
data_all = pd.concat([train, test])

print('Shape of all data: {}'.format(data_all.shape))
n_house = data_all['idhogar'].nunique()
print('# unique households in data: {}'.format(n_house))
n_row = data_all.shape[0]
n_null = data_all.drop('Target', axis='columns').isnull().values.sum(axis=0)
columns = list(data_all.columns)
columns.remove('Target')
pd.DataFrame({'column': columns, 
              'n_null': n_null}).sort_values('n_null', ascending=False).head(10)
def mk_derived_feats(df):
    feats_div = [('children_fraction', 'r4t1', 'r4t3'), 
                 ('working_man_fraction', 'r4h2', 'r4t3'),
                 ('all_man_fraction', 'r4h3', 'r4t3'),
                 ('human_density', 'tamviv', 'rooms'),
                 ('human_bed_density', 'tamviv', 'bedrooms'),
#                  ('rent_per_person', 'v2a1', 'r4t3'),
#                  ('rent_per_room', 'v2a1', 'rooms'),
                 ('mobile_density', 'qmobilephone', 'r4t3'),
#                  ('tablet_density', 'v18q1', 'r4t3'),
                 ('mobile_adult_density', 'qmobilephone', 'r4t2'),
#                  ('tablet_adult_density', 'v18q1', 'r4t2'),
                 #('', '', ''),
                ]
    
    feats_sub = [('people_not_living', 'tamhog', 'tamviv'),
                 ('people_weird_stat', 'tamhog', 'r4t3')]

    for f_new, f1, f2 in feats_div:
        df['fe_' + f_new] = (df[f1] / df[f2]).astype(np.float32)       
    for f_new, f1, f2 in feats_sub:
        df['fe_' + f_new] = (df[f1] - df[f2]).astype(np.float32)
    return df

def mk_agg_feats(df):
    # aggregation rules over household
    aggs_num = {'age': ['min', 'max', 'mean'],
                'escolari': ['min', 'max', 'mean']
               }
    aggs_cat = {'dis': ['sum', 'mean']} # mean will give us percentage of disable members
    for s_ in ['estadocivil', 'parentesco', 'instlevel']:
        for f_ in [f_ for f_ in df.columns if f_.startswith(s_)]:
            aggs_cat[f_] = ['mean', 'count'] # mean will give us percentage of the type
    
    # aggregate over household
    for name_, df_ in [('18', df.query('age >= 18'))]:
        df_agg = df_.groupby('idhogar').agg({**aggs_num, **aggs_cat}).astype(np.float32)
        df_agg.columns = pd.Index(['agg' + name_ + '_' + e[0] + "_" + e[1].upper() for e in df_agg.columns.tolist()])
        df = df.join(df_agg, how='left', on='idhogar')
        del df_agg
    
    return df

def drop_redundant(df):
    # Drop SQB variables, as they are just squres of other vars 
    df.drop([f_ for f_ in df.columns if f_.startswith('SQB') or f_ == 'agesq'], axis=1, inplace=True)
    # Drop id's
#     df.drop(['Id', 'idhogar'], axis=1, inplace=True)
    # Drop repeated columns
    df.drop(['hhsize', 'area2'], axis=1, inplace=True)
    return df
from sklearn.preprocessing import LabelEncoder

yes_no_map = {'no': 0, 'yes': 1}
data_all['dependency'] = data_all['dependency'].replace(yes_no_map).astype(np.float32)
data_all['edjefe'] = data_all['edjefe'].replace(yes_no_map).astype(np.float32)
data_all['edjefa'] = data_all['edjefa'].replace(yes_no_map).astype(np.float32)

data_all['idhogar'] = LabelEncoder().fit_transform(data_all['idhogar'])

data_all = mk_derived_feats(data_all)
data_all = mk_agg_feats(data_all)
data_all = drop_redundant(data_all)
fe_feats = [ff for ff in data_all.columns if ff.startswith('fe_')]
agg_feats = [ff for ff in data_all.columns if ff.startswith('agg')]
basic_feats = ['dependency']
# basic_feats = ['hogar_nin', 'hogar_adul', 'hogar_mayor', 'dependency', 
#                 'overcrowding', 'rooms', 'bedrooms']
is_head = (data_all.parentesco1 == 1)
head_df = data_all.loc[is_head, :]
# print('Shape of head_df: {}'.format(head_df.shape))

n_head = head_df.shape[0]
print('# unique household heads: {}'.format(n_head))
head_df.loc[head_df['male'] == 1, 'head_gender'] = 'male'
head_df.loc[head_df['female'] == 1, 'head_gender'] = 'female'
print('Shape of head_df: {}'.format(head_df.shape))

# one-hot encode head gender
head_df = onehot_encode('head_gender', head_df)

head_gender_feats = [cc for cc in head_df.columns if 'head_gender' in cc]
head_gender_feats
# convert binary edu levels to numeric values
for i in range(1, 10):
    head_df.loc[head_df['instlevel{}'.format(i)] == 1, 'head_edu_level'] = i
    
head_df = head_df.rename(columns={'escolari': 'head_school_years'})
# as there are a few households with no head, we need an left outer join 
# to avoid missing those houses
cols = ['idhogar', 'head_school_years', 'head_edu_level'] + head_gender_feats
data_all = pd.merge(data_all, head_df[cols], how='left', on='idhogar')
print(data_all.shape)
house_head_feats = ['head_school_years', 'head_edu_level'] + head_gender_feats
data_all = add_quality(data_all, componente='pared', component='wall')
data_all = add_quality(data_all, componente='techo', component='roof')
data_all = add_quality(data_all, componente='viv', component='floor')
print(data_all.shape)
# rename material columns
# wall
translate = {'blolad': 'block',
             'zocalo': 'socket',
             'preb': 'cement',
             'des': 'waste',
             'mad': 'wood',
             'zinc': 'zink',
             'fibras': 'natural_fibers',
             'other': 'other'}
data_all = to_english(data_all, sp_pre='pared', eng_pre='wall_', 
                   translate=translate)
wall_feats = [cc for cc in data_all.columns if 'wall_' in cc]

# floor
translate = { 
    'moscer': 'mosaic',
    'cemento': 'cement',
    'other': 'other',
    'natur': 'natural',
    'notiene': 'no_floor',
    'madera': 'wood'
}
data_all = to_english(data_all, sp_pre='piso', eng_pre='floor_', translate=translate)
floor_feats = [cc for cc in data_all.columns if 'floor_' in cc]

# roof
translate = {
     'zinc': 'zinc',
     'entrepiso': 'fiber cement',
     'cane': 'natural fibers',
     'otro': 'other'
}
data_all = to_english(data_all, sp_pre='techo', eng_pre='roof_', translate=translate)
roof_feats = [cc for cc in data_all.columns if 'roof_' in cc]

material_feats = roof_feats + wall_feats + floor_feats
# water
translate = {
    'guadentro': 'inside_house',
    'guafuera': 'outside_house',
    'guano': 'no'
}
data_all = to_english(data_all, sp_pre='abasta', eng_pre='water_provision_', 
                   translate=translate)
water_feats = [cc for cc in data_all.columns if 'water_provision_' in cc]

# electricity
translate = {
    'public': 'public',
    'planpri': 'private_plan',
    'noelec': 'no',
    'coopele': 'cooperate'
}
data_all = to_english(data_all, sp_pre='', eng_pre='electric_', translate=translate)
elec_feats = [cc for cc in data_all.columns if 'electric_' in cc]

# energy
translate = {
    'cinar1': 'no',
    'cinar2': 'electricity',
    'cinar3': 'gas',
    'cinar4': 'charcoal'
}
data_all = to_english(data_all, sp_pre='energco', eng_pre='energy_', translate=translate)
energy_feats = [cc for cc in data_all.columns if 'energy_' in cc]

# toilet
translate = {
    '1': 'no',
    '2': 'sewer',
    '3': 'septic_tank',
    '5': 'black hole',
    '6': 'other'
}
data_all = to_english(data_all, sp_pre='sanitario', eng_pre='toilet_', translate=translate)
toilet_feats = [cc for cc in data_all.columns if 'toilet_' in cc]

# rubbish
translate = {
    '1': 'tanker truck',
    '2': 'buried',
    '3': 'burning',
    '4': 'throw empty place',
    '5': 'throw to river',
    '6': 'other'
}
data_all = to_english(data_all, sp_pre='elimbasu', eng_pre='rubbish_', translate=translate)
rubbish_feats = [cc for cc in data_all.columns if 'rubbish_' in cc]
facility_feats = water_feats + elec_feats + energy_feats + toilet_feats + rubbish_feats
translate = {
    '1': 'own_fully_paid',
    '2': 'own_pay_installment',
    '3': 'rented',
    '4': 'precarious',
    '5': 'other'
}
data_all = to_english(data_all, sp_pre='tipovivi', eng_pre='living_type_', 
                      translate=translate)
live_feats = [cc for cc in data_all.columns if 'living_type_' in cc]
head_df = data_all.query('parentesco1 == 1')
train = head_df.loc[head_df['Target'].notnull(), :]
test = head_df.loc[head_df['Target'].isnull(), :]
# features to be used
features = basic_feats + house_head_feats + material_feats + facility_feats + live_feats + fe_feats + agg_feats
print('# features: {}'.format(len(features)))
# use a validation set to check how the trained model perform
X, y = train[['idhogar', 'Id'] + features], train['Target']
X_train, X_valid, y_train, y_valid = train_test_split(X[features], y, 
                                                      test_size=0.1, random_state=0)
import lightgbm as lgb

gbm = lgb.LGBMClassifier(n_jobs=4, random_state=0, class_weight='balanced')
param_grid = {'num_leaves': np.arange(10, 50, 10), 
              'learning_rate': np.arange(0.05, 0.2, 0.05),
             'n_estimators': np.arange(10, 25, 5)}
scoring = {'f1_macro': get_scorer('f1_macro')}
metric = 'f1_macro'
# train and param tuning
gs = GridSearchCV(gbm,
                  param_grid=param_grid,
                  scoring=scoring,
                  cv=5,
                  refit=metric,
                  verbose=True,
                 n_jobs=4,
                 )
gs.fit(X_train, y_train)

best_estimator = gs.best_estimator_
from sklearn.metrics import f1_score
# check perf of trained model on validaton set
y_pred = best_estimator.predict(X_valid)
macro_f1 = f1_score(y_valid, y_pred, average='macro')
print(macro_f1)
# predict
X_test = test[['Id', 'idhogar'] + features]
y_pred = best_estimator.predict(X_test[features])
pred_for_heads = pd.DataFrame({'Id': X_test['Id'], 'idhogar': X_test['idhogar'], 
                              'Target': y_pred})
pred_for_heads['Target'] = pred_for_heads['Target'].apply(int)
pred_for_heads['Target'].head()
#merge back to include other rows into submission
test_all = data_all.loc[data_all['Target'].isnull(), ['Id', 'idhogar']]
submit = pd.merge(test_all, pred_for_heads[['idhogar', 'Target']], how='left')
submit.head()
sum(submit['Target'].isnull())
submit.fillna(1, inplace=True)
submit['Target'] = submit['Target'].astype(np.int16)
submit.head()
submit[['Id', 'Target']].to_csv('submisssion.csv', index=False)
lgb.plot_importance(best_estimator, figsize=(15, 15), grid=False)
