import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# load data
train = pd.read_csv('../input/train.csv')
print('Shape of train set {}'.format(train.shape))
test = pd.read_csv('../input/test.csv')
print('Shape of test set {}'.format(test.shape))
def histogram(col, data, xlabel):
    fig = data[col].hist()
    plt.xlabel(xlabel); plt.ylabel('# households')
    return fig

def corr_plot(df, xlabels='auto', ylabels='auto', figsize=(5, 5), fmt='.2f'):
    plt.figure(figsize=figsize)
    return sns.heatmap(df.corr(), annot=True, fmt=fmt, 
                       xticklabels=xlabels, yticklabels=ylabels)

def cal_dependency_rate(df):
    df['n_dependency'] = df['hogar_total'] - df['hogar_adul']
    non_zero = (df.hogar_adul != 0)
    df.loc[non_zero, 'num_dep_rate'] = df.loc[non_zero, 'n_dependency']/df.loc[non_zero, 'hogar_adul']
    df.loc[-non_zero, 'num_dep_rate'] = np.nan
    return df
import math
def to_nearest_int(e):
    floor_int = int(math.floor(e))
    if e - floor_int < 0.5:
        return floor_int
    else:
        return floor_int + 1

def round_to_nearest_int(arr):
    # round each entry of given array to its nearest integer (floor or ceiling)
    return [to_nearest_int(e) for e in arr]
train.dependency.value_counts().head()
train = cal_dependency_rate(train)
train['num_dep_rate'].describe()
train.query('num_dep_rate == 8')['Target']
basic_feats = ['hogar_nin', 'hogar_adul', 'hogar_mayor', 'num_dep_rate', 
                'overcrowding', 'rooms', 'bedrooms']
basic_labels = ['n_kids', 'n_adults', 'n_seniors', 'dependency_rate', 
               'overcrowding', 'rooms', 'bedrooms', 'Target']
cols = basic_feats + ['Target']
corr_plot(df=train.drop_duplicates('idhogar')[cols], figsize=(8, 8), 
          xlabels=basic_labels, ylabels=basic_labels)
plt.savefig('basic_corr.pdf')
print('# households in train set: {}'.format(train.idhogar.nunique()))
print('# household heads in train set: {}'.format(sum(train['parentesco1'] == 1)))
cols = ['idhogar', 'parentesco1', 'male', 'female'] + ['Target']
gender_df = train.loc[train.parentesco1 == 1, cols]
gender_df.loc[gender_df['male'] == 1, 'house_head_gender'] = 'male'
gender_df.loc[gender_df['female'] == 1, 'house_head_gender'] = 'female'
pd.crosstab(gender_df.house_head_gender, gender_df.Target, 
            margins=True).style.background_gradient(cmap='viridis', low=.5, high=0)
edu_cols = [cc for cc in train.columns if 'instlevel' in cc] + ['escolari']
cols = ['idhogar', 'parentesco1'] + edu_cols + ['Target']
edu_df = train.loc[train.parentesco1 == 1, cols]

# convert binary edu levels to numeric values
for i in range(1, 10):
    edu_df.loc[edu_df['instlevel{}'.format(i)] == 1, 'head_edu_level'] = i

# query years of schooling of house head
edu_df['head_school_years'] = edu_df['escolari']
edu_df.head()
edu_vs_target_cor = edu_df.corr().loc['head_edu_level', 'Target']
print('Correlation between education level of household head and financial status: {}'.format(edu_vs_target_cor))
# add education data of household head to original data
train = pd.merge(train, edu_df[['idhogar', 'head_edu_level', 'head_school_years']], 
                 how='outer', on='idhogar')
train.shape
cols = ['meaneduc', 'head_edu_level', 'head_school_years'] + ['Target']
labels = ['mean_edu_level', 'head_edu_level', 'head_school_years', 'Target']
corr_plot(train[cols], xlabels=labels, ylabels=labels)
plt.savefig('edu_vs_poverty.pdf')
def add_quality(df, componente='pared', component='wall'):
    for i in [1,2,3]:
        i_quality = (df['e{}{}'.format(componente, i)] == 1)
        df.loc[i_quality, '{}_quality'.format(component)] = i
    return df
# add wall quality
train = add_quality(train, componente='pared', component='wall')
cols = ['epared1', 'epared2', 'epared3', 'wall_quality']
train[cols].head()
# add roof quality
train = add_quality(train, componente='techo', component='roof')
cols = ['etecho1', 'etecho2', 'etecho3', 'roof_quality']
train[cols].head()
# add floor quality
train = add_quality(train, componente='viv', component='floor')
cols = ['eviv1', 'eviv2', 'eviv3', 'floor_quality']
train[cols].head()
# correlation with target
cols = ['floor_quality', 'roof_quality', 'wall_quality'] + ['Target']
corr_plot(train[cols])
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
# rename columns from Spanish to English
translate = {'blolad': 'block',
             'zocalo': 'socket',
             'preb': 'cement',
             'des': 'waste',
             'mad': 'wood',
             'zinc': 'zink',
             'fibras': 'natural_fibers',
             'other': 'other'}
train = to_english(train, sp_pre='pared', eng_pre='wall_', 
                   translate=translate)

# add material column
material = translate.values()
for mat in material:
    is_mat = (train['wall_{}'.format(mat)] == 1)
    train.loc[is_mat, 'wall_material'] = mat
_ = train.drop_duplicates('idhogar')
pd.crosstab(_['wall_material'], _['Target'], 
            normalize=True, margins=True).style.background_gradient(cmap='viridis', low=.5, high=0).format('{:.2%}')
# rename columns
translate = { 
    'moscer': 'mosaic',
    'cemento': 'cement',
    'other': 'other',
    'natur': 'natural',
    'notiene': 'no_floor',
    'madera': 'wood'
}
train = to_english(train, sp_pre='piso', eng_pre='floor_', translate=translate)

# add material
material = translate.values()
for mat in material:
    is_mat = (train['floor_{}'.format(mat)] == 1)
    train.loc[is_mat, 'floor_material'] = mat
_ = train.drop_duplicates('idhogar')
pd.crosstab(_['floor_material'], _['Target'], 
            normalize=True, margins=True).style.background_gradient(cmap='viridis', low=.5, high=0).format('{:.2%}')
# rename columns
translate = {
     'zinc': 'zinc',
     'entrepiso': 'fiber cement',
     'cane': 'natural fibers',
     'otro': 'other'
}
train = to_english(train, sp_pre='techo', eng_pre='roof_', translate=translate)
# add material
material = translate.values()
for mat in material:
    is_mat = (train['roof_{}'.format(mat)] == 1)
    train.loc[is_mat, 'roof_material'] = mat
_ = train.drop_duplicates('idhogar')
pd.crosstab(_['roof_material'], _['Target'], 
            normalize=True, margins=True).style.background_gradient(cmap='viridis', low=.5, high=0).format('{:.2%}')
# rename columns 
translate = {
    'guadentro': 'inside_house',
    'guafuera': 'outside_house',
    'guano': 'no'
}
train = to_english(train, sp_pre='abasta', eng_pre='water_provision_', 
                   translate=translate)

# add water_provision category   
for cat in translate.values():
    is_cat = (train['water_provision_{}'.format(cat)] == 1)
    train.loc[is_cat, 'water_provision'] = cat
pd.crosstab(train['water_provision'], train['Target'], 
            normalize=True, margins=True).style.background_gradient(cmap='viridis', low=.5, high=0).format('{:.2%}')
translate = {
    'public': 'public',
    'planpri': 'private_plan',
    'noelec': 'no',
    'coopele': 'cooperate'
}
train = to_english(train, sp_pre='', eng_pre='electric_', translate=translate)
# refresh data
train = pd.read_csv('../input/train.csv')
print('Shape of train set {}'.format(train.shape))
test = pd.read_csv('../input/test.csv')
print('Shape of test set {}'.format(test.shape))
test['Target'] = np.nan
data_all = pd.concat([train, test])
print('Shape of all data: {}'.format(data_all.shape))
n_house = data_all['idhogar'].nunique()
print('# unique households in data: {}'.format(n_house))
is_head = (data_all.parentesco1 == 1)
head_df = data_all.loc[is_head, :]
print('Shape of head_df: {}'.format(head_df.shape))
n_head = head_df['Id'].nunique()
print('# unique household heads: {}'.format(n_head))
head_df.loc[head_df['male'] == 1, 'head_gender'] = 'male'
head_df.loc[head_df['female'] == 1, 'head_gender'] = 'female'
print('Shape of head_df: {}'.format(head_df.shape))
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
# one-hot encode head gender
head_df = onehot_encode('head_gender', head_df)
head_gender_feats = [cc for cc in head_df.columns if 'head_gender' in cc]
head_gender_feats
# convert binary edu levels to numeric values
for i in range(1, 10):
    head_df.loc[head_df['instlevel{}'.format(i)] == 1, 'head_edu_level'] = i
    
print(head_df.shape)
head_df = head_df.rename(columns={'escolari': 'head_school_years'})
print(data_all.shape)
# as there are a few households with no head, we need an outer join 
# to avoid missing those houses
cols = ['idhogar', 'head_school_years', 'head_edu_level'] + head_gender_feats
data_all = pd.merge(data_all, head_df[cols], how='outer', on='idhogar')
print(data_all.shape)
house_head_feats = ['head_school_years', 'head_edu_level'] + head_gender_feats
house_head_feats
data_all = add_quality(data_all, componente='pared', component='wall')
data_all = add_quality(data_all, componente='techo', component='roof')
data_all = add_quality(data_all, componente='viv', component='floor')
print(data_all.shape)
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
wall_feats
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
floor_feats
# roof
translate = {
     'zinc': 'zinc',
     'entrepiso': 'fiber cement',
     'cane': 'natural fibers',
     'otro': 'other'
}
data_all = to_english(data_all, sp_pre='techo', eng_pre='roof_', translate=translate)
roof_feats = [cc for cc in data_all.columns if 'roof_' in cc]
roof_feats
print(data_all.shape)
material_feats = roof_feats + wall_feats + floor_feats
material_feats
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
energy_feats
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
toilet_feats
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
rubbish_feats
facility_feats = water_feats + elec_feats + energy_feats + toilet_feats + rubbish_feats
features = house_head_feats + material_feats + facility_feats + basic_feats
print('# features: {}'.format(len(features)))
# prepare train and test sets, valid maybe
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import get_scorer
cols = ['Id'] + features + ['Target']
train = data_all.loc[data_all['Target'].notnull(), cols]
X_train, y_train = train[features], train['Target']

test = data_all.loc[data_all['Target'].isnull(), cols]
X_test = test[['Id'] + features]
import lightgbm as lgb
gbm = lgb.LGBMRegressor(n_jobs=2, random_state=0)
param_grid = {'num_leaves': np.arange(10, 50, 10), 
              'learning_rate': np.arange(0.05, 0.2, 0.05),
             'n_estimators': np.arange(10, 25, 5)}
scoring = {'mse': get_scorer('neg_mean_squared_error')}
metric = 'mse'
# train and param tuning
gs = GridSearchCV(gbm,
                  param_grid=param_grid,
                  scoring=scoring,
                  cv=5,
                  refit=metric,
                  verbose=True)
gs.fit(X_train, y_train)
# dump model
best_estimator = gs.best_estimator_
# json_model = best_estimator.dump_model('model.json')
# predict
y_pred = best_estimator.predict(X_test[features])
round_pred = round_to_nearest_int(y_pred)
submit = pd.DataFrame({'Id': X_test['Id'], 'Target': round_pred})
submit.to_csv('submisssion.csv', index=False)
# feature importance
print('Feature importances:', list(best_estimator.feature_importances_))
