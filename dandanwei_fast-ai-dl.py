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
import torch
torch.cuda.is_available()
from fastai.structured import *
from fastai.column_data import *
np.set_printoptions(threshold=50, edgeitems=20)

PATH='./'
from IPython.display import HTML, display
train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')
train.shape
test.shape
train.loc[train['idhogar'] == '2b58d945f'][['Id', 'idhogar', 'v2a1', 'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5']]
train

def show_null(df):
    df_null = df.isnull().sum()
    print(df_null[df_null>0])

def show_inf(df):
    df_inf = df.isinf().sum()
    print(df_inf[df_inf>0])

def fill_null_value(df):
    df['v2a1'].fillna(0, inplace=True)
    df['v18q1'].fillna(0, inplace=True)
    df['rez_esc'].fillna(-1, inplace=True)
    df['meaneduc'].fillna(df['escolari'], inplace=True)
    df['SQBmeaned'].fillna(df['meaneduc']**2, inplace=True)
    return df

def replace_yes_no(df, column):
    mapping = {'yes': 1, 'no': 0}
    df[column] = [mapping[item] if item in mapping else float(item) for item in df[column]]
    #print(df[['Target', 'dependency', 'edjefe', 'edjefa']])
    return df
    
def calculate_dependency(df):
    #df['dependency'] = (df['hogar_nin'] + df['hogar_mayor']) / (df['hogar_adul'] - df['hogar_mayor']) 
    #df_inf = df[df.dependency == np.inf]
    #df_inf['dependency'] =  (df_inf['hogar_nin'] + df_inf['hogar_mayor']) / df_inf['hogar_adul']
    dep = []
    for index, row in df.iterrows():
        if row['hogar_adul'] == row['hogar_mayor']:
            if row['hogar_adul'] > 0:
                dep.append( (row['hogar_nin'] + row['hogar_mayor']) * 2 / row['hogar_adul'] )
            else:
                dep.append( 4 )
        else:
            dep.append( (row['hogar_nin'] + row['hogar_mayor']) / (row['hogar_adul'] - row['hogar_mayor']))
    df['dependency'] = dep
    return df

def pick_features(df):
    return df[['v2a1', 'hacdor', 'hacapo', 'tipovivi2', 'computer', 'total_dis', 'tipovivi4', 'lugar5', 'lugar3', 'female', 'no_toilet_or_energy', 'toilet_and_refrig', 'wall_cond', 'floor_cond', 'roof_cond', 'home_cond', 'SQB_home_cond', 'v14a', 'refrig', 'v18q1', 'pisonotiene', 'cielorazo', 'abastaguadentro', 'abastaguafuera', 'abastaguano', 'noelec', 'sanitario1', 'energcocinar1', 'energcocinar4', 'elimbasu4', 'elimbasu5', 'dependency', 'overcrowding', 'SQBovercrowding', 'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4']]

def add_total_disable(df):
    print('add_total_disable')
    df['total_dis'] = df['dis']
    for index, row in df[df.dis == 1].iterrows():
        others = df[df.idhogar == row['idhogar']]
        for index_2, row in others.iterrows():
            if index != index_2:
                df.loc[index_2, 'total_dis'] += 1
    df['avg_dis_adult'] = df['dis'] / df['total_adults']
    df['avg_dis'] = df['dis'] / df['tamhog']
    return df

def add_total_adults(df):
    print('add_total_adults')
    df['is_adult'] = df.apply(lambda x: int(bool(int(x['age']/18))), axis=1)
    def get_total_adults(x):
        if x['parentesco1'] == 1:
            value =  df.loc[df.idhogar == x['idhogar']]['is_adult'].sum()
            if value == 0:
                return 0.01
            else:
                return value
        else:
            return 0
    df['total_adults'] = df.apply(get_total_adults, axis=1)
    return df

def add_avg_devices(df):
    df['avg_qmobilephone_adult'] = df['qmobilephone']/df['total_adults']
    df['avg_qmobilephone'] = df['qmobilephone']/df['tamhog']
    df['avg_computer_adult'] = df['computer'] / df['total_adults']
    df['avg_computer'] = df['computer'] / df['tamhog']
    df['avg_television_room'] = df['television'] / df['rooms']
    df['avg_television_adult'] = df['television'] / df['total_adults']
    df['avg_television'] = df['television'] / df['tamhog']
    df['avg_tablet'] = df['v18q1'] / df['tamhog']
    df['avg_tablet_adult'] = df['v18q1'] / df['total_adults']
    return df

def add_education_score(df):
    print('add_education_score')
    df['edu_level'] = df['instlevel9']*(2**8) + df['instlevel8']*(2**7) + df['instlevel7']*(2**6) + df['instlevel6']*(2**5) + df['instlevel5']*(2**4) + df['instlevel4']*(2**3) + df['instlevel3']*(2**2) + df['instlevel2']*(2**1) + df['instlevel1']
    def get_total_adults_edu(x):
        if x['parentesco1'] == 1:
            adults = df.loc[(df.idhogar == x['idhogar']) & (df.is_adult == 1)]
            if len(adults) > 0:
                return adults['edu_level'].sum() / len(adults)
            else:
                return 0
        else:
            return 0
    df['adults_edu'] = df.apply(get_total_adults_edu, axis=1)
    return df
    

def add_no_toilet_or_energy(df):
    df['no_toilet_or_energy'] = df['energcocinar1'] + df['sanitario1']
    df['toilet_and_refrig'] = df['v14a'] + df['refrig']
    df['wall_cond'] = df.apply(lambda row: row['epared1'] + 2*row['epared2'] + 4*row['epared3'], axis=1)
    df['floor_cond'] = df.apply(lambda row: row['etecho1'] + 2*row['etecho2'] + 4*row['etecho3'], axis=1)
    df['roof_cond'] = df.apply(lambda row: row['eviv1'] + 2*row['eviv2'] + 4*row['eviv3'], axis=1)
    df['home_cond'] = df.apply(lambda row: row['wall_cond'] + row['floor_cond'] + row['roof_cond'], axis=1)
    df['SQB_home_cond'] = df.apply(lambda row: row['home_cond']**2, axis=1)
    return df

train = fill_null_value(train)
train = replace_yes_no(train, 'edjefe')
train = replace_yes_no(train, 'edjefa')
train = calculate_dependency(train)
train = add_total_adults(train)
train = add_total_disable(train)
train = add_no_toilet_or_energy(train)
train = add_education_score(train)


test = fill_null_value(test)
test = replace_yes_no(test, 'edjefe')
test = replace_yes_no(test, 'edjefa')
test = calculate_dependency(test)
test = add_total_adults(test)
test = add_total_disable(test)
test = add_no_toilet_or_energy(test)
test = add_education_score(test)

train.shape
test.shape
#import feather
#train.to_feather('train.feather')
#test.to_feather('test.feather')
#import feather
#train = feather.read_dataframe('train.feather')
#test = feather.read_dataframe('test.feather')
train_head = train.loc[train.parentesco1 == 1]
test_head = test.loc[test.parentesco1 == 1]
train_head.shape
def get_overfit_training(ptrain):

    traind = ptrain
    ntrain = ptrain[ptrain.Target == 1]
    ntrain['age'] += 1
    #print(traind.shape)
    traind = traind.append(ntrain)
    ntrain['age'] += 1
    traind = traind.append(ntrain)
    ntrain['age'] += 1
    traind = traind.append(ntrain)
    ntrain['age'] -= 4
    traind = traind.append(ntrain)
    ntrain['age'] -= 1
    traind = traind.append(ntrain)
    ntrain['age'] -= 1
    traind = traind.append(ntrain)
    ntrain['age'] -= 1
    traind = traind.append(ntrain)
    n2train = ptrain[ptrain.Target == 2]
    n2train['age'] += 1
    traind = traind.append(n2train)
    n2train['age'] += 1
    traind = traind.append(n2train)
    n2train['age'] -=3
    traind = traind.append(n2train)
    n3train = ptrain[ptrain.Target == 3]
    n3train['age'] += 1
    traind = traind.append(n3train)
    n2train['age'] += 1
    traind = traind.append(n3train)
    n2train['age'] += 1
    traind = traind.append(n3train)
    n2train['age'] -= 4
    traind = traind.append(n3train)
    #print(traind.shape)
    return traind

ptrain_head = get_overfit_training(train_head); ptrain_head.shape
ptest_head = test_head.copy()

ptest_head['Target'] = np.random.randint(1, high=5, size=ptest_head.shape[0])
ptrain_target = ptrain_head['Target']
ptrain_head.drop('Target', axis=1, inplace=True)
ptrain_head['Target'] = ptrain_target

a = (ptrain_head.columns == ptest_head.columns)
a[a == False]
ptrain_head.set_index('Id', inplace=True)
ptest_head.set_index('Id', inplace=True)
alld = ptrain_head.append(ptest_head)
alld
alld.loc[set(ptrain_head.index.tolist())].shape == ptrain_head.shape
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
cats = ['hacdor', 'rooms', 'hacapo', 'v14a', 'refrig', 'r4h1', 
        'r4h2', 'r4m1', 'r4m2', 'tamhog', 'tamviv', 'escolari', 
        'hhsize', 'pared', 'piso', 'techo', 'cielorazo', 'abastagua', 'electricity', 'sanitario',
        'energcocinar', 'elimbasu', 'epared', 'etecho', 'eviv', 'dis', 'male_or_famale', 'estadocivil',
        'hogar_nin', 'hogar_adul', 'hogar_mayor', 'hogar_total', 'instlevel', 'bedrooms', 'tipovivi',
        'computer', 'television', 'mobilephone', 'lugar', 'area', 'is_adult']
to_drop = ['v18q', 'r4h3', 'r4m3', 'r4t1', 'r4t2', 'r4t3', 'parentesco', 'idhogar', 'SQB', 'agesq', ]
to_merge = ['pared', 'piso', 'techo', 'abastagua', 
            {'electricity': ['public', 'planpri', 'noelec', 'coopele']}, 
            'sanitario', 'energcocinar', 'elimbasu', 'epared', 'etecho',
            'eviv', {'male_or_famale': ['male', 'female']},
            'estadocivil', 'instlevel', 'tipovivi', 'lugar', 'area']
def merge_columns(df, to_merge):
    for c in to_merge:
        if isinstance(c, str):
            new_col = df[[ col for col in df if col.startswith(c)]].astype(str).apply(lambda x: x.str.cat(), axis=1)
            df = df.drop([ col for col in df if col.startswith(c)], axis=1)
            df[c] = new_col
        elif isinstance(c, dict):
            new_col_name = list(c.keys())[0]
            new_col = df[c[new_col_name]].astype(str).apply(lambda x: x.str.cat(), axis=1)
            df = df.drop(c[new_col_name], axis=1)
            df[new_col_name] = new_col
    return df
#ptrain = merge_columns(train_head, to_merge)
#ptest = merge_columns(test_head, to_merge)
alld = merge_columns(alld, to_merge)
def drop_columns(df, to_drop):
    for c in to_drop:
        df = df.drop([ col for col in df if col.startswith(c)], axis=1)
    return df
#ptrain = drop_columns(ptrain, to_drop)
#ptest = drop_columns(ptest, to_drop)
alld = drop_columns(alld, to_drop)
#ptest.shape
#ptrain.shape
alld.shape
from sklearn.metrics import f1_score
def macro_f1(y_pred, targ):
    #print(targ)
    target = targ.cpu().numpy().argmax(axis=1)
    #print(y_pred)
    y_predict = y_pred.cpu().numpy().argmax(axis=1)
    return f1_score(target, y_predict, average='macro')
#dtrain = ptrain
#dtrain.shape
#dtrain.set_index('Id')
#ptest.set_index('Id')
#for v in cats: dtrain[v] = dtrain[v].astype('category').cat.as_ordered()
#for v in cats: ptest[v] = ptest[v].astype('category').cat.as_ordered()
for v in cats: alld[v] = alld[v].astype('category').cat.as_ordered()
alld['Target'] = alld['Target'].astype('int')
#dtrain_target = dtrain['Target']
#dtrain.drop('Target', axis=1, inplace=True)
#dtrain['Target'] = dtrain_target
#ptest.reset_index(drop=True, inplace=True)

#ptest.drop('rez_esc', axis=1, inplace=True)
#dtrain.drop('rez_esc', axis=1,  inplace=True)
alld.drop('rez_esc', axis=1,  inplace=True)
DataFrameSummary(alld).summary()
#DataFrameSummary(dtrain).summary()
#a = (DataFrameSummary(ptest).summary().loc['types'] == DataFrameSummary(dtrain).summary().loc['types'])
#a[a == False]
#
contin_vars = [c for c in alld.columns if c not in cats + ['Target', 'Id']]
alld
for v in contin_vars:
    #dtrain[v] = dtrain[v].fillna(0).astype('float32')
    #ptest[v] = ptest[v].fillna(0).astype('float32')
    alld[v] = alld[v].fillna(0).astype('float32')
alld.shape
#a = (DataFrameSummary(ptest).summary().loc['types'] == DataFrameSummary(dtrain).summary().loc['types'])
#a[30:-1]
_, _, nas, mapper = proc_df(alld, 'Target', do_scale=True)
traindf = alld.loc[set(ptrain_head.index.tolist())]
traindf.shape[0] == ptrain_head.shape[0]
testdf = alld.loc[set(ptest_head.index.tolist())]
testdf.shape[0] == ptest_head.shape[0]
df_test, _, nas, mapper = proc_df(testdf, 'Target', do_scale=True, mapper=mapper)
df, y, nas, mapper = proc_df(traindf, 'Target', do_scale=True, mapper=mapper)
df.shape
df_test.shape
DataFrameSummary(df).summary()
samp_size = df.shape[0]
train_ratio = 0.8
# train_ratio = 0.9
train_size = int(samp_size * train_ratio)
train_size
val_idx = random.sample(range(0, df.shape[0]), df.shape[0] - train_size)
df
#df.loc[[val_idx[0]]]
val_idx = sorted(val_idx)
len(val_idx)
y
y.size
yl = np.zeros((y.size, 4))
yindex = np.array([t -1 for t in y]); yindex

yl[np.arange(y.size), yindex] = 1; yl
df.shape
df_test.shape
df[0:1]
df_test[500:501]
md = ColumnarModelData.from_data_frame(PATH, val_idx, df, yl, cat_flds=cats, bs=64, is_reg=False, is_multi=True, test_df=df_test)
cat_sz = [(c, len(alld[c].cat.categories)+1) for c in cats]
cat_sz
emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]
emb_szs
#DataFrameSummary(df).summary()
#for c in cats:
#    if len(dtrain[dtrain['rooms'].isna()  ]) > 0:
#        print(c)
DataFrameSummary(df)
#DataFrameSummary(dtrain).summary()
#emb_szs
type(emb_szs)

m = md.get_learner(emb_szs, len(df.columns)-len(cats),
                   0.3, 4, [500,250], [0.1,0.05])
lr = 1e-3
#m.data.val_dl
#m.data.test_dl
#m.data.val_dl.get_batch(indices=)
#m.summary()
#m.fit(lr, 5, metrics=[macro_f1], cycle_len=3)
m.fit(lr, 3, metrics=[macro_f1], cycle_len=3)
m.fit(lr, 2, metrics=[macro_f1], cycle_len=3)
m.fit(lr, 1, metrics=[macro_f1], cycle_len=3)
pred_test=m.predict(is_test=True)
pred_test.shape
pred_test
predicts = pred_test.argmax(axis=1); predicts
predicts = predicts + 1; predicts
predicts.shape
df_test.shape
df_test['Target'] = predicts
df_test['Id'] = df_test.index
final = df_test[['Id', 'Target']]

test['Target'] = np.random.randint(1, high=5, size=test.shape[0])
ftest = test[['Id', 'Target']]
ftest
dtest = ftest.set_index('Id')
dtest['Id'] = dtest.index; dtest
#dtest.loc[final.Id.values.tolist]
dtest.loc[final.index.tolist()]['Target']  = final['Target']
np.histogram(dtest.loc[final.index.tolist()]['Target'], bins=4)
np.histogram(final['Target'], bins=4)
final.loc['ID_21471f283']['Target']
dtest['Target'] = dtest.apply(lambda x: x['Target'] if x['Id'] not in final['Id'] else final.loc[x['Id']]['Target'], axis=1)
np.histogram(dtest.loc[final.index.tolist()]['Target'], bins=4)
dtest.shape
dtest.to_csv('submission_dl.csv', index=False)
