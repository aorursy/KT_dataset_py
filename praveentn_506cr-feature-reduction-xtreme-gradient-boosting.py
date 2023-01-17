# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from numpy import sort

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import warnings
warnings.filterwarnings('ignore')
import multiprocessing

n_jobs = multiprocessing.cpu_count()
n_jobs
#prediction and Classification Report
from sklearn.metrics import classification_report

# select features using threshold
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.metrics.scorer import make_scorer

# plot tree, importance
from xgboost import plot_tree, plot_importance

# load xgboost, test train split
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/train.csv')
num_of_cols = len(list(df.columns))
num_of_cols
pd.options.display.max_columns = num_of_cols
len(df)
# columns with null values
df_isna = pd.DataFrame(df.isnull().sum())
df_isna.loc[(df_isna.loc[:, df_isna.dtypes != object] != 0).any(1)]
nan_cols = list(df_isna.loc[(df_isna.loc[:, df_isna.dtypes != object] != 0).any(1)].T.columns)
nan_cols
df[nan_cols].describe()
df.describe(include='all')
df[nan_cols].sample(3000).describe()
df['parentesco1'].loc[df.parentesco1 == 1].describe()
(df['parentesco1'].loc[df.parentesco1 == 1].describe()['count']/len(df))*100
# find number of households

df['idhogar'].describe()
house_ids = list(df['idhogar'].unique())
df[['parentesco1','idhogar']].loc[df.parentesco1 == 1].head(5)
hid_heads = df.groupby(['idhogar'])['parentesco1'].apply(lambda x: pd.unique(x.values.ravel()).tolist()).reset_index()
len(hid_heads)
df_hid = pd.DataFrame(hid_heads, index=None, columns=['idhogar','parentesco1'])
df_hid.sample(5)
df_hid['parentesco1'] = df_hid['parentesco1'].apply(lambda x: ''.join(map(str, x)))
df_hid.sample(5)
df_hid.loc[df_hid.parentesco1 == '0']
# id's without head!
hid_wo_heads = list(df_hid['idhogar'].loc[df_hid.parentesco1 == '0'])
len(hid_wo_heads)
df_hwoh = df[df['idhogar'].isin(hid_wo_heads)]
df_hwoh[['idhogar', 'parentesco1','v2a1']]
df['v2a1'].hist()
df['v2a1'].loc[-df['idhogar'].isin(hid_wo_heads)].hist()
df_hwoh['v2a1'].hist()
len(df_hwoh)
# these 15 households (23 rows) doesn't have a head..
# we should exclude these from analysis and scoring perhaps...
df_hwoh['idhogar'].unique()
print(df[['Id','v2a1','idhogar','parentesco1','Target']].loc[df.idhogar == '09b195e7a'])
print(df[['Id','v2a1','idhogar','parentesco1','Target']].loc[df.idhogar == 'f2bfa75c4'])

# required dataframe - without households without a head!!
print("before removal: ", len(df))
df = df.loc[-df['idhogar'].isin(hid_wo_heads)]
print("after removal: ", len(df))
df['v2a1'].describe().plot()
df[['v18q1', 'rez_esc', 'meaneduc', 'SQBmeaned']].describe().plot()
import gc

gc.collect()
len(df['v2a1'].unique())
df['v2a1'].unique()
df['v2a1'].max()
df[['v2a1','idhogar','parentesco1','Target']].loc[df.v2a1 > 1000000]
df[['v2a1','idhogar','parentesco1','Target']].loc[df.v2a1 >= 1000000]
# remove these two rows...
df[['v2a1','idhogar','parentesco1','Target']].loc[df.idhogar == '563cc81b7']
print("before removal: ", len(df))
df.drop(df[df.idhogar == '563cc81b7'].index, inplace=True)
print("after removal: ", len(df))
df['v2a1'].hist()
sns.kdeplot(df['v2a1'])
sns.kdeplot(df['v18q1'])
sns.kdeplot(df['rez_esc'])
sns.kdeplot(df['meaneduc'])
sns.kdeplot(df['SQBmeaned'])
cols = list(df.columns)
cols
df.sample(10)
set(df.dtypes)
col_types = {}

for col in cols:
    col_types[col] = df[col].dtype
    # print(col, df[col].dtype)
# import collections

# od = collections.OrderedDict(sorted(col_types.items()))

#for k, v in od.items():
#    print(k, v)        # sorted columns by name 
# alternately we can use just sorted
# sorted(col_types)
print(len(col_types))
for key in sorted(col_types):
    print(key, col_types[key])
cat_cols = []
num_cols = []
for col in cols:
    if df[col].dtype == 'O':
        cat_cols.append(col)
        print(col, df[col].dtype)
    else:
        num_cols.append(col)
# categorical columns
cat_cols
df[cat_cols].sample(10)
len(num_cols)
# numerical columns
sorted(num_cols)
g = sns.PairGrid(df[nan_cols])
g = g.map_offdiag(plt.scatter)
cols_electronics = ['refrig','mobilephone','television','qmobilephone','computer', 'v18q', 'v18q1', ]
cols_house_details = ['v2a1', 'area1', 'area2', 'bedrooms','rooms', 'cielorazo', 'v14a', 
                    'tamhog', 'hacdor', 'hacapo', 'r4t3', ]
cols_person_details = ['age', 'agesq', 'female', 'male',]
cols_SQ = ['SQBage', 'SQBdependency', 'SQBedjefe', 'SQBescolari', 'SQBhogar_nin', 
           'SQBhogar_total', 'SQBmeaned', 'SQBovercrowding',]
cols_water = ['abastaguadentro', 'abastaguafuera', 'abastaguano',]

cols_h = [ 'hhsize', 'hogar_adul', 'hogar_mayor', 'hogar_nin', 'hogar_total',]
cols_r = ['r4h1', 'r4h2', 'r4h3', 'r4m1', 'r4m2', 'r4m3', 'r4t1', 'r4t2', 'r4t3',]
cols_tip = ['tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5',]
cols_roof = ['techocane', 'techoentrepiso', 'techootro', 'techozinc',]
cols_floor = ['pisocemento', 'pisomadera', 'pisomoscer', 'pisonatur', 'pisonotiene', 'pisoother',]
cols_sanitary = [ 'sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6',]
cols_parents = [ 'parentesco1', 'parentesco10', 'parentesco11', 'parentesco12', 'parentesco2', 'parentesco3',
                'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9',]
cols_outside_wall = [ 'paredblolad', 'pareddes', 'paredfibras', 'paredmad', 'paredother', 
              'paredpreb', 'paredzinc', 'paredzocalo',]
cols_instlevel = [ 'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6',
                  'instlevel7', 'instlevel8', 'instlevel9',]
cols_lugar = [ 'lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6',]
cols_estadoc = [ 'estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 
                'estadocivil5', 'estadocivil6', 'estadocivil7',]
cols_elim = ['elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6',]
cols_energ = ['energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4',]
cols_eviv = [ 'eviv1', 'eviv2', 'eviv3',]
cols_etech = [ 'etecho1', 'etecho2', 'etecho3',]
cols_pared = [ 'epared1', 'epared2', 'epared3',]
cols_unknown = [ 'dis', 'escolari', 'meaneduc', 
                'overcrowding', 'rez_esc', 'tamhog', 'tamviv', ]
cols_elec = ['coopele', 'noelec', 'planpri', 'public',]

total_features = cols_electronics+cols_house_details+cols_person_details+\
cols_SQ+cols_water+cols_h+cols_r+cols_tip+cols_roof+\
cols_floor+cols_sanitary+cols_parents+cols_outside_wall+\
cols_instlevel+cols_lugar+cols_estadoc+cols_elim+cols_energ+\
cols_eviv+cols_etech+cols_pared+cols_unknown+cols_elec

len(total_features)
df[cols_electronics].plot.area()
df['Target'].unique()
cols_electronics_target = cols_electronics.append('Target')
df[cols_electronics].corr()
cols_electronics.remove('Target')
cols_electronics
df.groupby('Target')[cols_electronics].sum()
df['tamhog'].unique()
# high correlation between 
# no. of persons in the household,
# persons living in the household 
# and size of the household
# we can use any one...!!
df[['tamhog','r4t3', 'tamviv']].corr()
df[['r4t3','tamviv']].corr()
total_features.remove('r4t3')
total_features.remove('tamhog')
total_features.remove('tamviv')

len(total_features)
df['escolari'].unique()
df['escolari'].hist()
df['escolari'].describe()
df['escolari'].plot.line()
sns.kdeplot(df.escolari)
correlations = df[num_cols].corr()
# correlation heatmap masking
mask = np.zeros_like(correlations, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(17, 13))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(correlations, mask=mask, cmap=cmap, vmax=.3, center=0,
square=True, linewidths=.5, cbar_kws={"shrink": .5})
# difficult to look into the above one
es_corr = df[num_cols].corrwith(df.escolari, axis=0)
for x,y in zip(num_cols, list(es_corr)):
    if (y >= 0.75) or (y < -0.6):
        print(x,y)
# escolari is highly correlated to SQBescolari
# let's see if SQBescolari is correlated to cols_house_details
sqbes_corr = df[num_cols].corrwith(df.SQBescolari, axis=0)
for x,y in zip(num_cols, list(sqbes_corr)):
    if (y >= 0.5) or (y < -0.6):
        print(x,y)
total_features.remove('escolari')
len(total_features)
df.loc[df.Target == 1].groupby('overcrowding').SQBescolari.value_counts().unstack().plot.bar()
df['overcrowding'].hist()
df['overcrowding'].unique()
df.plot.scatter(x='Target', y='overcrowding')
df.groupby('Target').overcrowding.value_counts().unstack().plot.bar()
df['Target'].describe()
df['Target'].unique()
df['Target'].hist()
nan_cols
# filling missing values

df[nan_cols].corr()
for col in nan_cols:
    if col != 'v2a1':
        print(col, df[col].unique())
# there's a clear quadratic relation between meaneduc and SQBmeaned
# hence, we can ignore either one of these..say, meaneduc
sns.regplot(df['meaneduc'],df['SQBmeaned'], order=2)
# filling na values in meaneduc and SQBmeaned
df['meaneduc'].fillna(0, inplace=True)
df['SQBmeaned'].fillna(0, inplace=True)
total_features.remove('meaneduc')

total_features
# we can fill v18q1 (household tablets) with 0 as individual tablet count is 0 for all such columns
df[['v18q','v18q1','idhogar']].loc[df.v18q1.isna()].describe()
df['v18q1'] = df['v18q'].groupby(df['idhogar']).transform('sum')
df.sample(7)
ff = pd.DataFrame(df.isnull().sum())
ff.loc[(ff.loc[:, ff.dtypes != object] != 0).any(1)]
# rez_esc - years behind in school
df['rez_esc'].describe()
df['rez_esc'].isnull().sum()
# only these many rows has values for years behind school
len(df) - df['rez_esc'].isnull().sum()
df['v2a1'].isnull().sum()
# only these many rows has values for income
len(df) - df['v2a1'].isnull().sum()
# number of rows where income and rez_esc has values
len(df.loc[(df.v2a1 >= 0)]), len(df.loc[(df.rez_esc >= 0)])
# how many rows with nan values for both income and rez_esc 
len(df.loc[(df.v2a1 >= 0) & (df.rez_esc >= 0)])
# how many rows with nan values for either income or rez_esc 
len(df.loc[(df.v2a1 >= 0) | (df.rez_esc >= 0)])
df['rez_esc'].hist()
df[['rez_esc','v2a1']].corr()
df[['Target','rez_esc']].corr()
df[['Target','rez_esc']].fillna(0).corr()
df[['v2a1','Target']].corr()
df[['v2a1','Target']].fillna(0).corr()
df['rez_esc'].unique()
plt.figure(figsize=(13,7))
sns.kdeplot(df['rez_esc'])
sns.kdeplot(df['rez_esc'].fillna(0))
plt.figure(figsize=(13,7))
sns.kdeplot(df['v2a1'])
sns.kdeplot(df['v2a1'].fillna(0))
x, y = df['rez_esc'], df['v2a1']
plt.figure(figsize=(23,17))
g = sns.jointplot(x, y, data=df, kind="kde", color="m")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$X$", "$Y$")
x, y = df['rez_esc'].fillna(0), df['v2a1'].fillna(0)
sns.jointplot(x, y, data=df, kind="kde")
df['rez_esc'].fillna(0, inplace=True)
ff = pd.DataFrame(df.isnull().sum())
ff.loc[(ff.loc[:, ff.dtypes != object] != 0).any(1)]
x, y = df['Target'], df['v2a1']
plt.figure(figsize=(23,17))
g = sns.jointplot(x, y, data=df, kind="kde", color="m")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$X$", "$Y$")
x, y = df['Target'], df['v2a1'].fillna(0)
plt.figure(figsize=(23,17))
g = sns.jointplot(x, y, data=df, kind="kde", color="m")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$X$", "$Y$")
df['Target'].value_counts()
df.groupby('Target').count()['v2a1']
fig, ax = plt.subplots(figsize=(15,7))
df.groupby('Target').count()['v2a1'].plot(ax=ax)
fig, ax = plt.subplots(figsize=(15,7))
df.fillna(0).groupby('Target').count()['v2a1'].plot(ax=ax)
fig, ax = plt.subplots(figsize=(15,7))
df.groupby(['Target','hhsize']).count()['v2a1'].unstack().plot(ax=ax)
fig, ax = plt.subplots(figsize=(15,7))
df.fillna(0).groupby(['Target','hhsize']).count()['v2a1'].unstack().plot(ax=ax)
df['hhsize'].value_counts()
df['hogar_total'].value_counts()
# use hhsize, ignore 'hogar_total',
total_features.remove('hogar_total')

len(total_features)
df[['hhsize','hogar_adul']].corr()
df[['hhsize','Target']].corr()
df[['Target','hogar_adul']].corr()
sns.kdeplot(df['hogar_adul'])
sns.kdeplot(df['hhsize'])

sns.kdeplot(df['hogar_total'])
sns.kdeplot(df['hogar_adul'])
max(df['hogar_adul']), max(df['hogar_total'])
df.groupby('idhogar').sum()[['hogar_adul','hogar_total']].sample(10).plot.bar()
sns.kdeplot(df['hogar_total'])
sns.kdeplot(df['hogar_nin'])
df['male'].value_counts()
df['female'].value_counts()
# removing female
total_features.remove('female')

len(total_features)
df['r4t3'].value_counts()
df['tamhog'].value_counts()
df['tamviv'].value_counts()
plt.figure(figsize=(17,13))
sns.kdeplot(df['tamviv'])
sns.kdeplot(df['tamhog'])
sns.kdeplot(df['r4t3'])
sns.kdeplot(df['hhsize'])
sns.kdeplot(df['hogar_total'])
#sns.kdeplot(df['hogar_adul'])

# removing 'r4t3', as 'hhsize' is of almost same distribution
total_features.remove('r4t3')
len(total_features)
df['dependency'].describe()
df['dependency'].value_counts()
df['SQBdependency'].value_counts()
df['SQBdependency'].describe()
cat_cols
df['edjefe'].describe()
df['edjefa'].describe()
df['edjefe'].value_counts()
df['edjefa'].value_counts()
df.loc[df.edjefa == 'yes', 'edjefa'] = 1
df.loc[df.edjefa == 'no', 'edjefa'] = 0

df.loc[df.edjefe == 'yes', 'edjefe'] = 1
df.loc[df.edjefe == 'no', 'edjefe'] = 0

df[['edjefa','edjefe']].describe()
df[['edjefa','edjefe']] = df[['edjefa','edjefe']].apply(pd.to_numeric)

df[['edjefa','edjefe']].dtypes
len(total_features)
total_features.append('edjefa')
total_features.append('edjefe')
len(total_features)
cols_water
df[cols_water].describe()
df[cols_water].corr()
df['abastaguadentro'].value_counts()
df['abastaguafuera'].value_counts()
df['abastaguano'].value_counts()
df_water_target = df.groupby('Target')[cols_water].sum().reset_index()
df_water_target
722+1496+1133+5844
df_water_target.corr()
len(total_features)

total_features.remove('abastaguano')
total_features.remove('abastaguafuera')

len(total_features)
# cols_floor
# 
# ['pisocemento', 'pisomadera', 'pisomoscer', 'pisonatur', 'pisonotiene', 'pisoother',]

df['pisocemento'].value_counts()
df_floor_target = df.groupby('Target')[cols_floor].sum().reset_index()
df_floor_target
len(total_features)
# removing these features -> inc by 0.002
total_features.remove('pisonatur')
total_features.remove('pisonotiene')
total_features.remove('pisoother')

len(total_features)
# cols_outside_wall

# [ 'paredblolad', 'pareddes', 'paredfibras', 'paredmad', 'paredother', 'paredpreb', 'paredzinc', 'paredzocalo',]
df_wall_target = df.groupby('Target')[cols_outside_wall].sum().reset_index()
df_wall_target
sns.kdeplot(df['paredblolad'])

sns.kdeplot(df['paredpreb'])
sns.kdeplot(df['paredmad'])

sns.kdeplot(df['paredmad'])
sns.kdeplot(df['paredzocalo'])

sns.kdeplot(df['paredpreb'])
sns.kdeplot(df['paredmad'])
sns.kdeplot(df['paredzocalo'])

len(total_features)
# removing these features -> reached time limit

# total_features.remove('pareddes')
# total_features.remove('paredfibras')
# total_features.remove('paredother')
# total_features.remove('paredzinc')
# total_features.remove('paredzocalo')

# len(total_features)
# from here till model -> do not submit 
# cols_roof
# ['techocane', 'techoentrepiso', 'techootro', 'techozinc',]
df_roof_target = df.groupby('Target')[cols_roof].sum().reset_index()
df_roof_target
sns.kdeplot(df['techozinc'])
sns.kdeplot(df['techozinc'])
sns.kdeplot(df['techoentrepiso'])

sns.kdeplot(df['techoentrepiso'])
sns.kdeplot(df['techocane'])
len(total_features)

# remove these features -> (1)
# total_features.remove('techootro')
# total_features.remove('techocane')

# len(total_features)

# [ 'sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6',]

df_sani_target = df.groupby('Target')[cols_sanitary].sum().reset_index()
df_sani_target
sns.kdeplot(df['sanitario1'])
sns.kdeplot(df['sanitario6'])

sns.kdeplot(df['sanitario3'])
sns.kdeplot(df['sanitario2'])
len(total_features)

# remove these features -> (2)
# total_features.remove('sanitario1')
# total_features.remove('sanitario5')
# total_features.remove('sanitario6')

# len(total_features)

# cols_tip 
# ['tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5',]

df_tipo_target = df.groupby('Target')[cols_tip].sum().reset_index()
df_tipo_target
sns.kdeplot(df['tipovivi2'])

sns.kdeplot(df['tipovivi1'])
sns.kdeplot(df['tipovivi3'])

sns.kdeplot(df['tipovivi5'])
sns.kdeplot(df['tipovivi4'])

df['v2a1'].isna().sum()
df['tipovivi3'].value_counts()
df.loc[(df['v2a1'].isna()) & (df.tipovivi3 == 1)]
# check the value of parentesco1 and fill corresponding value
# group by idhogar -> check and fill 
df['v2a1'].loc[df.parentesco1 == 1].plot.line()
df['v2a1'].loc[df.parentesco1 == 1].plot.hist()
df['v2a1'].loc[df.parentesco1 == 1].mean(), df['v2a1'].loc[df.parentesco1 == 1].max(), df['v2a1'].loc[df.parentesco1 == 1].min()
df[['v2a1','idhogar','parentesco1']].loc[(df.parentesco1 == 1) & (df.v2a1.isna())].describe(include='all')
df[['v2a1','idhogar','parentesco1']].loc[(df.parentesco1 == 1) & (-df.v2a1.isna()) & (df.tipovivi3==1) & (df.tipovivi1==0)].describe(include='all')
df[['v2a1','idhogar','parentesco1']].loc[(df.parentesco1 == 1) & (-df.v2a1.isna()) & (df.tipovivi3==1) & (df.tipovivi1==0)].mean()
df[['v2a1','idhogar','parentesco1']].loc[(df.parentesco1 != 1) & (-df.v2a1.isna()) & (df.tipovivi3==1) & (df.tipovivi1==0)].mean()
df[['v2a1','idhogar','parentesco1']].loc[(df.parentesco1 != 1) & (-df.v2a1.isna()) & (df.tipovivi3==1) & (df.tipovivi1==0)].describe(include='all')
df[['v2a1','idhogar','parentesco1']].loc[df.parentesco1 != 1].describe(include='all')
df[['v2a1','idhogar','parentesco1']].loc[df.parentesco1 == 1].describe(include='all')
# 50% of the samples have ~120000 as the monthly rent..
# 
df['v2a1'].fillna(120000, inplace=True)
df['v2a1'].isna().sum()




'Target' in total_features
X, y = df[total_features], df['Target']
#Split the dataset into train and Test
seed = 42
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
#Train the XGboost Model for Classification
model1 = xgb.XGBClassifier(n_jobs=n_jobs)
model1
train_model1 = model1.fit(X_train, y_train)
model2 = xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, subsample=0.5, n_jobs=n_jobs)
model2
train_model2 = model2.fit(X_train, y_train)
# predictions
pred1 = train_model1.predict(X_test)
pred2 = train_model2.predict(X_test)

print('Model 1 XGboost Report %r' % (classification_report(y_test, pred1)))
print('Model 2 XGboost Report %r' % (classification_report(y_test, pred2)))
print("Accuracy for model 1: %.2f" % (accuracy_score(y_test, pred1) * 100))
print("Accuracy for model 2: %.2f" % (accuracy_score(y_test, pred2) * 100))
#Let's do a little Gridsearch, Hyperparameter Tunning
model3 = xgb.XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 n_jobs=n_jobs,
 scale_pos_weight=1,
 seed=27)
train_model3 = model3.fit(X_train, y_train)
pred3 = train_model3.predict(X_test)
print("Accuracy for model 3: %.2f" % (accuracy_score(y_test, pred3) * 100))
print('Model 3 XGboost Report %r' % (classification_report(y_test, pred3)))
gc.collect()


parameters = {
    'n_estimators': [100],
    'max_depth': [6, 9],
    'subsample': [0.9, 1.0],
    'colsample_bytree': [0.9, 1.0],
}

grid = GridSearchCV(model3,
                    parameters, n_jobs=n_jobs,
                    scoring="neg_log_loss",
                    cv=3)
grid
# grid.fit(X_train, y_train)
# print("Best: %f using %s" % (grid.best_score_, grid.best_params_))

#means = grid.cv_results_['mean_test_score']
#stds = grid.cv_results_['std_test_score']
#params = grid.cv_results_['params']

#for mean, stdev, param in zip(means, stds, params):
#    print("%f (%f) with: %r" % (mean, stdev, param))
#pred4 = grid.predict(X_test)
#classification_report(y_test, pred4)
#print("Accuracy for model 4: %.2f" % (accuracy_score(y_test, pred4) * 100))
gc.collect()
fig, ax = plt.subplots(figsize=(23, 17))
plot_importance(model3, ax=ax)
less_imp_features = ['estadocivil1','instlevel9','techocane','parentesco10','v14a',
                     'parentesco11','parentesco5','paredother','parentesco7','noelec',
                     'elimbasu4','elimbasu6']

# before removing less important features
len(total_features)
for f in less_imp_features:
    if f in total_features:
        total_features.remove(f)

len(total_features)
X, y = df[total_features], df['Target']
#Split the dataset into train and Test
seed = 43
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
train_model5 = model3.fit(X_train, y_train)
pred5 = train_model5.predict(X_test)
print("Accuracy for model 5: %.2f" % (accuracy_score(y_test, pred5) * 100))
print('Model 5 XGboost Report %r' % (classification_report(y_test, pred5)))
train_model6 = train_model5.fit(X, y)

train_model6

# plot_tree(model3)
# scoring = ['precision_macro', 'recall_macro']
# scores = cross_validate(model3, X_train, y_train, cv=5, scoring=scoring)
# scores
# sorted(scores.keys())
# scoring = {'precision_macro': 'precision_macro', 'recall_macro': make_scorer(recall_score, average='macro')}
# scores = cross_validate(model3, X_train, y_train, cv=7, scoring=scoring)
# scores
# 

# prediction using cross_val_predict()
# predicted = cross_val_predict(model3, X_test, y_test, cv=10)
# accuracy_score(y_test, predicted)
thresholds = sort(model3.feature_importances_)
thresholds
thresholds.shape
np.unique(thresholds).shape
model4 = RandomForestClassifier(n_jobs=n_jobs)
model4
gc.collect()
train_model4 = model4.fit(X_train, y_train)
pred4 = train_model4.predict(X_test)
print("Accuracy for model 4: %.2f" % (accuracy_score(y_test, pred4) * 100))

print('Model 4 XGboost Report %r' % (classification_report(y_test, pred4)))
confusion_matrix(y_test, pred4)



df_test = pd.read_csv('../input/test.csv')
len(df_test)
df_test.sample(10)
# considering only head of household
# df_test = df_test.loc[df_test.parentesco1 == 1]
# 
len(df_test)
df_test.loc[df_test.edjefa == 'yes', 'edjefa'] = 1
df_test.loc[df_test.edjefa == 'no', 'edjefa'] = 0

df_test.loc[df_test.edjefe == 'yes', 'edjefe'] = 1
df_test.loc[df_test.edjefe == 'no', 'edjefe'] = 0
df_test[['edjefa','edjefe']] = df_test[['edjefa','edjefe']].apply(pd.to_numeric)
df_test[['edjefa','edjefe']].dtypes
X_actual_test = df_test[total_features]
X_actual_test.shape
pred_actual = train_model6.predict(X_actual_test)
pred_actual
pred_actual.shape
df_final = pd.DataFrame(df['Id'], pred_actual).reset_index()
df_final.columns = ['Target','Id']

cols = df_final.columns.tolist()
cols
cols = cols[-1:] + cols[:-1]
cols

df_final = df_final[cols]
df_final.head(7)
df_final.index.name = None
df_final.head(7)
df_final['Target'].value_counts()

df_final[cols].sample(4)
df_final[cols].to_csv('sample_submission.csv', index=False)
os.listdir('../input/')
