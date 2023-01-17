# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# imports 

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.impute import SimpleImputer
from sklearn.model_selection import (train_test_split, StratifiedKFold, 
                                     cross_val_score, cross_validate, 
                                     cross_val_predict, GridSearchCV, RandomizedSearchCV)
from sklearn import (preprocessing as pp,
                    feature_selection as fs)
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier)
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (roc_auc_score, auc, roc_curve, 
                            confusion_matrix, f1_score, log_loss, 
                            classification_report)

import missingno as msno 

from scipy.stats import chi2_contingency

from statsmodels.stats import weightstats as stests

from scipy import stats

from xgboost import XGBClassifier

df = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
df_train, df_test = train_test_split(df, test_size=0.20, random_state=42, stratify=df['TARGET'])
set(df_train.dtypes)
#separating columns by type

binary_features = [col for col in df_train.drop(columns=['TARGET', 'SK_ID_CURR']).columns.values if df_train[col].nunique()<=2]

num_features = df_train.drop(columns=['TARGET', 'SK_ID_CURR']).select_dtypes(include=['int64', 'float64']).columns 
num_features = list(set(num_features) - set(binary_features))

cat_features = df_train.drop(columns=['TARGET', 'SK_ID_CURR']).select_dtypes(include='O').columns
cat_features = list(set(cat_features) - set(binary_features))

title = str(df_train['TARGET'].value_counts(normalize=True))
df_train['TARGET'].value_counts(normalize=True).plot.bar(title=title)
#plotting missing numerical features values
plt.figure(figsize=(10,40))
sns.barplot(x=df_train.count()[:],y=df_train.count().index)
plt.xlabel('Non-Null Values Count')
plt.ylabel('Numerical Features')
# showing percentage of missing data per feature
missing_p_feat= {feat: round(df_train[feat].isnull().mean(), 3) for feat in df_train.columns.values if df_train[feat].isnull().sum()>0}

# frequency of columns per missing percentage
plt.hist( np.array(list(missing_p_feat.values())), bins=70)
sorted([(value,key) for (key,value) in missing_p_feat.items()])
msno.dendrogram(df_train)
housig_features_w_missing = [ 
        'LANDAREA_AVG',
        'LANDAREA_MEDI',
        'LANDAREA_MODE',
        'BASEMENTAREA_AVG',
        'BASEMENTAREA_MEDI',
        'BASEMENTAREA_MODE',
        'YEARS_BEGINEXPLUATATION_AVG',
        'YEARS_BEGINEXPLUATATION_MEDI',
        'YEARS_BEGINEXPLUATATION_MODE',
        'TOTALAREA_MODE',
        'EMERGENCYSTATE_MODE',
        'LIVINGAREA_AVG',
        'LIVINGAREA_MEDI',
        'LIVINGAREA_MODE',
        'FLOORSMAX_AVG',
        'FLOORSMAX_MEDI',
        'FLOORSMAX_MODE',
        'ENTRANCES_AVG',
        'ENTRANCES_MEDI',
        'ENTRANCES_MODE',
        'WALLSMATERIAL_MODE',
        'HOUSETYPE_MODE',
        'APARTMENTS_AVG',
        'APARTMENTS_MEDI',
        'APARTMENTS_MODE',    
        'ELEVATORS_AVG',
        'ELEVATORS_MEDI',
        'ELEVATORS_MODE',
        'NONLIVINGAPARTMENTS_AVG',
        'NONLIVINGAPARTMENTS_MEDI',
        'NONLIVINGAPARTMENTS_MODE',
        'LIVINGAPARTMENTS_AVG',
        'YEARS_BUILD_AVG',
        'YEARS_BUILD_MEDI',
        'YEARS_BUILD_MODE',
        'FLOORSMIN_AVG',
        'FLOORSMIN_MEDI',
        'FLOORSMIN_MODE',
        'NONLIVINGAREA_AVG',
        'NONLIVINGAREA_MEDI',
        'NONLIVINGAREA_MODE',
        'LIVINGAPARTMENTS_MEDI',
        'LIVINGAPARTMENTS_MODE',
        'FONDKAPREMONT_MODE',
        'COMMONAREA_AVG',
        'COMMONAREA_MEDI',
        'COMMONAREA_MODE']
all_housing_missing = df_train[housig_features_w_missing].isnull().all(axis=1)
ratio_on_housing_miss = df_train['TARGET'][all_housing_missing].mean()
ratio_on_housing_avail = df_train['TARGET'][~all_housing_missing].mean()

print(f'default rate on missing housing data aplicants: {ratio_on_housing_miss:.2%}')
print(f'default rate on available housing data aplicants: {ratio_on_housing_avail:.2%}')
cont = pd.crosstab( all_housing_missing, df_train['TARGET'])

test = chi2_contingency(cont, lambda_="log-likelihood")
p_val = test[1]
is_stats_diff = p_val<0.03

print(f'is there statistical significance: {is_stats_diff}')
num_housing = list(set(num_features).intersection(set(housig_features_w_missing)))
df_train[num_housing].hist(figsize=(25,18), bins=50)
xtreme_inputer = SimpleImputer(fill_value=9999, strategy="constant")

df_train[num_housing] = xtreme_inputer.fit_transform(df_train[num_housing])
df_test[num_housing] = xtreme_inputer.transform(df_test[num_housing])
# replacing missing values in categorical features
cat_housing = list(set(cat_features).intersection(set(housig_features_w_missing)))
cat_housing
fig, axes = plt.subplots(nrows=3, figsize=(10, 8))
for i, col in enumerate(cat_housing):
    df_train[col].hist(ax=axes[i])
plt.show()
df_train[cat_housing] = df_train[cat_housing].fillna("MISSING")
df_test[cat_housing] = df_test[cat_housing].fillna("MISSING")
fig, axes = plt.subplots(nrows=3, figsize=(10, 8))
for i, col in enumerate(cat_housing):
    df_train[col].hist(ax=axes[i])
plt.show()
#Listing the rest of the feature with missing values

sorted([(value,key) for (key,value) in missing_p_feat.items() if key not in housig_features_w_missing])
to_drop_missing = sorted([key for (key,value) in missing_p_feat.items() if key not in housig_features_w_missing and value < 0.01])
to_drop_missing
# dropping values where missing <1%

df_train = df_train.dropna(axis=0, how='any', subset=to_drop_missing)
df_test = df_test.dropna(axis=0, how='any', subset=to_drop_missing)
rest_of_missing = sorted([key for (key,value) in missing_p_feat.items() if key not in housig_features_w_missing and value>0.01])
rest_of_missing
df_train['OCCUPATION_TYPE'].value_counts().plot.bar()
df_train['OCCUPATION_TYPE'] = df_train['OCCUPATION_TYPE'].fillna("MISSING")
df_test['OCCUPATION_TYPE'] = df_test['OCCUPATION_TYPE'].fillna("MISSING")
df_train['OCCUPATION_TYPE'].value_counts().plot.bar()
df_train[rest_of_missing[:6]].hist(figsize=(8,8))
missing_req_credit = df_train[rest_of_missing[:6]].isnull().any(axis=1)
ratio_on_missing_req_credit = df_train['TARGET'][missing_req_credit].mean()
ratio_on_avail_req_credit = df_train['TARGET'][~missing_req_credit].mean()

print(f'default rate on missing credit requests: {ratio_on_missing_req_credit:.2%}')
print(f'default rate on available credit requests: {ratio_on_avail_req_credit:.2%}')

cont = pd.crosstab(missing_req_credit, df_train['TARGET'])

test = chi2_contingency(cont, lambda_="log-likelihood")
p_val = test[1]
is_stats_diff = p_val<0.03

print(f'is there statistical significance: {is_stats_diff}')
print(f'p: {p_val}')
req_credit = [
    'AMT_REQ_CREDIT_BUREAU_DAY',
    'AMT_REQ_CREDIT_BUREAU_HOUR',
    'AMT_REQ_CREDIT_BUREAU_MON',
    'AMT_REQ_CREDIT_BUREAU_QRT',
    'AMT_REQ_CREDIT_BUREAU_WEEK',
    'AMT_REQ_CREDIT_BUREAU_YEAR'
]

df_train[req_credit] = xtreme_inputer.fit_transform(df_train[req_credit])
df_test[req_credit] = xtreme_inputer.transform(df_test[req_credit])
for col in ['EXT_SOURCE_1','EXT_SOURCE_3','OWN_CAR_AGE']:
    missing = df_train[col].isnull()
    missing_rate = df_train['TARGET'][missing].mean()
    not_missing_rate = df_train['TARGET'][~missing].mean()
    
    print(col)
    print(f'default rate on missing: {missing_rate:.2%}')
    print(f'default rate on available: {not_missing_rate:.2%}')

    cont = pd.crosstab(missing, df_train['TARGET'])

    test = chi2_contingency(cont, lambda_="log-likelihood")
    p_val = test[1]
    is_stats_diff = p_val<0.03

    print(f'Is there statistical significance: {is_stats_diff}')
    print(f'p val: {p_val}')
    print('_'*20, '\n')
    
df_train[['EXT_SOURCE_1','EXT_SOURCE_3','OWN_CAR_AGE']].hist(figsize=(8,8))
cols = ['EXT_SOURCE_1','EXT_SOURCE_3','OWN_CAR_AGE']

df_train[cols] = xtreme_inputer.fit_transform(df_train[cols])
df_test[cols] = xtreme_inputer.transform(df_test[cols])
#checking if there's any remaining features with missing values
remaining_missing = df_train.columns[np.where(df_train.isnull().sum()!=0)]
remaining_missing
df_train['EMERGENCYSTATE_MODE'].hist(figsize=(5, 5))
df_train = df_train.dropna(axis=0, how='any', subset=remaining_missing)
df_test = df_test.dropna(axis=0, how='any', subset=remaining_missing)
df_train[cat_features].head(5)
sns.set(style="whitegrid")

def show_plots(feature, figsize=None):
    if not figsize:
        figsize = (20, 5)
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(feature)
    
    order = df_train[[feature,'TARGET']].groupby(feature)['TARGET'].mean().sort_values().keys()

    ct = sns.countplot(data=df_train, y=feature, ax=ax[0], order=order)
    ct.set_title("COUNT")
    ax[0].set_xlabel('')
    ax[0].set_ylabel('')
    

    dfr = sns.barplot(y=feature, x="TARGET", data=df_train, ax=ax[1], order=order)
    dfr.set_title("DEFAULT %")
    dfr.set(yticklabels=list())
    ax[1].set_xlabel('')
    ax[1].set_ylabel('')

df_train = df_train.drop(df_train.loc[~df_train['CODE_GENDER'].isin(['F','M'])].index)
df_test = df_test.drop(df_test.loc[~df_test['CODE_GENDER'].isin(['F','M'])].index)
show_plots("CODE_GENDER", figsize = (15,2))
df_train["CODE_GENDER"] = df_train["CODE_GENDER"].replace({'F':0, 'M':1})
df_test["CODE_GENDER"] = df_test["CODE_GENDER"].replace({'F':0, 'M':1})
show_plots("WALLSMATERIAL_MODE", figsize = (15,3))
vc = dict(df_train["WALLSMATERIAL_MODE"].value_counts(normalize=True))
rare_lb = [k for k,v in vc.items() if v<=0.05]

df_train["WALLSMATERIAL_MODE"] = df_train["WALLSMATERIAL_MODE"].replace(rare_lb,'RARE')
df_test["WALLSMATERIAL_MODE"] = df_test["WALLSMATERIAL_MODE"].replace(rare_lb,'RARE')
show_plots("WALLSMATERIAL_MODE", figsize = (15,3))
#One hot encode WALLSMATERIAL_MODE

df_train = pd.concat([df_train.drop('WALLSMATERIAL_MODE', axis=1), 
                       pd.get_dummies(df_train['WALLSMATERIAL_MODE'], prefix='WALLSMATERIAL_MODE')], 
                          axis=1)

df_test = pd.concat([df_test.drop('WALLSMATERIAL_MODE', axis=1), 
                       pd.get_dummies(df_test['WALLSMATERIAL_MODE'], prefix='WALLSMATERIAL_MODE')], 
                          axis=1)
show_plots("HOUSETYPE_MODE", figsize=(15, 2))
vc = dict(df_train["HOUSETYPE_MODE"].value_counts(normalize=True))
rare_lb = [k for k,v in vc.items() if v<=0.5]

df_train["HOUSETYPE_MODE"] = df_train["HOUSETYPE_MODE"].replace(rare_lb,'RARE')
df_test["HOUSETYPE_MODE"] = df_test["HOUSETYPE_MODE"].replace(rare_lb,'RARE')
show_plots("HOUSETYPE_MODE", figsize=(15, 2))
#One hot encode HOUSETYPE_MODE

df_train = pd.concat([df_train.drop('HOUSETYPE_MODE', axis=1), 
                       pd.get_dummies(df_train['HOUSETYPE_MODE'], prefix='HOUSETYPE_MODE')], 
                          axis=1)

df_test = pd.concat([df_test.drop('HOUSETYPE_MODE', axis=1), 
                       pd.get_dummies(df_test['HOUSETYPE_MODE'], prefix='HOUSETYPE_MODE')], 
                          axis=1)
show_plots("NAME_INCOME_TYPE", figsize=(10, 3))
df_train['NAME_INCOME_TYPE'].value_counts()
df_train["NAME_INCOME_TYPE"] = np.where(df_train["NAME_INCOME_TYPE"]=='Working', 1, 0)
df_test["NAME_INCOME_TYPE"] = np.where(df_test["NAME_INCOME_TYPE"]=='Working', 1, 0)
show_plots("ORGANIZATION_TYPE", figsize=(20, 15))
df_train = df_train.drop("ORGANIZATION_TYPE", axis=1)
df_test = df_test.drop("ORGANIZATION_TYPE", axis=1)
show_plots("NAME_FAMILY_STATUS", figsize=(15, 2))
set(df_train["NAME_FAMILY_STATUS"])^set(df_test["NAME_FAMILY_STATUS"])
# one hot encode NAME_FAMILY_STATUS

df_train = pd.concat([df_train.drop('NAME_FAMILY_STATUS', axis=1), 
                       pd.get_dummies(df_train['NAME_FAMILY_STATUS'], prefix='NAME_FAMILY_STATUS')], 
                          axis=1)

df_test = pd.concat([df_test.drop('NAME_FAMILY_STATUS', axis=1), 
                       pd.get_dummies(df_test['NAME_FAMILY_STATUS'], prefix='NAME_FAMILY_STATUS')], 
                          axis=1)
show_plots("FONDKAPREMONT_MODE", figsize=(15, 2))
vc = dict(df_train["FONDKAPREMONT_MODE"].value_counts(normalize=True))
rare_lb = [k for k,v in vc.items() if v<=0.1]

df_train["FONDKAPREMONT_MODE"] = df_train["FONDKAPREMONT_MODE"].replace(rare_lb,'RARE')
df_test["FONDKAPREMONT_MODE"] = df_test["FONDKAPREMONT_MODE"].replace(rare_lb,'RARE')
df_train = df_train.drop("FONDKAPREMONT_MODE", axis=1)
df_test = df_test.drop("FONDKAPREMONT_MODE", axis=1)
show_plots("OCCUPATION_TYPE", figsize=(15, 5))
df_train['OCCUPATION_TYPE'].value_counts(1)
df_train['OCCUPATION_TYPE'] = np.where(df_train['OCCUPATION_TYPE']=='MISSING',0,1)
df_test['OCCUPATION_TYPE'] = np.where(df_test['OCCUPATION_TYPE']=='MISSING',0,1)
show_plots("WEEKDAY_APPR_PROCESS_START", figsize=(15, 3))
# one hot encode WEEKDAY_APPR_PROCESS_START

df_train = pd.concat([df_train.drop('WEEKDAY_APPR_PROCESS_START', axis=1), 
                       pd.get_dummies(df_train['WEEKDAY_APPR_PROCESS_START'], prefix='WEEKDAY_APPR_PROCESS_START')], 
                          axis=1)

df_test = pd.concat([df_test.drop('WEEKDAY_APPR_PROCESS_START', axis=1), 
                       pd.get_dummies(df_test['WEEKDAY_APPR_PROCESS_START'], prefix='WEEKDAY_APPR_PROCESS_START')], 
                          axis=1)
show_plots("NAME_TYPE_SUITE", figsize=(15,3))
df_train["NAME_TYPE_SUITE"] = np.where(df_train["NAME_TYPE_SUITE"]=='Unaccompanied', 0, 1) 
df_test["NAME_TYPE_SUITE"] = np.where(df_test["NAME_TYPE_SUITE"]=='Unaccompanied', 0, 1) 
show_plots("NAME_HOUSING_TYPE", figsize=(15,3))
set(df_train['NAME_HOUSING_TYPE'])^set(df_train['NAME_HOUSING_TYPE'])
vc = dict(df_train["NAME_HOUSING_TYPE"].value_counts(normalize=True))
rare_lb = [k for k,v in vc.items() if v<=0.04]

df_train = df_train[~df_train["NAME_HOUSING_TYPE"].isin(rare_lb)]
df_test = df_test[~df_test["NAME_HOUSING_TYPE"].isin(rare_lb)]

show_plots("NAME_HOUSING_TYPE", figsize=(15,3))
# one hot encode NAME_HOUSING_TYPE

df_train = pd.concat([df_train.drop('NAME_HOUSING_TYPE', axis=1), 
                       pd.get_dummies(df_train['NAME_HOUSING_TYPE'], prefix='NAME_HOUSING_TYPE')], 
                          axis=1)

df_test = pd.concat([df_test.drop('NAME_HOUSING_TYPE', axis=1), 
                       pd.get_dummies(df_test['NAME_HOUSING_TYPE'], prefix='NAME_HOUSING_TYPE')], 
                          axis=1)
show_plots("NAME_EDUCATION_TYPE", figsize=(15,3))
df_train['NAME_EDUCATION_TYPE'].value_counts(1)
set(df_train['NAME_EDUCATION_TYPE'])^set(df_train['NAME_EDUCATION_TYPE'])
df_train['NAME_EDUCATION_TYPE'].unique()
mapping = {
'Higher education': 2,
'Secondary / secondary special' : 1,
'Incomplete higher' : 1,
'Lower secondary' : 1,
'Academic degree': 2
}

df_train["NAME_EDUCATION_TYPE"] = df_train["NAME_EDUCATION_TYPE"].map(mapping)
df_test["NAME_EDUCATION_TYPE"] = df_test["NAME_EDUCATION_TYPE"].map(mapping)

df_train[['NAME_EDUCATION_TYPE','TARGET']].groupby('NAME_EDUCATION_TYPE')['TARGET'].value_counts(1)
cols = list(set(cat_features).intersection(set(df_train.columns.values)))
cols
df_train[binary_features].head()
for feature in binary_features:
    show_plots(feature, figsize=(15,2))
df_train["NAME_CONTRACT_TYPE"] = df_train["NAME_CONTRACT_TYPE"].replace({'Cash loans':1, 'Revolving loans':0})
df_test["NAME_CONTRACT_TYPE"] = df_test["NAME_CONTRACT_TYPE"].replace({'Cash loans':1, 'Revolving loans':0}) 

df_train["FLAG_OWN_CAR"] = df_train["FLAG_OWN_CAR"].replace({'Y':1, 'N':0})
df_test["FLAG_OWN_CAR"] = df_test["FLAG_OWN_CAR"].replace({'Y':1, 'N':0})

df_train["FLAG_OWN_REALTY"] = df_train["FLAG_OWN_REALTY"].replace({'Y':1, 'N':0})
df_test["FLAG_OWN_REALTY"] = df_test["FLAG_OWN_REALTY"].replace({'Y':1, 'N':0})

df_train["EMERGENCYSTATE_MODE"] = df_train["EMERGENCYSTATE_MODE"].replace({'Yes':1, 'No':0})
df_test["EMERGENCYSTATE_MODE"] = df_test["EMERGENCYSTATE_MODE"].replace({'Yes':1, 'No':0})
var = fs.VarianceThreshold(0.02)
var.fit(df_train)

non_informative = df_train.columns[~var.get_support()]
non_informative
df_train[non_informative].hist(bins=30, figsize=(20,20))
non_informative = list(filter(lambda x: x!='REGION_POPULATION_RELATIVE', non_informative))

df_train = df_train.drop(non_informative, 1)
df_test = df_test.drop(non_informative, 1)
correlation = df_train.corr().abs()
sns.clustermap(correlation, cmap='coolwarm', 
               vmin=0, vmax=0.8, center=0, 
               square=True, linewidths=.5, 
               figsize=(50,50), yticklabels=1)
corr_mat = correlation.unstack() 
corr_mat = corr_mat.sort_values(ascending=False)
corr_mat = corr_mat[(corr_mat >= 0.8) & (corr_mat < 1)]
corr_mat = pd.DataFrame(corr_mat).reset_index()
corr_mat.columns = ['f1', 'f2', 'correlation']
corr_mat.head()
grouped_feature = set()
correlated_groups = list()

for feature in corr_mat.f1.unique():
    if feature not in grouped_feature:
        correlated_block = corr_mat[corr_mat.f1==feature]
        grouped_feature = grouped_feature | set(correlated_block.f2) | set(feature)
        correlated_groups.append(correlated_block)
        
print(f'correlated groups count: {len(correlated_groups)}')
for i, group in enumerate(correlated_groups):
    print(f'\tgroup {i+1}:\n{group}\n\n')
importance_groups = list()

for group in correlated_groups:
    group_features = list(set(group.f1) | set(group.f2))
    
    rf = RandomForestClassifier(n_estimators=200, random_state=39, max_depth=4, verbose=0)
    
    f_import = rf.fit(df_train[group_features], 
                       df_train['TARGET']).feature_importances_
    imp_g = pd.DataFrame({'feature': group_features,
                            'importance': f_import}).sort_values(by='importance', ascending=False)
    importance_groups.append(imp_g)
    
print(f'total importance groups: {len(importance_groups)}\n\n')
for i, g in enumerate(importance_groups):
    print(f'img_g: {i}\n {g}\n\n')
not_imp_features = set()

for g in importance_groups:
    not_imp_features |= set(g.feature[1:])
df_train = df_train.drop(columns=list(not_imp_features)).reset_index()
df_test = df_test.drop(columns=list(not_imp_features)).reset_index()
updated_numericals = list(set(num_features).intersection(set(df_train.columns.values)))
df_train[updated_numericals].hist(figsize=(20,20), bins=50)
updated_numericals.remove('CNT_CHILDREN')
updated_numericals.remove('REGION_RATING_CLIENT_W_CITY')
ss = pp.StandardScaler()
ss.fit(df_train[updated_numericals])

df_train[updated_numericals] = ss.transform(df_train[updated_numericals])
df_test[updated_numericals] = ss.transform(df_test[updated_numericals])
show_plots("CNT_CHILDREN", figsize=(15,3))
vc = dict(df_train["CNT_CHILDREN"].value_counts(normalize=True))
rare_lb = [k for k,v in vc.items() if v<=0.05]

df_train = df_train[~df_train["CNT_CHILDREN"].isin(rare_lb)]
df_test = df_test[~df_test["CNT_CHILDREN"].isin(rare_lb)]

show_plots("CNT_CHILDREN", figsize=(15,3))
df_train[['CNT_CHILDREN', 'TARGET']].groupby('CNT_CHILDREN')['TARGET'].mean()
oh = pp.OneHotEncoder(handle_unknown='ignore')
oh.fit(df_train[['CNT_CHILDREN']])

df_train = pd.concat([df_train.drop('CNT_CHILDREN', axis=1).reset_index(drop=True), 
                       pd.DataFrame( oh.transform(df_train['CNT_CHILDREN'].values.reshape(-1,1)).toarray(), 
                                        columns=oh.get_feature_names(['CNT_CHILDREN']) 
                                    )  
                      ], 
                    axis=1).drop('index', axis=1)

df_test = pd.concat([df_test.drop('CNT_CHILDREN', axis=1).reset_index(drop=True), 
                       pd.DataFrame( oh.transform(df_test['CNT_CHILDREN'].values.reshape(-1,1)).toarray(), 
                                        columns=oh.get_feature_names(['CNT_CHILDREN']) 
                                    )  
                      ], 
                    axis=1).drop('index', axis=1)
show_plots("REGION_RATING_CLIENT_W_CITY", figsize=(15,3))
df_train['REGION_RATING_CLIENT_W_CITY'].value_counts(1)
df_train[['REGION_RATING_CLIENT_W_CITY', 'TARGET']].groupby('REGION_RATING_CLIENT_W_CITY')['TARGET'].mean()
df_train.head()
df_sample = df_train.sample(frac=0.1,replace=False, random_state=1)

X_f = df_sample[[col for col in df_sample.columns if col not in ['TARGET', 'SK_ID_CURR']]]
y_f = df_sample['TARGET']

X_f_train, X_f_test, y_f_train, y_f_test = train_test_split(X_f, y_f, test_size=0.3, random_state=0) 
sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l1',solver='liblinear'))
sel_.fit(X_f_train, y_f_train)
not_select_feat = X_f_train.columns[~(sel_.get_support())]
not_select_feat
df_train = df_train.drop('FLAG_WORK_PHONE', axis=1)
df_test = df_test.drop('FLAG_WORK_PHONE', axis=1)
AUC_values = list()

for feature in X_f.columns:
    clf = DecisionTreeClassifier()
    clf.fit(X_f_train[[feature]], y_f_train)
    pred = clf.predict_proba(X_f_test[[feature]])
    score = roc_auc_score(y_f_test,pred[:,1])
    AUC_values.append(score)
AUC_df = pd.Series(AUC_values)
AUC_df.index = X_f_train.columns
AUC_df = AUC_df.sort_values(ascending=False)
sns.distplot(AUC_values)
fig, ax = plt.subplots(figsize=(10,20))
sns.barplot(y=AUC_df.index, x=AUC_df)
informative_features = list(AUC_df[AUC_df>0.5].index)
rf_model = RandomForestClassifier()
rf_model.fit(X_f, y_f)

fv = dict(zip(X_f.columns,rf_model.feature_importances_))
fv_dict = {k: v for k, v in sorted(fv.items(), key=lambda item: item[1], reverse=True)}
fig, ax = plt.subplots(figsize=(10,20))
sns.barplot(y=list(fv_dict.keys()), x=list(fv_dict.values()))
important_features = [k for k,v in fv_dict.items() if v>0.01]
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
metrics = ['neg_log_loss', 'accuracy', 'f1', 'precision', 'roc_auc']
informative_and_important_features = list(set(informative_features).union(set(important_features)))
all_features = [col for col in df_train.columns if col not in ['SK_ID_CURR','TARGET']]
results = dict()

for set_name, feature_set in dict(zip(['', 'important_features', 'informative_features', 'informative_and_important_features'], 
                                      [all_features, important_features, informative_features, informative_and_important_features])).items():
    
    for model_name, model in dict(zip(['XGBClassifier', 'RandomForestClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier'], 
                               [XGBClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier])).items():

        validator = cross_validate(model(), df_train[feature_set], df_train['TARGET'], 
                                   cv=kfold, scoring=metrics, return_train_score=True,
                                   n_jobs=-1, verbose=10)
        results[f'{model_name}_{set_name}'] = validator
        print(f'{model_name}_{set_name}')
d = pd.DataFrame(results.values())
d.index = results.keys()
d = d.applymap(np.mean)
d = d.sort_values(by='test_roc_auc',ascending=False)
d['diff'] = (d['train_roc_auc'] - d['test_roc_auc']) *100
d

parameters = { 
                'n_estimators' : range(50, 150, 50),
                'max_depth' : range(4, 5, 1),
                'min_samples_split': range(200,400,200),
                "reg_alpha": [1.5, 2, 2.5],
                "reg_lambda": [3.5, 4, 4.5]
             }


grid_search = GridSearchCV(
            XGBClassifier(tree_method='gpu_hist'),
            parameters,
            scoring=['roc_auc', 'neg_log_loss'],
            refit='roc_auc',
            n_jobs=-1, 
            cv = kfold, 
            verbose=10,
            return_train_score = True 
) 

grid_search.fit(df_train[all_features], df_train['TARGET'])

print(grid_search.best_score_)
print(grid_search.best_params_)
train = grid_search.cv_results_['mean_train_roc_auc']
test = grid_search.cv_results_['mean_test_roc_auc']
params = grid_search.cv_results_['params']

sns.lineplot(y=test, x=list(range(0,len(test),1)))
sns.lineplot(y=train, x=list(range(0,len(train),1)))
for train, test,  param in zip(train, test, params):
    print("test: %f train: %f with: %r" % (test, train, param))
print("mean_train_roc_auc -----------------{}".format( np.mean(grid_search.cv_results_['mean_train_roc_auc']) ))
print("mean_test_roc_auc ------------------{}".format( np.mean(grid_search.cv_results_['mean_test_roc_auc']) ))

print("mean_train_neg_log_loss ------------{}".format( np.mean(grid_search.cv_results_['mean_train_neg_log_loss'])))
print("mean_test_neg_log_loss -------------{}".format( np.mean(grid_search.cv_results_['mean_test_neg_log_loss'])))
sns.distplot(grid_search.cv_results_['mean_test_roc_auc'])
sns.distplot(grid_search.cv_results_['mean_train_roc_auc'])
best_model = XGBClassifier(**grid_search.best_params_)

best_model.fit(df_train[all_features], df_train['TARGET'])

y_predict = best_model.predict_proba( df_test[all_features] )
df_results = pd.DataFrame({'true':df_test['TARGET'],
                            'predict': y_predict[:,1]})
roc_auc_score(df_results['true'], df_results['predict'])
sns.distplot( df_results.predict[df_results.true==1] )
sns.distplot( df_results.predict[df_results.true==0] )
print( confusion_matrix(df_results.true, 
                        np.where(df_results.predict>0.5, 1, 0)
                       ) 
     )
df_results.true.value_counts()
print( classification_report( df_results.true, 
                        np.where(df_results.predict>0.5, 1, 0)
                       ) 
     )
print( log_loss( df_results.true, df_results.predict) )
df_test['TARGET'].value_counts(1)
