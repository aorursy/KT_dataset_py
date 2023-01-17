#from plotly.offline import init_notebook_mode, iplot
#import plotly.graph_objs as go
#import plotly.plotly as py
#from plotly import tools
#from datetime import date
import pandas as pd
import numpy as np 
%matplotlib inline
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import seaborn as sns
#import random 
#import warnings
#import operator
#warnings.filterwarnings("ignore")
#init_notebook_mode(connected=True)

test = pd.read_csv("../input/test.csv")
poverty_train = pd.read_csv("../input/train.csv")
display(poverty_train.head(), poverty_train.shape)
display(test.head(), test.shape, test[["Id", "idhogar", "parentesco1"]].head(10))
print ("Top Columns having missing values in the train set.")
missmap = poverty_train.isnull().sum().to_frame().sort_values(0, ascending = False) / len(poverty_train)
missmap.head()
print ("Top Columns having missing values in the test set.")
missmap = test.isnull().sum().to_frame().sort_values(0, ascending = False) / len(test)
missmap.head()
poverty_train[poverty_train.meaneduc.isnull()]
value1 = poverty_train["meaneduc"].mean()
value2 = value1*value1
display(value1, value2)

poverty_train["meaneduc"].fillna(value1, inplace = True)
poverty_train["SQBmeaned"].fillna(value2, inplace = True)

print ("Top Columns having missing values")
missmap = poverty_train.isnull().sum().to_frame().sort_values(0, ascending = False) / len(poverty_train)
missmap.head()
poverty_train.describe()
display(poverty_train.loc[( poverty_train["tipovivi1" ] == 1, "v2a1")].isna().sum(), 

        len(poverty_train.loc[( poverty_train["tipovivi1" ] == 1, "v2a1")]))
poverty_train.loc[(poverty_train["tipovivi1" ] == 1, "v2a1")] = 0

print ("Top Columns having missing values")
missmap = poverty_train.isnull().sum().to_frame().sort_values(0, ascending = False) / len(poverty_train)
missmap.head()
display(poverty_train.loc[( poverty_train["tipovivi2" ] == 1, "v2a1")].isna().sum(), 
        len(poverty_train.loc[( poverty_train["tipovivi2" ] == 1, "v2a1")]))
display(poverty_train.loc[( poverty_train["tipovivi3" ] == 1, "v2a1")].isna().sum(), 
        len(poverty_train.loc[( poverty_train["tipovivi3" ] == 1, "v2a1")]))
display(poverty_train.loc[( poverty_train["tipovivi4" ] == 1, "v2a1")].isna().sum(), 
        len(poverty_train.loc[( poverty_train["tipovivi4" ] == 1, "v2a1")]))
display(poverty_train.loc[( poverty_train["tipovivi5" ] == 1, "v2a1")].isna().sum(), 
        len(poverty_train.loc[( poverty_train["tipovivi5" ] == 1, "v2a1")]))
poverty_train["v2a1"].fillna(0, inplace = True)

print ("Top Columns having missing values (%)")
missmap = poverty_train.isnull().sum().to_frame().sort_values(0, ascending = False) / len(poverty_train)
missmap.head()
poverty_train["v18q1"].describe()
poverty_train["v18q"].describe()
display(poverty_train.loc[( poverty_train["v18q" ] == 0, "v18q1")].isna().sum(),
        len(poverty_train[poverty_train["v18q"] == 0]))
poverty_train.loc[(poverty_train["v18q" ] == 0, "v18q1")] = 0

print ("Top Columns having missing values (%)")
missmap = poverty_train.isnull().sum().to_frame().sort_values(0, ascending = False) / len(poverty_train)
missmap.head()
poverty_train["rez_esc"].describe()
poverty_train["rez_esc"].median()
poverty_train['rez_esc'] = poverty_train.apply(
    lambda row: 0 if (row['instlevel1'] == 1) else row['rez_esc'],
    axis=1
)

poverty_train['rez_esc'] = poverty_train.apply(
    lambda row: poverty_train["rez_esc"].median() if np.isnan(row["rez_esc"]) else row['rez_esc'],
    axis=1
)

print ("Top Columns having missing values (%)")
missmap = poverty_train.isnull().sum().to_frame().sort_values(0, ascending = False) / len(poverty_train)
missmap.head()

def plot_value_counts(series, title=None):
    '''
    Plot distribution of values counts in a pd.Series
    '''
    _ = plt.figure(figsize=(12,6))
    z = series.value_counts()
    sns.barplot(x=z, y=z.index)
    _ = plt.title(title)
    
plot_value_counts(poverty_train['edjefe'], 'Value counts of edjefe')
plot_value_counts(poverty_train['edjefa'], 'Value counts of edjefa')
plot_value_counts(poverty_train['dependency'], 'Value counts of dependency')
poverty_train["dependency"].replace('yes', 1, inplace = True)
poverty_train["dependency"].replace('no', 0, inplace = True)
poverty_train["edjefa"].replace('yes', 1, inplace = True)
poverty_train["edjefa"].replace('no', 0, inplace = True)
poverty_train["edjefe"].replace('yes', 1, inplace = True)
poverty_train["edjefe"].replace('no', 0, inplace = True)

#check if our solution worked
display(poverty_train["dependency"].value_counts(),
        poverty_train["edjefa"].value_counts(),
        poverty_train["edjefe"].value_counts())
poverty_train["edjefe"] = poverty_train["edjefe"].astype(float)
poverty_train["edjefa"] = poverty_train["edjefa"].astype(float)
poverty_train["dependency"] = poverty_train["dependency"].astype(float)
d={}
weird=[]
for row in poverty_train.iterrows():
    idhogar=row[1]['idhogar']
    target=row[1]['Target']
    if idhogar in d:
        if d[idhogar]!=target:
            weird.append(idhogar)
    else:
        d[idhogar]=target

len(set(weird))
for i in set(weird):
    hhold=poverty_train[poverty_train['idhogar']==i][['idhogar', 'parentesco1', 'Target']]
    target=hhold[hhold['parentesco1']==1]['Target'].tolist()[0]
    for row in hhold.iterrows():
        idx=row[0]
        if row[1]['parentesco1']!=1:
            poverty_train.at[idx, 'Target']=target

poverty_train[poverty_train['idhogar']==weird[1]][['idhogar','parentesco1', 'Target']]
#poverty_train.hist(bins=30, figsize=(20,15))
#plt.title( "Histogram Plots")
#plt.show()
#poverty_train[["meaneduc", "v2a1", "age", "agesq", "SQBmeaned",
              #"SQBovercrowding", "overcrowding", "SQBage","escolari", "SQBescolari",
               #"SQBdependency", "SQBhogar_total" ]].hist(bins=30, figsize=(20,15))
#plt.title( "Histogram Plots")
#plt.show()
poverty_train[["v18q1", "v2a1", "rez_esc"]].describe()
very_poor = poverty_train[poverty_train["Target"] == 1]
poor = poverty_train[poverty_train["Target"] == 2]
vulnerable = poverty_train[poverty_train["Target"] == 3]
safe = poverty_train[poverty_train["Target"] == 4]
display(very_poor.shape,
       poor.shape,
       vulnerable.shape,
       safe.shape)
display(
    very_poor[["v18q1", "v2a1", "rez_esc"]].describe(),
    poor[["v18q1", "v2a1", "rez_esc"]].describe(),
    vulnerable[["v18q1", "v2a1", "rez_esc"]].describe(),
    safe[["v18q1", "v2a1", "rez_esc"]].describe())
very_poor[["Target", "rez_esc", "meaneduc", "v2a1", "age","overcrowding",
           "rooms", "v18q1"]].hist(bins=25, figsize=(20,15))
plt.title( "Histogram Plots")
plt.show()
safe[["Target", "rez_esc", "meaneduc", "v2a1", "age","overcrowding",
           "rooms", "v18q1"]].hist(bins=25, figsize=(20,15))
plt.title( "Histogram Plots")
plt.show()
plt.figure(figsize=(12,7))
plt.title("Distributions of Age")
sns.kdeplot(very_poor["age"], label = "Very Poor Group", shade = True)
sns.kdeplot(poor["age"], label = "Poor Group", shade = True)
sns.kdeplot(vulnerable["age"], label = "Vulnerable Group", shade = True)
sns.kdeplot(safe["age"], label="Safe Group", shade = True)
plt.legend();
import numpy as np
import scipy as sp
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h
display(mean_confidence_interval(very_poor["age"], confidence = 0.95),
mean_confidence_interval(safe["age"], confidence = 0.95))
def do_features(df):
    feats_div = [('children_fraction', 'r4t1', 'r4t3'), 
                 ('working_man_fraction', 'r4h2', 'r4t3'),
                 ('all_man_fraction', 'r4h3', 'r4t3'),
                 ('human_density', 'tamviv', 'rooms'),
                 ('human_bed_density', 'tamviv', 'bedrooms'),
                 ('rent_per_person', 'v2a1', 'r4t3'),
                 ('rent_per_room', 'v2a1', 'rooms'),
                 ('mobile_density', 'qmobilephone', 'r4t3'),
                 ('tablet_density', 'v18q1', 'r4t3'),
                 ('mobile_adult_density', 'qmobilephone', 'r4t2'),
                 ('tablet_adult_density', 'v18q1', 'r4t2'),
                 #('', '', ''),
                ]
    
    feats_sub = [('people_not_living', 'tamhog', 'tamviv'),
                 ('people_weird_stat', 'tamhog', 'r4t3')]

    for f_new, f1, f2 in feats_div:
        df['fe_' + f_new] = (df[f1] / df[f2]).astype(np.float32)       
    for f_new, f1, f2 in feats_sub:
        df['fe_' + f_new] = (df[f1] - df[f2]).astype(np.float32)
    
    # aggregation rules over household
    aggs_num = {'age': ['min', 'max', 'mean'],
                'escolari': ['min', 'max', 'mean']
               }
    aggs_cat = {'dis': ['mean']}
    for s_ in ['estadocivil', 'parentesco', 'instlevel']:
        for f_ in [f_ for f_ in df.columns if f_.startswith(s_)]:
            aggs_cat[f_] = ['mean', 'count']
    # aggregation over household
    for name_, df_ in [('18', df.query('age >= 18'))]:
        df_agg = df_.groupby('idhogar').agg({**aggs_num, **aggs_cat}).astype(np.float32)
        df_agg.columns = pd.Index(['agg' + name_ + '_' + e[0] + "_" + e[1].upper() for e in df_agg.columns.tolist()])
        df = df.join(df_agg, how='left', on='idhogar')
        del df_agg
    # do something advanced above...
    
    # Drop SQB variables, as they are just squres of other vars 
    #df.drop([f_ for f_ in df.columns if f_.startswith('SQB') or f_ == 'agesq'], axis=1, inplace=True)
    # Drop id's
    df.drop(['Id'], axis=1, inplace=True)
    # Drop repeated columns
    df.drop(['hhsize', 'female', 'area2'], axis=1, inplace=True)
    return df

def convert_OHE2LE(df):
    print_check = True
    
    tmp_df = df.copy(deep=True)
    for s_ in ['pared', 'piso', 'techo', 'abastagua', 'sanitario', 'energcocinar', 'elimbasu', 
               'epared', 'etecho', 'eviv', 'estadocivil', 'parentesco', 
               'instlevel', 'lugar', 'tipovivi',
               'manual_elec']:
        if 'manual_' not in s_:
            cols_s_ = [f_ for f_ in df.columns if f_.startswith(s_)]
        elif 'elec' in s_:
            cols_s_ = ['public', 'planpri', 'noelec', 'coopele']
        if print_check:
            sum_ohe = df[cols_s_].sum(axis=1).unique()
            if sum_ohe.shape[0]>1:
                print(s_)
                print(df[cols_s_].sum(axis=1).value_counts())
                #print(df[list(cols_s_+['Id'])].loc[df[cols_s_].sum(axis=1) == 0])
        tmp_cat = df[cols_s_].idxmax(axis=1)
        tmp_df[s_ + '_LE'] = LabelEncoder().fit_transform(tmp_cat).astype(np.int16)
        if 'parentesco1' in cols_s_:
            cols_s_.remove('parentesco1')
        tmp_df.drop(cols_s_, axis=1, inplace=True)
    return tmp_df
from sklearn.preprocessing import LabelEncoder

def encode_data(df):
    '''
    The function does not return, but transforms the input pd.DataFrame
    
    Encodes the Costa Rican Household Poverty Level data 
    following studies in https://www.kaggle.com/mlisovyi/categorical-variables-in-the-data
    and the insight from https://www.kaggle.com/c/costa-rican-household-poverty-prediction/discussion/61403#359631
    
    The following columns get transformed: edjefe, edjefa, dependency, idhogar
    The user most likely will simply drop idhogar completely (after calculating houshold-level aggregates)
    '''
    
    yes_no_map = {'no': 0, 'yes': 1}
    
    df['dependency'] = df['dependency'].replace(yes_no_map).astype(np.float32)
    
    df['edjefe'] = df['edjefe'].replace(yes_no_map).astype(np.float32)
    df['edjefa'] = df['edjefa'].replace(yes_no_map).astype(np.float32)
    
    df['idhogar'] = LabelEncoder().fit_transform(df['idhogar'])
def process_df(df_):
    # fix categorical features
    encode_data(df_)
    #fill in missing values based on https://www.kaggle.com/mlisovyi/missing-values-in-the-data
    for f_ in ['v2a1', 'v18q1', 'meaneduc', 'SQBmeaned']:
        df_[f_] = df_[f_].fillna(0)
    df_['rez_esc'] = df_['rez_esc'].fillna(0)
    # do feature engineering and drop useless columns
    return do_features(df_)

#train = process_df(train)
test = process_df(test)

def train_test_apply_func(train_, test_, func_):
    test_['Target'] = 0
    xx = pd.concat([train_, test_])

    xx_func = func_(xx)
    train_ = xx_func.iloc[:train_.shape[0], :]
    test_  = xx_func.iloc[train_.shape[0]:, :].drop('Target', axis=1)

    del xx, xx_func
    return train_, test_
train = do_features(poverty_train)
#train, test = train_test_apply_func(train, test, convert_OHE2LE)
X = train.query('parentesco1==1')
#X = train

# pull out the target variable
y = X['Target'] - 1
X = X.drop(['Target'], axis=1)
cols_2_drop = ['agg18_estadocivil1_MEAN', 'agg18_estadocivil3_COUNT', 'agg18_estadocivil4_COUNT', 
               'agg18_estadocivil5_COUNT', 'agg18_estadocivil6_COUNT', 'agg18_estadocivil7_COUNT', 
               'agg18_instlevel1_COUNT', 'agg18_instlevel2_COUNT', 'agg18_instlevel3_COUNT', 
               'agg18_instlevel4_COUNT', 'agg18_instlevel5_COUNT', 'agg18_instlevel6_COUNT', 
               'agg18_instlevel7_COUNT', 'agg18_instlevel8_COUNT', 'agg18_instlevel9_COUNT', 
               'agg18_parentesco10_COUNT', 'agg18_parentesco10_MEAN', 'agg18_parentesco11_COUNT', 
               'agg18_parentesco11_MEAN', 'agg18_parentesco12_COUNT', 'agg18_parentesco12_MEAN', 
               'agg18_parentesco1_COUNT', 'agg18_parentesco2_COUNT', 'agg18_parentesco3_COUNT', 
               'agg18_parentesco4_COUNT', 'agg18_parentesco4_MEAN', 'agg18_parentesco5_COUNT', 
               'agg18_parentesco6_COUNT', 'agg18_parentesco6_MEAN', 'agg18_parentesco7_COUNT', 
               'agg18_parentesco7_MEAN', 'agg18_parentesco8_COUNT', 'agg18_parentesco8_MEAN', 
               'agg18_parentesco9_COUNT', 'fe_people_weird_stat', 'hacapo', 'hacdor', 'mobilephone',
               'parentesco1', 'rez_esc', 'v14a', 'v18q', # here
                'agg18_age_MIN', 'agg18_age_MAX' , 'agg18_age_MEAN', 'agg18_escolari_MIN',
                'agg18_escolari_MAX', 'agg18_escolari_MEAN', 'agg18_dis_MEAN', 
                'agg18_estadocivil1_COUNT', 'agg18_estadocivil2_MEAN', 'agg18_estadocivil2_COUNT',
                'agg18_estadocivil3_MEAN', 'agg18_estadocivil4_MEAN', 'agg18_estadocivil5_MEAN',
                'agg18_estadocivil6_MEAN','agg18_estadocivil7_MEAN','agg18_parentesco1_MEAN',
                'agg18_parentesco2_MEAN','agg18_parentesco3_MEAN','agg18_parentesco5_MEAN','agg18_parentesco9_MEAN',
                'agg18_instlevel1_MEAN','agg18_instlevel2_MEAN','agg18_instlevel3_MEAN','agg18_instlevel4_MEAN',
                'agg18_instlevel5_MEAN','agg18_instlevel6_MEAN','agg18_instlevel7_MEAN','agg18_instlevel8_MEAN',
                'agg18_instlevel9_MEAN']

X.drop((cols_2_drop+['idhogar']), axis=1, inplace=True)
test.drop((cols_2_drop+['idhogar']), axis=1, inplace=True)
#use the following to test model generalization
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=314, stratify=y)
#display(X_train.shape,test.shape)
#num_cols = poverty_train.columns[poverty_train.dtypes.apply(lambda c: np.issubdtype(c, np.number))]
#num_cols = train.select_dtypes(include=[np.float32, np.int])
#num_cols.head()

#cols_to_norm = ['v2a1','meaneduc', "overcrowding", "SQBovercrowding", "SBQdependency", 
                #"SQBmeaned", "age", "SQBescolari", "escolari",
                #"SQBage", "SQBhogar_total", "SQBedjefe", "SQBhogar_nin",
                #"agesq"]
#train[cols_to_norm] = survey_data[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#train[num_cols] = scaler.fit_transform(train[num_cols])
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()

#X_train = X
#y_train = train["Target"].copy()
#X_train_scaled = scaler.fit_transform(X_train)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from scipy.stats import expon, reciprocal

param_distribs = { 
        'penalty': ['l1', 'l2'],
        'C': [0.001, 0.01, 0.1, 0.25, 0.50, 0.75, 1, 5,
              10, 500, 750, 1000, 1250, 1500, 2000, 10000],
        #'multi_class': ['ovr', 'multinomial'],
        'fit_intercept': [True, False],
        'class_weight': [None, 'balanced']
        
    }

logit = LogisticRegression(random_state=42, solver = "liblinear", multi_class = 'ovr',
                             n_jobs = 4, max_iter = 200)
rnd_search = GridSearchCV(logit, param_grid=param_distribs,
                                cv=5, scoring='f1_macro',
                                verbose=2, n_jobs=4)
rnd_search.fit(X, y)
cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)
print(rnd_search.best_score_, rnd_search.best_estimator_, rnd_search.best_params_)
#to test model generalization
#from sklearn.metrics import f1_score
#y_pred = rnd_search.predict(X_test)
#f1_score(y_test, y_pred, average='macro')
#fit the entire model with selected parameters.

y_subm = pd.read_csv('../input/sample_submission.csv')
y_subm['Target'] = rnd_search.predict(test) + 1
y_subm.to_csv('submission.csv', index=False)
#from sklearn.model_selection import RandomizedSearchCV
#from scipy.stats import expon, reciprocal
#from sklearn.svm import SVC

# see https://docs.scipy.org/doc/scipy/reference/stats.html
# for `expon()` and `reciprocal()` documentation and more probability distribution functions.

# Note: gamma is ignored when kernel is "linear"
#param_distribs = {
        #'kernel': ['linear', 'rbf', 'poly'],
        #'C': reciprocal(20, 200000),
        #'gamma': expon(scale=1.0),
#}

#svm = SVC()
#rnd_search = RandomizedSearchCV(svm, param_distributions=param_distribs,
                                #n_iter=50, cv=5, scoring='neg_mean_squared_error',
                                #averbose=2, n_jobs=4, random_state=42)
#rnd_search.fit(X_train_scaled, y_train)
#cvres = rnd_search.cv_results_
#for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    #print(mean_score, params)