import pandas as pd

import numpy as np

import seaborn as sns

import missingno as msno

import gc
import matplotlib.pyplot as plt

%matplotlib inline



from IPython.display import set_matplotlib_formats

set_matplotlib_formats('pdf', 'png')

pd.options.display.float_format = '{:.2f}'.format

rc={'savefig.dpi': 75, 'figure.autolayout': False, 'figure.figsize': [12, 8], 'axes.labelsize': 18,\

   'axes.titlesize': 18, 'font.size': 18, 'lines.linewidth': 2.0, 'lines.markersize': 8, 'legend.fontsize': 16,\

   'xtick.labelsize': 16, 'ytick.labelsize': 16}



sns.set(style='dark',rc=rc)
default_color = '#56B4E9'

colormap = plt.cm.cool
# Setting working directory



path = '../input/'
train = pd.read_csv(path + 'train.csv', index_col='id', na_values=-1)

test = pd.read_csv(path + 'test.csv', index_col='id', na_values=-1)
train.shape
test.shape
ax = sns.countplot('target',data=train,ax=ax,color=default_color)

sns.set(font_scale=1.5)

ax.set_xlabel(' ')

ax.set_ylabel(' ')

fig = plt.gcf()

fig.set_size_inches(10,5)

ax.set_ylim(top=700000)

for p in ax.patches:

    ax.annotate('{:.2f}%'.format(100*p.get_height()/len(train['target'])), (p.get_x()+ 0.3, p.get_height()+10000))
def get_meta(train):

    ##thanks to https://www.kaggle.com/bertcarremans/data-preparation-exploration

    data = []

    for col in train.columns:

        # Defining the role

        if col == 'target':

            role = 'target'

        elif col == 'id':

            role = 'id'

        else:

            role = 'input'



        # Defining the level

        if 'bin' in col or col == 'target':

            level = 'binary'

        elif 'cat' in col or col == 'id':

            level = 'nominal'

        elif train[col].dtype == np.float64:

            level = 'interval'

        elif train[col].dtype == np.int64:

            level = 'ordinal'



        # Initialize keep to True for all variables except for id

        keep = True

        if col == 'id':

            keep = False



        # Defining the data type 

        dtype = train[col].dtype



        # Creating a Dict that contains all the metadata for the variable

        col_dict = {

            'varname': col,

            'role'   : role,

            'level'  : level,

            'keep'   : keep,

            'dtype'  : dtype

        }

        data.append(col_dict)

    meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])

    meta.set_index('varname', inplace=True)

    return meta

        
meta_data = get_meta(train)

meta_data
meta_counts = meta_data.groupby(['role', 'level']).agg({'dtype': lambda x: x.count()}).reset_index()

meta_counts
fig,ax = plt.subplots()

fig.set_size_inches(20,5)

sns.barplot(data=meta_counts[meta_counts.role != 'target'],x="level",y="dtype",ax=ax,color=default_color)

ax.set(xlabel='Variable Type', ylabel='Count',title="Variables Count Across Datatype")
col_ordinal   = meta_data[(meta_data.level == 'ordinal') & (meta_data.keep)].index

col_nominal   = meta_data[(meta_data.level == 'nominal') & (meta_data.keep)].index

col_internval = meta_data[(meta_data.level == 'interval') & (meta_data.keep)].index

col_binary    = meta_data[(meta_data.level == 'binary') & (meta_data.keep) & (meta_data.role != 'target')].index
missingValueColumns = train.columns[train.isnull().any()].tolist()

df_null = train[missingValueColumns] 
msno.bar(df_null,figsize=(20,8),color=default_color,fontsize=18,labels=True)
msno.heatmap(df_null,figsize=(20,8),cmap=colormap)
msno.dendrogram(df_null,figsize=(20,8))
sorted_data = msno.nullity_sort(df_null, sort='descending') # or sort='ascending'

msno.matrix(sorted_data,figsize=(20,8),fontsize=14)
plt.figure(figsize=(18,16))

plt.title('Pearson correlation of continuous features', y=1.05, size=15)

sns.heatmap(train[col_internval].corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
train.fillna(-1, inplace=True)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=0)

rf.fit(train.drop(['target'],axis=1), train.target)

features = train.drop(['target'],axis=1).columns.values

print("----- Training Done -----")
def get_feature_importance_df(feature_importances, 

                              column_names, 

                              top_n=25):

    """Get feature importance data frame.

 

    Parameters

    ----------

    feature_importances : numpy ndarray

        Feature importances computed by an ensemble 

            model like random forest or boosting

    column_names : array-like

        Names of the columns in the same order as feature 

            importances

    top_n : integer

        Number of top features

 

    Returns

    -------

    df : a Pandas data frame

 

    """

     

    imp_dict = dict(zip(column_names, 

                        feature_importances))

    top_features = sorted(imp_dict, 

                          key=imp_dict.get, 

                          reverse=True)[0:top_n]

    top_importances = [imp_dict[feature] for feature 

                          in top_features]

    df = pd.DataFrame(data={'feature': top_features, 

                            'importance': top_importances})

    return df
feature_importance = get_feature_importance_df(rf.feature_importances_, features)
feature_importance
fig,ax = plt.subplots()

fig.set_size_inches(20,10)

sns.barplot(data=feature_importance[:10],x="feature",y="importance",ax=ax,color=default_color,)

ax.set(xlabel='Variable name', ylabel='Importance',title="Variable importances")
from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
def cross_val_model(X,y, model, n_splits=3):

    from sklearn.model_selection import StratifiedKFold

    from sklearn.model_selection import cross_val_score

    X = np.array(X)

    y = np.array(y)





    folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2017).split(X, y))



    for j, (train_idx, test_idx) in enumerate(folds):

        X_train = X[train_idx]

        y_train = y[train_idx]

        X_holdout = X[test_idx]

        y_holdout = y[test_idx]



        print ("Fit %s fold %d" % (str(model).split('(')[0], j+1))

        model.fit(X_train, y_train)

        cross_score = cross_val_score(model, X_holdout, y_holdout, cv=3, scoring='roc_auc')

        print("    cross_score: %.5f" % cross_score.mean())       
train.fillna(-1, inplace=True)

test.fillna(-1, inplace=True)
#RandomForest params

rf_params = {}

rf_params['n_estimators'] = 200

rf_params['max_depth'] = 6

rf_params['min_samples_split'] = 70

rf_params['min_samples_leaf'] = 30

rf_params['n_jobs '] = -1
rf_model = RandomForestClassifier(**rf_params)
X = train.drop('target',axis=1)

y = train['target']
cross_val_model(X, y, rf_model)
# XGBoost params

xgb_params = {}

xgb_params['learning_rate'] = 0.02

xgb_params['n_estimators'] = 1000

xgb_params['max_depth'] = 4

xgb_params['subsample'] = 0.9

xgb_params['colsample_bytree'] = 0.9  

xgb_params['n_jobs'] = -1
XGB_model = XGBClassifier(**rf_params)
X = train.drop('target',axis=1)

y = train['target']
cross_val_model(X, y, XGB_model)
one_hot = {c: list(train[c].unique()) for c in train.columns if c not in ['id','target']}
def OHE_by_unique(train, one_hot, limit):

    

    #ONE-HOT enconde features with more than 2 and less than 'limit' unique values

    df = train.copy()

    for c in one_hot:

        if len(one_hot[c])>2 and len(one_hot[c]) < limit:

            for val in one_hot[c]:

                df[c+'_oh_' + str(val)] = (df[c].values == val).astype(np.int)

    return df
train_ohe = OHE_by_unique(train, one_hot, 7)
train.shape
train_ohe.shape
X_ohe = train_ohe.drop('target',axis=1)

y_ohe = train_ohe['target']
cross_val_model(X_ohe, y_ohe, rf_model)
cross_val_model(X_ohe, y_ohe, XGB_model)
train['null_sum'] = train[train==-1].count(axis=1)

test['null_sum']  = test[test==-1].count(axis=1)
train['bin_sum']  = train[col_binary].sum(axis=1)

test['bin_sum']  =  test[col_binary].sum(axis=1)
train['ord_sum']  = train[col_ordinal].sum(axis=1)

test['ord_sum']  =  test[col_ordinal].sum(axis=1)
X_ext = train.drop('target',axis=1)

y_ext = train['target']
cross_val_model(X_ext, y_ext, rf_model)
cross_val_model(X_ext, y_ext, XGB_model)