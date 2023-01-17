import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np
!ls ../input/bod-results/
results = pd.read_csv('../input/bod-results/BOD Kaggle_results.csv',usecols=['Methodology','RMSE','Change log'])
pd.set_option('display.max_colwidth', -1)
results['Methodology'] = results['Methodology'].str.replace('\n','')
results
df = pd.read_csv('../input/prediction-bod-in-river-water/train.csv')
df.head()
df.describe()
corr =df[df.columns.to_list()[1:]].corr()
corr
plt.figure(figsize=(10,8))

sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns)
len(df)
df.drop(columns=['Id'],inplace=True)
sns.set()

plt.figure(figsize=(14,10))

sns.distplot(df['target'])

plt.title('Distribution of data points in the dataset');
df.isna().sum()
df.columns.to_list()[:3]
df = df[df.columns.to_list()[:3]]
df.isna().sum()
nan_cols = df.isna().sum()[df.isna().sum()>0].index.to_list()
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df[nan_cols] = imputer.fit_transform(df[nan_cols])
df
df.max()
df.min()
corr = df[df.columns.to_list()].corr()
corr
df.columns.to_list()[1:3]
df['combined'] = df[df.columns.to_list()[1:]].mean(axis=1)
df.head()
corr = df[df.columns.to_list()].corr()

display(corr)

plt.figure(figsize=(10,8))

sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns)
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error,make_scorer
from sklearn.model_selection import cross_val_score, train_test_split
from skopt import forest_minimize
def rmse(y,y_pred):

    return np.sqrt(mean_squared_error(y_true=y,y_pred=y_pred))
scorer = make_scorer(rmse,greater_is_better=False)
feature_columns = df.columns.to_list()

target_column = feature_columns.pop(0)

(feature_columns,target_column)
y = df[target_column].values
X = df[feature_columns].values
def optimize_gbd(space):

    alpha,learning_rate, max_depth,max_features,max_leaf_nodes,n_estimators= space

    gbd = GradientBoostingRegressor(alpha=alpha,learning_rate=learning_rate, max_depth=max_depth,

                                    max_features=max_features,max_leaf_nodes=max_leaf_nodes,

                                    n_estimators=n_estimators,random_state=5, criterion='mse')

    score =  -1*cross_val_score(gbd,X,y,cv=5,scoring=scorer).mean()

    print('Error : {}'.format(score))

    return score
def train_best_gbd(space):

    alpha,learning_rate, max_depth,max_features,max_leaf_nodes,n_estimators= space

    max_depth,max_features,max_leaf_nodes,n_estimators = int(max_depth), int(max_features), int(max_leaf_nodes), int(n_estimators)

    gbd = GradientBoostingRegressor(alpha=alpha,learning_rate=learning_rate, max_depth=max_depth,

                                    max_features=max_features,max_leaf_nodes=max_leaf_nodes,

                                    n_estimators=n_estimators,random_state=5,

                                   criterion='mse')

    gbd.fit(X=X,y=y)

    error = np.sqrt(mean_squared_error(y_pred=gbd.predict(X),y_true=y))

    print('Overall error : {}'.format(error))

    return gbd
space = [(0.1,0.9),#alpha

         (1e-3,0.8),#learning_rate

         (2,20),#max_depth

         (1,3),#max_features

         (2,100),#max_leaf_nodes

         (100,1000)#n_estimators

    

]
best_params = forest_minimize(optimize_gbd,dimensions=space,n_calls=200,n_jobs=6,random_state=5)
def to_df(best_params,cols=['alpha','learning_rate','max_depth',

                            'max_features','max_leaf_nodes','n_estimators']):

    params =  np.array(best_params['x_iters'])

    df = pd.DataFrame(columns=cols,data=params)

    df['scores'] = best_params['func_vals']

    return df

    
df_scores = to_df(best_params)
df_scores.head()
df_scores['scores'].median()
df_scores['scores'].quantile(0.25)
by_percentile = df_scores[(df_scores['scores']>df_scores['scores'].quantile(0.45)) & (df_scores['scores']<df_scores['scores'].quantile(0.55))]
by_percentile
np.random.seed(14)

choice = np.random.choice(len(by_percentile))
params = by_percentile.iloc[choice][['alpha','learning_rate','max_depth',

                            'max_features','max_leaf_nodes','n_estimators']].values
params
gbd_trained = train_best_gbd(params)
ensemble_df = pd.DataFrame()

ensemble_df['gbd'] = gbd_trained.predict(X)
ensemble_df.head()
sns.set()

plt.figure(figsize=(14,10))

plt.scatter(range(len(df)),df['target'],color='black')

plt.plot(range(len(df)),ensemble_df['gbd'],color='blue')

plt.legend(['Values predicted with GBT','Real values'])

plt.xlabel('Timestamps')

plt.ylabel('Target values')

plt.title('Results of GBT model');
from sklearn.ensemble import ExtraTreesRegressor
def optimize_extratree(space):

    n_estimators,max_depth,min_samples_split,min_samples_leaf,max_features = space

    extra_tree = ExtraTreesRegressor(min_samples_split=min_samples_split,max_features=max_features,random_state=5,

                            min_samples_leaf=min_samples_leaf,max_depth=max_depth,n_estimators=n_estimators)

    score =  -1*cross_val_score(extra_tree,X,y,cv=5,scoring=scorer).mean()

    print('Error : {}'.format(score))

    return score
def train_best_extratree(space):

    n_estimators,max_depth,min_samples_split,min_samples_leaf,max_features = list(map(int,space))

    extra_tree = ExtraTreesRegressor(min_samples_split=min_samples_split,max_features=max_features,random_state=5,

                            min_samples_leaf=min_samples_leaf,max_depth=max_depth,n_estimators=n_estimators)

    extra_tree.fit(X,y)

    error = np.sqrt(mean_squared_error(y_true=extra_tree.predict(X),y_pred=y))

    print('Overall error : {}'.format(error))

    return extra_tree
space = [(100,1000),#n_estimators

        (8,20),#max_depth

        (2,4),#min_samples_split

         (2,5),#min_samples_leaf

        (1,3)#max_features

        ]
best_params= forest_minimize(optimize_extratree,random_state=5,dimensions=space,n_calls=30)
df_scores = to_df(best_params,cols=["n_estimators","max_depth","min_samples_split",

                                    "min_samples_leaf",

                                    "max_features"])
df_scores.head()
df_scores['scores'].mean()
params = df_scores.iloc[2][["n_estimators","max_depth","min_samples_split",

                                    "min_samples_leaf",

                                    "max_features"]].values
ex_tree_reg = train_best_extratree(params)
ensemble_df['extra_tree'] = ex_tree_reg.predict(X)
sns.set()

plt.figure(figsize=(14,10))

plt.scatter(range(len(df)),df['target'],color='black')

plt.plot(range(len(df)),ensemble_df['extra_tree'],color='green')

plt.legend(['Values predicted with ETR','Real values'])

plt.xlabel('Timestamps')

plt.ylabel('Target values')

plt.title('Results of ETR model');
from sklearn.linear_model import LinearRegression
ensemble_df['y'] = y
ensemble_df.head()
ensemble_df.corr()
first_lvl_features = ensemble_df[['gbd','extra_tree']].values
labels = ensemble_df['y'].values
lr_second_lvl = LinearRegression()
-1*cross_val_score(lr_second_lvl,first_lvl_features,labels,scoring=scorer).mean()
lr_second_lvl.fit(first_lvl_features,labels)
np.sqrt(mean_squared_error(y_pred=lr_second_lvl.predict(first_lvl_features),y_true=labels))
sns.set()

plt.figure(figsize=(14,10))

plt.scatter(range(len(df)),df['target'],color='black')

plt.plot(range(len(df)),lr_second_lvl.predict(first_lvl_features),color='red')

plt.legend(['Final predictions with second level model','Real values'])

plt.xlabel('Timestamps')

plt.ylabel('Target values')

plt.title('Results of predictions using second level model');
sns.set()

plt.figure(figsize=(14,10))

plt.scatter(range(len(df)),df['target'],color='black')

plt.plot(range(len(df)),ensemble_df['extra_tree'],color='green')

plt.plot(range(len(df)),ensemble_df['gbd'],color='blue')

plt.plot(range(len(df)),lr_second_lvl.predict(first_lvl_features),color='red')

plt.legend(['Values predicted with ETR','Values predicted with GBT','Final predictions with second level model','Real values'])

plt.xlabel('Timestamps')

plt.ylabel('Target values')

plt.title('Results of predictions using second level model');
def detect_outliers(df,constant=2):

    outliers = df[(df['target']<=df['target'].mean()-constant*df['target'].std()) | (df['target']>=df['target'].mean()+constant*df['target'].std())]

    return outliers
outliers = detect_outliers(df,1)
sns.set()

from matplotlib.lines import Line2D

legend_elements = [Line2D([0], [0], color='red', lw=4, label='Final predictions with second level model'),

                   Line2D([0], [0], marker='o', label='Outliers in real values',

                          markerfacecolor='orange',color='#999999', markersize=15),

                   Line2D([0], [0], marker='o',color='#999999', label='Real values',

                          markerfacecolor='black', markersize=15),

                   ]

plt.figure(figsize=(14,10))

for c,i in enumerate(df['target'].values):

    if i in outliers['target'].values:

        plt.scatter(c,i,color='orange',label='Outliers in real values')

    else:

        plt.scatter(c,i,color='black',label='Real values')

plt.plot(range(len(df)),lr_second_lvl.predict(first_lvl_features),color='red')

plt.legend(handles=legend_elements)

plt.xlabel('Timestamps')

plt.ylabel('Target values')

plt.title('Results of predictions using second level model');
sns.set()

plt.figure(figsize=(14,10))

plt.scatter(range(len(df))[-24:],df['target'][-24:],color='black')

plt.plot(range(len(df))[-24:],lr_second_lvl.predict(first_lvl_features)[-24:],color='red')

plt.legend(['Final predictions with second level model','Real values'])

plt.xlabel('Monthes')

plt.ylabel('Target values')

plt.title('Results of predictions using second level model (showing data only for last two years)')

plt.yticks(np.arange(min(df['target'][-24:]),max(df['target'][-24:])+0.5,0.5))

plt.xticks(range(len(df))[-24:],range(1,25));
sub_df = pd.read_csv("../input/prediction-bod-in-river-water/test.csv",usecols=['Id','1','2'])
sub_df.head()
sub_df.isna().sum()
sub_df.head()
sub_df['combined'] = sub_df[sub_df.columns.to_list()[1:]].mean(axis=1)
sub_df.head()
feature_columns = sub_df.columns.to_list()[1:]
ensemble_df_test = pd.DataFrame()

ensemble_df_test['gbd'] = gbd_trained.predict(sub_df[feature_columns])

ensemble_df_test['extra_tree'] = ex_tree_reg.predict(sub_df[feature_columns])
ensemble_df_test.head()
sub_df['Predicted'] = lr_second_lvl.predict(ensemble_df_test.values)
sub_df = sub_df[['Id','Predicted']]
sub_df.to_csv('submission.csv',index=False)
sub_df['Predicted'].values