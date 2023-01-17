import numpy as np 

import pandas as pd 

import pandas_profiling

from matplotlib import pyplot as plt

import seaborn as sns

import re

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier

import xgboost as xgb

from sklearn.metrics import roc_auc_score

import lightgbm as lgb

from skopt import BayesSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn import decomposition

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

from sklearn import metrics

%matplotlib inline

pd.set_option('display.max_columns',150)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#reading data

data= pd.read_csv('/kaggle/input/angara/datafile.csv')
# getting column list

data.columns.tolist()
data.info()
#preparing pandas profile report for eda and particularly univariate analysis

# profile = pandas_profiling.ProfileReport(data, title='Pandas Profiling Report', html={'style':{'full_width':True}})

# profile.to_file(output_file="angara_eda.html")
# profile
inspection_cols =['ActiveSinceDays','avgdays_btw_loads_l6','Contact_ind','Customer_status','dayssincelastload','daystofirstload']
data[inspection_cols]
# number of unique values in each column

data[inspection_cols].nunique()
data.Customer_status.unique()
#columns with missing values as nan.

data.columns[data.isna().sum()>0]
missing_val_columns  = ['r_avg_max_rtlX_m0', 'r_avg_max_rtlX_m1', 'r_avg_max_rtlX_m2','r_avg_max_rechX_m0', 'r_avg_max_rechX_m1', 'r_avg_max_rechX_m2']
data[data.isna()]=-999
data.columns[data.isna().sum()>0]
drop_Columns = list(data.nunique()[data.nunique()==1].index)
bill_amount_cols = [i for i in data.columns if i.startswith('billX_amt')]

bill_count_cols = [i for i in data.columns if i.startswith('billX_cnt')]

bill_derived_cols = [i for i in data.columns if (i.startswith('r_bill') or i.startswith('tot_bill'))]

data[bill_amount_cols+bill_count_cols+bill_derived_cols]
sum(data.tot_billX_amt_m012==0)
data[data.tot_billX_amt_m012>0][bill_amount_cols+bill_count_cols+bill_derived_cols].describe()
sum(data[bill_count_cols].apply(lambda x:  sum(x),axis=1)==0)
data[data.tot_billX_amt_m012>0][bill_amount_cols].plot(kind='line', figsize=(40,8))
data[data.tot_billX_amt_m012>0][bill_amount_cols+bill_count_cols+bill_derived_cols].quantile(np.arange(0.95,1,0.01))
sns.pairplot(data[bill_amount_cols+bill_count_cols+['targetid']],hue='targetid',size=5)
grid_kws = {"height_ratios": (.9, .03), "hspace": .3}

f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws,figsize=(12,12))

ax = sns.heatmap(data[bill_amount_cols+bill_count_cols+bill_derived_cols].corr(), ax=ax,

                 cbar_ax=cbar_ax,

                 cbar_kws={"orientation": "horizontal"})
balance_cols = [i for i in data.columns if i.startswith('avgbal')]
data[balance_cols].head()
data[balance_cols].describe()


data[data[balance_cols].apply(lambda x : sum(x),axis=1)>0][balance_cols].plot(kind='line', figsize=(40,8))
data[data[balance_cols].apply(lambda x : sum(x),axis=1)>0][balance_cols].quantile(np.arange(0.95,1,0.01))
sns.pairplot(data[data[balance_cols].apply(lambda x : sum(x),axis=1)>0][balance_cols+['targetid']],hue='targetid',size=5)
grid_kws = {"height_ratios": (.9, .03), "hspace": .3}

f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws,figsize=(6,6))

ax = sns.heatmap(data[balance_cols].corr(), ax=ax,

                 cbar_ax=cbar_ax,

                 cbar_kws={"orientation": "horizontal"})
rtl_columns = [i for i in data.columns if re.compile(r'rtl').search(i.lower()) ]
data[rtl_columns]
data[rtl_columns].describe()
data[rtl_columns].quantile(np.arange(0.95,1,0.01))
sns.pairplot(data[rtl_columns+['targetid']],hue='targetid')
grid_kws = {"height_ratios": (.9, .05), "hspace": .3}

f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws,figsize=(40,10))

ax = sns.heatmap(data[rtl_columns].corr(), ax=ax,

                 cbar_ax=cbar_ax,

                 cbar_kws={"orientation": "horizontal"})

rech_columns = [i for i in data.columns if re.compile(r'rech').search(i.lower()) ]
data[rech_columns]
data[rech_columns].describe()
data[rech_columns].quantile(np.arange(0.95,1,0.01))
sns.pairplot(data[rech_columns+['targetid']],hue='targetid')
grid_kws = {"height_ratios": (.9, .05), "hspace": .3}

f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws,figsize=(40,10))

ax = sns.heatmap(data[rech_columns].corr(), ax=ax,

                 cbar_ax=cbar_ax,

                 cbar_kws={"orientation": "horizontal"})
rev_columns = [i for i in data.columns if re.compile(r'rev').search(i.lower()) ]
data[rev_columns]
data[rev_columns].describe()
data[rev_columns].quantile(np.arange(0.95,1,0.01))
sns.pairplot(data[rev_columns+['targetid']],hue='targetid')
grid_kws = {"height_ratios": (.9, .05), "hspace": .3}

f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws,figsize=(40,10))

ax = sns.heatmap(data[rev_columns].corr(), ax=ax,

                 cbar_ax=cbar_ax,

                 cbar_kws={"orientation": "horizontal"})
alr_columns =  [i for i in data.columns if re.compile(r'acd').search(i.lower()) or re.compile(r'ld').search(i.lower()) or re.compile(r'resolve').search(i.lower()) ]
alr_columns
data[alr_columns]
data[alr_columns].describe()
data[alr_columns].quantile(np.arange(0.95,1,0.01))
sns.pairplot(data[alr_columns+['targetid']],hue='targetid')
grid_kws = {"height_ratios": (.9, .05), "hspace": .3}

f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws,figsize=(40,10))

ax = sns.heatmap(data[rev_columns].corr(), ax=ax,

                 cbar_ax=cbar_ax,

                 cbar_kws={"orientation": "horizontal"})
def outlier_effect(column):

    normal_group = data[data[column]<= data[column].quantile(0.99)]['targetid']

    outlier_group = data[data[column]> data[column].quantile(0.99)]['targetid']

    temp = pd.concat([pd.DataFrame(normal_group.value_counts().values, index=['norm_churn_0', 'norm_churn_1']).apply(lambda x : x*100 / len(normal_group)), pd.DataFrame(outlier_group.value_counts().values,index=['out_churn_0', 'out_churn_1']).apply(lambda x : x*100 / len(outlier_group))])

    temp.columns=[column]

    return temp.T
outlier_df=pd.DataFrame()

for i in set(data.columns)- set(inspection_cols)-set(missing_val_columns)-set(drop_Columns)-set(['targetid']):

    temp = outlier_effect(i)

    outlier_df=pd.concat([outlier_df,temp])

outlier_df
outlier_df.plot(kind = 'line', figsize=(24,6));

plt.title('Mirror Image Analyis : Outlier effects of a feature on Target ')
# drop identified columns during eda

def feature_selector_from_raw_data(data,drop_columns):

    if any(pd.Series(data.columns).isin(drop_columns)):

        print('dropping columns : {}'.format(drop_columns))

        X = data.drop(drop_columns,axis=1)

    else:

        X = data

    return X

        
X = feature_selector_from_raw_data(data,drop_Columns)

X.shape
# identifying target and identifiers 

target = 'targetid'

identifiers = ['userID']

#  drop from predictor set

identifiers.append(target)

# segregating predictors and target

X_final = feature_selector_from_raw_data(X,identifiers)

Y_final = data[target]

print('shape of predictor set:',X_final.shape)

print('shape of target:',len(Y_final))
X_train, X_test, y_train, y_test = train_test_split(X_final, Y_final, test_size=0.3,stratify=Y_final)

X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5,stratify=y_test)
print (X_train.shape, y_train.shape)

print (X_test.shape, y_test.shape)

print (X_val.shape, y_val.shape)

sum([any(X_train[i]==-999) for i in  X_train.columns])
numeric_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(missing_values=-999,strategy='mean'))])
numeric_features = X_train.columns

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numeric_transformer, numeric_features)])
rf = Pipeline(steps=[('preprocessor', preprocessor)])
X_train_imputed = rf.fit_transform(X_train)

X_test_imputed = rf.fit_transform(X_test)
classifiers = [

    KNeighborsClassifier(3),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    XGBClassifier()

    ]
for classifier in classifiers:

    pipe = Pipeline(steps=[('preprocessor', preprocessor),

                      ('classifier', classifier)])

    pipe.fit(X_train, y_train)   

    print(classifier)

    print("model score: %.3f" % pipe.score(X_test, y_test))

    print("model score: %.3f" % pipe.score(X_train, y_train))

ITERATIONS=10

space={ 'learning_rate': (0.01, 1.0, 'log-uniform'),

        'min_child_weight': (0, 10),

        'max_depth': (0, 50),

        'max_delta_step': (0, 20),

        'subsample': (0.01, 1.0, 'uniform'),

        'colsample_bytree': (0.01, 1.0, 'uniform'),

        'colsample_bylevel': (0.01, 1.0, 'uniform'),

        'reg_lambda': (1e-9, 1000, 'log-uniform'),

        'reg_alpha': (1e-9, 1.0, 'log-uniform'),

        'gamma': (1e-9, 0.5, 'log-uniform'),

        'n_estimators': (50, 1000),

        'scale_pos_weight': (1e-6, 50, 'log-uniform') }

# Classifier

bayes_cv_tuner = BayesSearchCV(estimator = xgb.XGBClassifier(n_jobs = 1,objective = 'binary:logistic',

                                                             eval_metric = 'auc',silent=1),search_spaces = space,

                               scoring = 'roc_auc',cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42),

                               n_jobs = -1,n_iter = ITERATIONS,verbose = 0,refit = True,random_state = 42)



def status_print(optim_result):      

    best_params = pd.Series(bayes_cv_tuner.best_params_)

    print('Model {} ROC-AUC: {}\n  params: {}\n'.format(

        np.round(bayes_cv_tuner.best_score_, 2),

        bayes_cv_tuner.best_params_))
# Fit the model

result = bayes_cv_tuner.fit(X_train_imputed, y_train.values, callback=status_print)
optimal_params = {'colsample_bylevel': 0.7366877378057127,

                  'colsample_bytree': 0.9399760402267441, 

                  'gamma': 2.6498051478267012e-08,

                  'learning_rate': 0.0238149998729586,

                  'max_delta_step': 16,

                  'max_depth': 19, 

                  'min_child_weight': 2,

                  'n_estimators': 764,

                  'reg_alpha': 0.011683028450342707,

                  'reg_lambda': 0.0048879464985534336,

                  'scale_pos_weight': 0.13267482411031659,

                  'subsample': 0.5689543694097536}

optimal_params['eval_metric'] = 'auc'

optimal_params['n_jobs'] = -1

optimal_params['objective'] = 'binary:logistic'

optimal_params['tree_method'] = 'approx'

optimal_params['verbose'] = 0

optimal_params['importance_type'] = 'gain'

optimal_params['missing'] = -999

# dtrain = xgb.DMatrix(X_train, label=y_train,missing=-999)

# dtest = xgb.DMatrix(X_test, label=y_test,missing=-999)

# dval = xgb.DMatrix(X_val, label=y_val,missing=-999)

# watchlist = [(dtest, 'eval'), (dtrain, 'train')]

# num_round = 200

# bst = xgb.train(optimal_params, dtrain, num_round, watchlist,early_stopping_rounds=20)

#xgb.to_graphviz(bst, num_trees=2)

#plt.rcParams["figure.figsize"] = (15,10)

#xgb.plot_importance(bst, importance_type='gain',max_num_features =25)
clf = xgb.XGBClassifier(n_estimators = optimal_params['n_estimators'], 

                        max_depth = optimal_params['max_depth'],

                        min_child_weight = optimal_params['min_child_weight'],

                        subsample = optimal_params['subsample'],

                        colsample_bylevel = optimal_params['colsample_bylevel'],

                        gamma = optimal_params['gamma'],

                        colsample_bytree = optimal_params['colsample_bytree'],

                        learning_rate = optimal_params['learning_rate'],

                        max_delta_step = optimal_params['max_delta_step'],

                        reg_alpha = optimal_params['reg_alpha'],

                       reg_lambda = optimal_params['reg_lambda'],

                       scale_pos_weight = optimal_params['scale_pos_weight'],

                       importance_type = optimal_params['importance_type'],

                       missing = optimal_params['missing'],

                       seed=42)
clf = xgb.XGBModel(**optimal_params)



clf.fit(X_train_imputed, y_train.values,

        eval_set=[(X_train_imputed, y_train.values), (X_test_imputed, y_test.values)],

        eval_metric='auc',

        early_stopping_rounds =20,

        verbose=True)



evals_result = clf.evals_result()
pd.concat([pd.DataFrame(evals_result['validation_0']),pd.DataFrame(evals_result['validation_1'])], axis=1).plot()
clf.get_params()
feature_imp  = pd.DataFrame([X_train.columns,clf.feature_importances_]).T

feature_imp.columns=['predictor','importance']

feature_imp.sort_values('importance',ascending=False)
pred_train = clf.predict(X_train_imputed)

y_train_pred = np.where(pd.Series(pred_train) > 0.004, 1,0)

pd.DataFrame({'predicted_probabilities':pred_train,'predicted':y_train_pred,'actual':y_train}).to_csv('train_thresold.csv',index= False)

print(classification_report(y_train, y_train_pred))
pred_test = clf.predict(X_test_imputed)

y_test_pred = np.where(pd.Series(pred_test) > 0.004, 1,0)

pd.DataFrame({'predicted_probabilities':pred_test,'predicted':y_test_pred,'actual':y_test}).to_csv('test_thresold.csv',index= False)

print(classification_report(y_test, y_test_pred))
X = np.vstack((X_train_imputed,X_test_imputed))

Y = np.hstack((y_train,y_test))
clf = xgb.XGBModel(**optimal_params)



clf.fit(X, Y,

        eval_set=[(X,Y)],

        eval_metric='auc',

        early_stopping_rounds =20,

        verbose=True)



evals_result = clf.evals_result()
X_val_imputed = rf.fit_transform(X_val)

pred_val = clf.predict(X_val_imputed)

y_val_pred = np.where(pd.Series(pred_val) > 0.004, 1,0)

pd.DataFrame({'predicted_probabilities':pred_val,'predicted':y_val_pred,'actual':y_val}).to_csv('val_thresold.csv',index= False)

print(classification_report(y_val, y_val_pred))
X.shape
X_final.shape
numeric_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(missing_values=-999,strategy='mean')),

    ('scaler', StandardScaler())])



numeric_features = X_final.columns

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numeric_transformer, numeric_features)])



pipe_processor = Pipeline(steps=[('preprocessor', preprocessor)])



X_processed = pipe_processor.fit_transform(X_final)
pca = decomposition.PCA(random_state=42)

pca.fit_transform(X_processed)

plt.figure(figsize=(10,10))

plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)

plt.xlabel('Number of components')

plt.ylabel('Total explained variance')

plt.xlim(0, 63)

plt.yticks(np.arange(0, 1.1, 0.1))

plt.axvline(56, c='b')

plt.axhline(0.95, c='r')

plt.show();


for i, component in enumerate(pca.components_[:5]):

    print("{} component: {}% of initial variance".format(i + 1, 

          round(100 * pca.explained_variance_ratio_[i], 2)))

    temp = {col: value for col, value in zip(X_final.columns,component)}

    df = pd.DataFrame.from_dict(temp,'index',columns=['feature_projection']).sort_values(by='feature_projection')

    print(df.head(5)),print(df.tail(5))

    
pca = decomposition.PCA(n_components=56,random_state=42)

X_reduced = pca.fit_transform(X_processed)
X_reduced.shape
inertia = []

for k in range(1, 40):

    kmeans = KMeans(n_clusters=k, random_state=1).fit(X_reduced)

    inertia.append(np.sqrt(kmeans.inertia_))

plt.figure(figsize=(10,10))

plt.plot(range(1, 40), inertia, marker='s');

plt.xlabel('$k$')

plt.ylabel('sum of sqrd dist b/w observations and their centroids');
kmeans = KMeans(n_clusters=4, random_state=1).fit(X_reduced)

print('Silhouette :', metrics.silhouette_score(X_reduced, kmeans.labels_))

kmeans = KMeans(n_clusters=5, random_state=1).fit(X_reduced)

print('Silhouette :', metrics.silhouette_score(X_reduced, kmeans.labels_))

kmeans = KMeans(n_clusters=6, random_state=1).fit(X_reduced)

print('Silhouette :', metrics.silhouette_score(X_reduced, kmeans.labels_))

kmeans_final = KMeans(n_clusters=4, random_state=1).fit(X_reduced)
X['labels']=kmeans_final.labels_
X.head()
X.groupby(['labels','targetid']).agg({'userID': 'count'}).groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
X_final[X['labels']==0].describe().T
temp_0 = X_final[X['labels']==0].describe().T

temp_0.columns = ['count_0','mean_0','std_0','min_0','25%_0','50%_0','75%_0','max_0']

temp_1 = X_final[X['labels']==1].describe().T

temp_1.columns = ['count_1','mean_1','std_1','min_1','25%_1','50%_1','75%_1','max_1']

temp_2 = X_final[X['labels']==2].describe().T

temp_2.columns = ['count_2','mean_2','std_2','min_2','25%_2','50%_2','75%_2','max_2']

temp_3 = X_final[X['labels']==3].describe().T

temp_3.columns = ['count_3','mean_3','std_3','min_3','25%_3','50%_3','75%_3','max_3']
profile_data = pd.concat([temp_0,temp_1,temp_2,temp_3],axis=1)
profile_data.loc[bill_amount_cols,['mean_0','mean_1','mean_2','mean_3']].plot(kind='barh', )
profile_data.loc[bill_count_cols,['mean_0','mean_1','mean_2','mean_3']].plot(kind='barh', )
profile_data.loc[bill_derived_cols,['mean_0','mean_1','mean_2','mean_3']].plot(kind='barh', )
profile_data.loc[rech_columns,['mean_0','mean_1','mean_2','mean_3']].plot(kind='barh',figsize=(10,10) )
profile_data.loc[rev_columns,['mean_0','mean_1','mean_2','mean_3']].plot(kind='barh',figsize=(10,10) )
profile_data.loc[rtl_columns,['mean_0','mean_1','mean_2','mean_3']].plot(kind='barh',figsize=(10,10) )
profile_data.loc[balance_cols,['mean_0','mean_1','mean_2','mean_3']].plot(kind='barh' )
profile_data.loc[alr_columns,['mean_0','mean_1','mean_2','mean_3']].plot(kind='barh',figsize=(10,10) )