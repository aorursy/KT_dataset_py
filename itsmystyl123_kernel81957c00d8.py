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

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
from scipy.stats import boxcox 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
train_df=pd.read_csv("../input/train.csv")
test_df=pd.read_csv("../input/test.csv")
submission_df=pd.read_csv("../input/sampleSubmission.csv")
submission_df.head()
import numpy as np  # linear algebra
import pandas as pd  # read and wrangle dataframes
import matplotlib.pyplot as plt # visualization
import seaborn as sns # statistical visualizations and aesthetics
from sklearn.base import TransformerMixin # To create new classes for transformations
from sklearn.preprocessing import (FunctionTransformer, StandardScaler) # preprocessing 
from sklearn.decomposition import PCA # dimensionality reduction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.stats import boxcox # data transform
from sklearn.model_selection import (train_test_split, KFold , StratifiedKFold, 
                                     cross_val_score, GridSearchCV, 
                                     learning_curve, validation_curve) # model selection modules
from sklearn.pipeline import Pipeline # streaming pipelines
from sklearn.base import BaseEstimator, TransformerMixin # To create a box-cox transformation class
from collections import Counter
import warnings
# load models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import (XGBClassifier, plot_importance)
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from time import time
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR


# train_df=pd.read_csv("../input/train.csv")
# train_df.head()
# import matplotlib.pyplot as plt
# import seaborn as sns
# # plt.hist(train_df['target'])
# # f, ax = plt.subplots(figsize=(20, 8))
# # train_df['target'].value_counts().plot('bar')
# # sns.heatmap(train_df)
# train_df.isnull().sum()
# train_df.fillna(value=np.nan, inplace=True)

# # Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)

# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# # load data
# # url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
# # names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# # dataframe = pandas.read_csv(url, names=names)
# array = train_df.values
# # print (array)
# X = array[:,0:94]
# Y = array[:,94]
# # feature extraction
# test = SelectKBest(score_func=chi2, k=60)
# fit = test.fit(X, Y)
# # # summarize scores
# np.set_printoptions(precision=7)
# print(fit.scores_)
# features = fit.transform(X)
# # summarize selected features
# print(features[0:5,:])
# # print (type(features))
# print('Feature list:', features.columns_)
# array=train_df.values
# X = array[:,0:94]
# Y = array[:,94]
# sns.countplot(Y,label='count')


# train_df.describe()
# # train_df.feat_3.unique()
# # plt.scatter(train_df['feat_3'],train_df['target'])
# y=train_df.target
# list = ['id','target']
# x = train_df.drop(list,axis = 1 )
# print(type(x))
# print(y.shape)
# data=pd.concat([y,train_df.iloc[:,1:11]],axis=1)
# data=pd.melt(data,id_vars="target",var_name="features" ,value_name="value")
# # data
# plt.figure(figsize=(10,10))
# sns.violinplot(x="features", y="value", hue="target", data=data,split=False, inner="quart")
# plt.xticks(rotation=90)
# tic = time.time()
# sns.swarmplot(x="features", y="value", hue="target", data=data)
# plt.xticks(rotation=90)
# correlation map
# f,ax = plt.subplots(figsize=(50, 50))
# sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
# corrmap=train_df.corr().abs()
# corrmap
# upper = corrmap.where(np.triu(np.ones(corrmap.shape), k=1).astype(np.bool))
# # upper
# to_drop = [column for column in upper.columns if any(upper[column] > 0.7)]
# print(to_drop)
# train_df_new=train_df.drop(to_drop,axis=1)
# train_df_new=train_df_new.drop(['target'],axis=1)
# train_df_new.columns
# df=x
# def get_redundant_pairs(df):
#     '''Get diagonal and lower triangular pairs of correlation matrix'''
#     pairs_to_drop = set()
#     cols = df.columns
#     for i in range(0, df.shape[1]):
#         for j in range(0, i+1):
#             pairs_to_drop.add((cols[i], cols[j]))
#     return pairs_to_drop

# def get_top_abs_correlations(df, n=5):
#     au_corr = df.corr().abs().unstack()
#     labels_to_drop = get_redundant_pairs(df)
#     au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
#     return au_corr[0:n]

# print("Top Absolute Correlations")
# print(get_top_abs_correlations(df,10))

# Top Absolute Correlations
# feat_39  feat_45    0.824146
# feat_3   feat_46    0.777517
# feat_15  feat_72    0.764664
# feat_30  feat_84    0.716862
# feat_9   feat_64    0.702951
test_df.head()

# y=train_df.target
# # split data train 70 % and test 30 %
# x_train, x_test, y_train, y_test = train_test_split(train_df_new, y, test_size=0.3, random_state=42)

# #random forest classifier with n_estimators=10 (default)
# clf_rf = RandomForestClassifier(random_state=43)      
# clr_rf = clf_rf.fit(x_train,y_train)
# pred_y=clf_rf.predict(x_test)
# ac = accuracy_score(y_test,pred_y)
# pred_y
# print('Accuracy is: ',ac)
# cm = confusion_matrix(y_test,clf_rf.predict(x_test))
# sns.heatmap(cm,annot=True,fmt="d")
# pred_y
# #select best
# drop_list=['id','target']
# X=train_df.drop(drop_list,axis=1)
# y=train_df.target

# test=SelectKBest(chi2,k=10)
# fit=test.fit(X,y)
# print('Score list:', fit.pvalues_)
# X_1=fit.transform(X)
# # y_1=fit.transform(y)
# y_1=y
# x_train, x_test, y_train, y_test = train_test_split(X_1, y_1, test_size=0.3, random_state=42)
# clf_rf_2 = RandomForestClassifier()   
# clr_rf_2 = clf_rf_2.fit(x_train,y_train)
# ac_2 = accuracy_score(y_test,clf_rf_2.predict(x_test))
# print('Accuracy is: ',ac_2)
# cm_2 = confusion_matrix(y_test,clf_rf_2.predict(x_test))
# sns.heatmap(cm_2,annot=True,fmt="d") 

# #Recursive feature elimination
# from sklearn.feature_selection import RFE

# drop_list=['id','target','feat_45', 'feat_46', 'feat_64', 'feat_72', 'feat_84']
# X=train_df.drop(drop_list,axis=1)
# y=train_df.target
# #boxcox normalization
# bc_features=[]
# for col in X.columns:
#     bc_transformed,_=boxcox(X[col]+1)
#     bc_features.append(bc_transformed)
    
# bc_features = np.column_stack(bc_features)
# df_bc=pd.DataFrame(data=bc_features,columns=X.columns)
# X=df_bc
# clf_rf_3 = RandomForestClassifier()      
# rfe = RFE(estimator=clf_rf_3, n_features_to_select=38, step=1)
# fit=rfe.fit(X,y)
# X_1=fit.transform(X)
# y_1=y
# x_train, x_test, y_train, y_test = train_test_split(X_1, y_1, test_size=0.3, random_state=42)

# clf_rf_rfe = RandomForestClassifier()   
# clr_rf_rfe = clf_rf_rfe.fit(x_train,y_train)
# ac_2 = accuracy_score(y_test,clf_rf_rfe.predict(x_test))
# print('Accuracy is: ',ac_2)


# drop_list=['id','target','feat_45', 'feat_46', 'feat_64', 'feat_72', 'feat_84']
# X=train_df.drop(drop_list,axis=1)
# y=train_df.target
# #boxcox normalization
# bc_features=[]
# for col in X.columns:
#     bc_transformed,_=boxcox(X[col]+1)
#     bc_features.append(bc_transformed)
    
# bc_features = np.column_stack(bc_features)
# df_bc=pd.DataFrame(data=bc_features,columns=X.columns)
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# n_components = 30
# pipelines = []
# n_estimators = 50
# seed=40
# #print(df.shape)
# pipelines.append( ('SVC',
#                    Pipeline([
#                               ('sc', StandardScaler()),
#                               ('pca', PCA(n_components = n_components, random_state=seed ) ),
#                              ('SVC', SVC(random_state=seed))]) ) )


# pipelines.append(('KNN',
#                   Pipeline([ 
#                               ('sc', StandardScaler()),
#                             ('pca', PCA(n_components = n_components, random_state=seed ) ),
#                             ('KNN', KNeighborsClassifier()) ])))
# pipelines.append( ('RF',
#                    Pipeline([
#                               ('sc', StandardScaler()),
#                              ('pca', PCA(n_components = n_components, random_state=seed ) ), 
#                              ('RF', RandomForestClassifier(random_state=seed, n_estimators=n_estimators)) ]) ))


# pipelines.append( ('ADA',
#                    Pipeline([ 
#                               ('sc', StandardScaler()),
#                              ('pca', PCA(n_components = n_components, random_state=seed ) ), 
#                     ('ADA', AdaBoostClassifier(random_state=seed,  n_estimators=n_estimators)) ]) ))

# pipelines.append( ('ET',
#                    Pipeline([
#                               ('sc', StandardScaler()),
#                              ('pca', PCA(n_components = n_components, random_state=seed ) ), 
#                              ('ET', ExtraTreesClassifier(random_state=seed, n_estimators=n_estimators)) ]) ))
# # pipelines.append( ('GB',
# #                    Pipeline([ 
# #                              ('sc', StandardScaler()),
# #                             ('pca', PCA(n_components = n_components, random_state=seed ) ), 
# #                              ('GB', GradientBoostingClassifier(random_state=seed)) ]) ))

# pipelines.append( ('LR',
#                    Pipeline([
#                               ('sc', StandardScaler()),
#                               ('pca', PCA(n_components = n_components, random_state=seed ) ), 
#                              ('LR', LogisticRegression(random_state=seed)) ]) ))

# results, names, times  = [], [] , []
# num_folds = 10
# scoring = 'accuracy'

# for name, model in pipelines:
#     start = time()
#     kfold = StratifiedKFold(n_splits=num_folds, random_state=seed)
#     cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring = scoring,
#                                 n_jobs=-1) 
#     t_elapsed = time() - start
#     results.append(cv_results)
#     names.append(name)
#     times.append(t_elapsed)
#     msg = "%s: %f (+/- %f) performed in %f seconds" % (name, 100*cv_results.mean(), 
#                                                        100*cv_results.std(), t_elapsed)
#     print(msg)


# fig = plt.figure(figsize=(12,8))    
# fig.suptitle("Algorithms comparison")
# ax = fig.add_subplot(1,1,1)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()
# n_components = 30
# pipelines = []
# n_estimators = 50
# seed=40
# #print(df.shape)
# pipelines.append( ('SVC',
#                    Pipeline([
#                               ('sc', StandardScaler()),
#                               ('pca', PCA(n_components = n_components, random_state=seed ) ),
#                              ('SVC', SVC(random_state=seed))]) ) )

# print ((pipelines) )
# for name, model in pipelines:
#     print (type(pipelines) )
#     print ('**********' )
#     print ((model ))

        
print(pipelines.named_steps )
# Create a pipeline with a Random forest classifier

# drop_list=['id','target','feat_45', 'feat_46', 'feat_64', 'feat_72', 'feat_84']
# X=train_df.drop(drop_list,axis=1)
# y=train_df.target
# #boxcox normalization
# bc_features=[]
# for col in X.columns:
#     bc_transformed,_=boxcox(X[col]+1)
#     bc_features.append(bc_transformed)
    
# bc_features = np.column_stack(bc_features)
# df_bc=pd.DataFrame(data=bc_features,columns=X.columns)
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# # pipe_rfc = Pipeline([ 
# #                       ('scl', StandardScaler()), 
# #                     ('rfc', RandomForestClassifier(random_state=42, n_jobs=-1) )])

# pipelines.append( ('rfc',
#                    Pipeline([
#                               ('sc', StandardScaler()),
#                              ('pca', PCA(n_components = n_components, random_state=seed ) ), 
#                              ('rfc', RandomForestClassifier(random_state=seed, n_estimators=n_estimators)) ]) ))
# pipelines.append( ('SVC',
#                    Pipeline([
#                               ('sc', StandardScaler()),
#                               ('pca', PCA(n_components = n_components, random_state=seed ) ),
#                              ('SVC', SVC(random_state=seed))]) ) )

# # Set the grid parameters
# # param_grid_rfc =  [ {rfc:{
# #     'rfc__n_estimators': [20, 30,30,50], # number of estimators
# #     #'rfc__criterion': ['gini', 'entropy'],   # Splitting criterion
# #     'rfc__max_features':[0.05 , 0.1], # maximum features used at each split
# #     'rfc__max_depth': [None, 5], # Max depth of the trees
# #     'rfc__min_samples_split': [0.005, 0.01], # mininal samples in leafs
# #     })]
# param_grid_rfc =   {'rfc':{
#     'rfc__n_estimators': [20, 30,30,50], # number of estimators
#     #'rfc__criterion': ['gini', 'entropy'],   # Splitting criterion
#     'rfc__max_features':[0.05 , 0.1], # maximum features used at each split
#     'rfc__max_depth': [None, 5], # Max depth of the trees
#     'rfc__min_samples_split': [0.005, 0.01], # mininal samples in leafs
#     },
#      'SVC':{}              }
# # abc=[param_grid_rfc['rfc']]
# # Use 10 fold CV
# scoring = 'accuracy'
# for name,model in pipelines:
#     print (model)
#     print (name)
   
#     param_grid_val=[param_grid_rfc[name]]
#     print(param_grid_val)
#     kfold = StratifiedKFold(n_splits=5, random_state= 42)
#     grid_rfc = GridSearchCV(model, param_grid= param_grid_val, cv=kfold, scoring=scoring, verbose= 1, n_jobs=-1)

#     #Fit the pipeline
#     start = time()
#     grid_rfc = grid_rfc.fit(x_train, y_train)
#     end = time()

#     print("RFC grid search took %.3f seconds" %(end-start))

#     # Best score and best parameters
#     print('-------Best score----------')
#     print(grid_rfc.best_score_ * 100.0)
#     print('-------Best params----------')
#     print(grid_rfc.best_params_)
# param_grid_rfc =   {'rfc':{
#     'rfc__n_estimators': [20, 30,30,50], # number of estimators
#     #'rfc__criterion': ['gini', 'entropy'],   # Splitting criterion
#     'rfc__max_features':[0.05 , 0.1], # maximum features used at each split
#     'rfc__max_depth': [None, 5], # Max depth of the trees
#     'rfc__min_samples_split': [0.005, 0.01], # mininal samples in leafs
#     }}
# abc=[param_grid_rfc['rfc']]
# print(abc)
# mask = rfe.get_support(indices=True)
# print(mask)
# columns=X.columns[rfe.get_support()]
# # X_trans = pd.DataFrame(X_1,columns=columns)
# X_trans=train_df[columns]
# X_trans.head()
# df=X_trans
# for feat in columns:
#     skew = df[feat].skew()
#     sns.distplot(df[feat], kde= False, label='Skew = %.3f' %(skew), bins=40)
#     plt.legend(loc='best')
#     plt.show()
# df=train_df.corr()
# df=df.reindex()
# df.columns
# df=train_df_out

# bc_features=[]
# for col in train_df.columns:
#     bc_transformed,_=boxcox(df[col]+1)
#     bc_features.append(bc_transformed)
    
# bc_features = np.column_stack(bc_features)
# df_bc=pd.DataFrame(data=bc_features,columns=columns)
# print(type(y_1))
# # y_trans = pd.DataFrame(y,columns=['target'])
# # df_bc['target']=train_df['target']
# x_train, x_test, y_train, y_test = train_test_split(df_bc, train_df_out['target'], test_size=0.3, random_state=42)
# clf_rf_rfe = RandomForestClassifier()   
# clr_rf_rfe = clf_rf_rfe.fit(x_train,y_train)
# ac_2 = accuracy_score(y_test,clf_rf_rfe.predict(x_test))

# print('Accuracy is: ',ac_2)

#     fig, ax = plt.subplots(1,2,figsize=10,10))  
# sns.pairplot(df_bc)

# # Detect observations with more than one outlier
# from collections import Counter
# def outlier_hunt(df):
#     """
#     Takes a dataframe df of features and returns a list of the indices
#     corresponding to the observations containing more than 2 outliers. 
#     """
#     outlier_indices = []
    
#     # iterate over features(columns)
#     for col in df.columns.tolist():
#         # 1st quartile (25%)
#         Q1 = np.percentile(df[col], 25)
        
#         # 3rd quartile (75%)
#         Q3 = np.percentile(df[col],75)
        
#         # Interquartile rrange (IQR)
#         IQR = Q3 - Q1
        
#         # outlier step
#         outlier_step = 1.5 * IQR
        
#         # Determine a list of indices of outliers for feature col
#         outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
#         # append the found outlier indices for col to the list of outlier indices 
#         outlier_indices.extend(outlier_list_col)
        
#     # select observations containing more than 2 outliers
#     outlier_indices = Counter(outlier_indices)        
#     multiple_outliers = list( k for k, v in outlier_indices.items() if v > 2 )
    
#     return multiple_outliers   

# print('The dataset contains %d observations with more than 2 outliers' %(len(outlier_hunt(train_df[columns]))))   

# outlier_df=outlier_hunt(train_df[columns])
# print((outlier_df))
# outlier_df.head()
# abc=pd.DataFrame()
# abc['feat_10']=np.log(train_df['feat_10']+1)
# f,ax = plt.subplots(figsize=(10, 10))
# plt.xticks(rotation=90)
# sns.boxplot(x=abc['feat_10'],data=train_df)
# plt.show()
# plt.xticks(rotation=90)

# train_df[columns]
# abc=train_df.groupby("feat_11")["feat_11"].count()
# train_df_out=train_df.drop(train_df.index[outlier_df])/
# train_df.describe()
# for feature in columns:
#     fig, ax = plt.subplots(1,2,figsize=(7,3.5))    
#     ax[0].hist(df[feature], color='blue', bins=30, alpha=0.3, label='Skew = %s' %(str(round(df[feature].skew(),3))) )
#     ax[0].set_title(str(feature))   
#     ax[0].legend(loc=0)
#     ax[1].hist(df_bc[feature], color='red', bins=30, alpha=0.3, label='Skew = %s' %(str(round(df_bc[feature].skew(),3))) )
#     ax[1].set_title(str(feature)+' after a Box-Cox transformation')
#     ax[1].legend(loc=0)
#     plt.show()
# PCA feature Extraction
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# scl=StandardScaler()
# pca=PCA(n_components=8)
# drop_list=['id','target','feat_45', 'feat_46', 'feat_64', 'feat_72', 'feat_84']
# X=train_df.drop(drop_list,axis=1)
# y=train_df.target
# X_1=scl.fit_transform(X)
# pca.fit(X_1)
# print(pca.explained_variance_ )
# X_2 = pca.transform(X_1)
# x_train, x_test, y_train, y_test = train_test_split(X_2, y, test_size=0.3, random_state=42)

# clf_rf_rfe = RandomForestClassifier()   
# clr_rf_rfe = clf_rf_rfe.fit(x_train,y_train)
# ac_2 = accuracy_score(y_test,clf_rf_rfe.predict(x_test))
# print('Accuracy is: ',ac_2)

# # plt.figure(1, figsize=(14, 13))
# # plt.clf()
# # plt.axes([.2, .2, .7, .7])
# # plt.plot(pca.explained_variance_ratio_, linewidth=2)
# # plt.axis('tight')
# # plt.xlabel('n_components')
# # plt.ylabel('explained_variance_ratio_')
# y_test.value_counts().plot('bar')


# # train_df['target'].value_counts().plot('bar')
# # a=['1','2','3','4','5','6','7','8','9','10']
# a=[1,2,3,4,5,6,7,8,9,10]
# df=pd.DataFrame(a,columns=['val'])
# Q1 = np.percentile(df['val'], 25
#                   )
# Q1
param_grid_rfc= {'RF':{
    'RF__n_estimators': [20], # number of estimators
    'RF__criterion': ['gini'],   # Splitting criterion  ['gini', 'entropy']
    'RF__max_features':['log2'], # maximum features used at each split ['log2','sqrt',None]
    'RF__max_depth': [5,8,15,25,30,None], # Max depth of the trees[,8,15,25,30,None]
    'RF__min_samples_split': [2,5,10,15,100], # mininal samples in leafs [1.1,2,5,10,15,100]
#      'RF__min_samples_leaf': [1,2,5,10]
    },
     'SVC':{
         'SVC__C': [0.001,0.01,10,100],
         'SVC__gamma':['auto'],
         'SVC__class_weight':['balanced',None]
    } ,     
    'LR':{
        'LR__penalty': ['L1'],
        'LR__C': [0.001,0.01,10,100]
    },
     'KNN':{
        'KNN__n_neighbors': [2,4,8,16,32],
        'KNN__p': [2,3]
    } , 
    'XGB':{
    'XGB__eta': [0.01,0.015,0.025,0.05,0.1], # number of estimators
    'XGB__gamma': [0.05,0.1,0.3,0.5,0.7,0.9,1.0],   # Splitting criterion
    'XGB__max_depth':[3,5,7,9,12,15,17,25], # maximum features used at each split
    'XGB__min_child_weight': [1,3,5,7], # Max depth of the trees
    'XGB__sub_sample': [0.6,0.7,0.8,0.9,1.0], # mininal samples in leafs
    'XGB__calsample_bytree': [0.6,0.7,0.8,0.9,1.0],
    'XGB__lambda' :[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]   ,
    'XGB__alpha' :[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
     } ,
     'LASOO' : {
         'LASOO__alpha': [0.1,1.0,10],
         'LASOO__normalize':['True','False']
     },
     'RIDGE' : {
         'LASOO__alpha': [0.01,0.1,1.0,10,100],
         'LASOO__normalize':['True','False'],
         'LASOO__fit_intercept':['True','False'],
     'ADA' : {}  ,
     'ET' :{}  
    }}
     
     
    
drop_list=['id','target','feat_45', 'feat_46', 'feat_64', 'feat_72', 'feat_84']
X=train_df.drop(drop_list,axis=1)
y=train_df.target
#boxcox normalization
bc_features=[]
for col in X.columns:
    bc_transformed,_=boxcox(X[col]+1)
    bc_features.append(bc_transformed)
    
bc_features = np.column_stack(bc_features)
df_bc=pd.DataFrame(data=bc_features,columns=X.columns)
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# n_components = 30
# pipelines = []
# n_estimators = 50
# seed=40
rfc=RandomForestClassifier(random_state=42, n_estimators=25,max_depth=6,min_samples_leaf=5)
rfecv = RFECV(rfc, step=5, cv=StratifiedKFold(3),scoring='accuracy')
rfecv = rfecv.fit(df_bc, y)

rfecv = rfecv.fit(X, y)
print(rfecv.support_ )
print(rfecv.ranking_)
print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
columns=X.columns[rfecv.get_support()]
print(columns)
X_1 = rfecv.transform(X)

# n_components = 30
# pipelines = []
# n_estimators = 10
# seed=40
#print(df.shape)
# pipelines.append( ('SVC',
#                    Pipeline([
# #                               ('sc', StandardScaler()),
# #                               ('pca', PCA(n_components = n_components, random_state=seed ) ),
#                              ('SVC', SVC(random_state=seed))]) ) )


# pipelines.append(('KNN',
#                   Pipeline([ 
# #                               ('sc', StandardScaler()),
# #                             ('pca', PCA(n_components = n_components, random_state=seed ) ),
#                             ('KNN', KNeighborsClassifier()) ])))
# pipelines.append( ('RF',
#                    Pipeline([
# #                               ('sc', StandardScaler()),
# #                              ('pca', PCA(n_components = n_components, random_state=seed ) ), 
#                              ('RF', RandomForestClassifier(random_state=seed, n_estimators=n_estimators)) ]) ))


# pipelines.append( ('ADA',
#                    Pipeline([ 
# #                               ('sc', StandardScaler()),
# #                              ('pca', PCA(n_components = n_components, random_state=seed ) ), 
#                     ('ADA', AdaBoostClassifier(random_state=seed,  n_estimators=n_estimators)) ]) ))

# pipelines.append( ('ET',
#                    Pipeline([
# #                               ('sc', StandardScaler()),
# #                              ('pca', PCA(n_components = n_components, random_state=seed ) ), 
#                              ('ET', ExtraTreesClassifier(random_state=seed, n_estimators=n_estimators)) ]) ))
# pipelines.append( ('GB',
#                    Pipeline([ 
#                              ('sc', StandardScaler()),
#                             ('pca', PCA(n_components = n_components, random_state=seed ) ), 
#                              ('GB', GradientBoostingClassifier(random_state=seed)) ]) ))

# pipelines.append( ('LR',
#                    Pipeline([
# #                               ('sc', StandardScaler()),
# #                               ('pca', PCA(n_components = n_components, random_state=seed ) ), 
#                              ('LR', LogisticRegression(random_state=seed)) ]) ))


# results, names, times  = [], [] , []
# num_folds = 2
# scoring = 'accuracy'

# for name,model in pipelines:
#     print (model)
#     print (name)
   
#     param_grid_val=[param_grid_rfc[name]]
#     print('grid val=%s' %param_grid_val)
#     print('model param=%s' %model.get_params().keys())
#     kfold = StratifiedKFold(n_splits=2, random_state= 42)
#     grid_rfc = GridSearchCV(model, param_grid= param_grid_val, cv=kfold, scoring=scoring, verbose= 1, n_jobs=1)

#     #Fit the pipeline
#     start = time()
#     grid_rfc = grid_rfc.fit(X_1, y)
#     end = time()

#     print("RFC grid search took %.3f seconds" %(end-start))

#     # Best score and best parameters
#     print('-------Best score----------')
#     print(grid_rfc.best_score_ * 100.0)
#     print('-------Best params----------')
#     print(grid_rfc.best_params_)
# {'RF__criterion': 'gini', 'RF__max_depth': None, 'RF__max_features': 'log2', 'RF__min_samples_split': 5, 'RF__n_estimators': 20}

#random forest classifier with n_estimators=10 (default)
clf_rf = RandomForestClassifier(random_state=43,criterion='gini', max_depth= None, max_features= 'log2', min_samples_split= 5, n_estimators= 20)      
clr_rf = clf_rf.fit(X_1,y)
pred_y=clf_rf.predict(test_df[columns])
# ac = accuracy_score(y_test,pred_y)
pred_y
# print('Accuracy is: ',ac)
# cm = confusion_matrix(y_test,clf_rf.predict(x_test))
# sns.heatmap(cm,annot=True,fmt="d")
# pred_y
# test_df
df = pd.DataFrame(pred_y)
# df
submit=pd.concat([test_df[['id']], df], axis=1)
# abc=submit.get_dummies(0)
abc=pd.get_dummies(submit, prefix=['0'])
abc.columns = abc.columns.str.replace('0_','')
abc
abc.to_csv('submission.csv', index=False)