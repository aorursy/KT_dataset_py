from time import time
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
% matplotlib inline
import seaborn as sns
# Input data files are available in the "../input/" directory.

import os
print(os.listdir("../input"))

raw = pd.read_csv('../input/winequality-red.csv')
raw.head()
raw.describe()
sns.pairplot(raw,diag_kind='kde',kind='reg',vars=['quality','fixed acidity','chlorides','pH','sulphates','alcohol'])
# Principal Component Analysis to find most important features
X = raw.drop('quality',axis=1)
X
# step 1: center and normalize features 
C = (X-np.mean(X))/np.std(X)
C
# step 2: compute covariance matrix of centered features
V = np.cov(C.T)
V
print('shape of cov={}'.format(V.shape))
# step 3: compute PC loadings (directions in feature space which have most variation)
eigvals,eigvecs = np.linalg.eig(V)
# enforce descending variance (eigenvalues)
ix = eigvals.argsort()[::-1] 
eigvals,eigvecs = eigvals[ix],eigvecs[:,ix]
loadingsheader = ['L'+str(i) for i in range(1,len(X.columns)+1)]
loadingsdf = pd.DataFrame(eigvecs,columns=loadingsheader,index=X.columns)
display(loadingsdf)
# step 4: compute PCs (i.e. scores: project features X onto loading vectors)
scores = loadingsdf.values.T.dot(C.T)
scoresheader = ['PC'+str(i) for i in range(1,len(C.columns)+1)]
scoresdf = pd.DataFrame(scores.T,columns=scoresheader,index=C.index)
display(scoresdf.head())
def screeplot(eigvals):
    '''
    function which computes percent variance explained plot
    eigvals   : eigenvalues returned by PCA
    '''
    with plt.style.context('seaborn-white'):
        f,ax=plt.subplots(figsize=(14,8))
        x = np.arange(1,len(eigvals)+1,1)
        ax.set_xticks(x)
        totalvar = eigvals.sum()
        pve = eigvals/totalvar
        cumpve = np.cumsum(pve)
        ax.plot(x,pve,label='pve')
        ax.plot(x,cumpve,label='cumpve')
        ax.set(title='Percent Variance Explained',xlabel='PC',ylabel='eigenvalue (loading variance %)')
        ax.axhline(y=0,color='k',linestyle='dotted')
        ax.axhline(y=1,color='k',linestyle='dotted')
        ax.legend(loc='best')

def biplot(loadingdf,scoredf,loadcolor='',scorecolor='',load_axlim=7,score_axlim=7,load_arrows=4):
    '''
    functon which plots first two PCs
    loadingdf        : loading vectors, DataFrame
    scoredf          : score vectors, DataFrame
    load,score_color : color of loadings,scores,str
    load,score_axlim : scale of loading,score axes, flt
    load_arrows      : size of loading arrow heads, flt
    '''
    with plt.style.context('seaborn-white'):
        f = plt.figure(figsize=(12,12))
        ax0 = plt.subplot(111)
        for ix in scoredf.index:
            # scatter scores onto 2d surface
            ax0.annotate(ix,(scoredf['PC1'][ix],-scoredf['PC2'][ix]),ha='center',color=scorecolor)
        ax0.set(xlim=(-score_axlim,score_axlim),ylim=(-score_axlim,score_axlim))
        ax0.set_xlabel('Principal Component 1',color=scorecolor)
        ax0.set_ylabel('Principal Component 2',color=loadcolor)
        # add ref line sthrough origin
        ax0.hlines(y=0,xmin=-score_axlim,xmax=score_axlim,linestyle='dotted',color='grey')
        ax0.vlines(x=0,ymin=-score_axlim,ymax=score_axlim,linestyle='dotted',color='grey')
        # overlay first two loading vector weights
        ax1 = ax0.twinx().twiny()
        ax1.set(xlim=(-load_axlim,load_axlim),ylim=(-load_axlim,load_axlim))
        ax1.tick_params(axis='y',color='red')
        ax1.set_xlabel('Principal Component Loading Weights')
        offset_scalar = 1.15
        for feature in loadingdf.index:
            ax1.annotate(feature,(loadingdf['L1'].loc[feature]*offset_scalar,-loadingdf['L2'].loc[feature]*offset_scalar),color=loadcolor)
        # display PCs as arrows
        for i in range(0,load_arrows):
            ax1.arrow(x=0,y=0,dx=loadingdf['L1'][i],dy=-loadingdf['L2'][i],head_width=0.009,shape='full')
screeplot(eigvals)
biplot(loadingsdf,scoresdf,loadcolor='red',scorecolor='lightblue',load_axlim=1,score_axlim=6,load_arrows=len(loadingsdf.columns))
# Assign target and features
X,y = raw.drop('quality',axis=1),raw[['quality']]
from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.3,random_state=0)


from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split,cross_validate,cross_val_predict,cross_val_score,validation_curve
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.model_selection import GridSearchCV

def pca_regression(estimator,X,y,testsplit=.3,seed=0,paramgrid={},folds=5):
    '''
    function which computes PCA regression
    estimator : linear estimator, sklearn linear estim ator object
    X,y       : features, target, ndarrays
    seed      : random seed, int
    folds     : cross validation folds, int 
    '''
    # interaction features
    pc1features = X[['citric acid','fixed acidity','sulphates','density','chlorides','density','pH']]
    Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=testsplit,random_state=seed)
    pc1features['pH2'] = pc1features['pH']*pc1features['pH']
    pc1features['fixedacidity2'] = pc1features['fixed acidity'] * pc1features['fixed acidity']
    pc1features['alcohol_pH'] = X['alcohol'] * pc1features['pH']
    pc1features['alcohol_fixedacidity'] = X['alcohol'] * pc1features['fixed acidity']
    pc1features['totalsulfurdioxide_pH'] = X['total sulfur dioxide'] * pc1features['pH']
    pc1features['freesulfurdioxide_fixedacidity'] = X['free sulfur dioxide'] * pc1features['fixed acidity']
    degbest,rmsebest,r2best,optimalestimator=1,np.inf,0,None
    pipe = make_pipeline(SimpleImputer(strategy='median'),
                         estimator)
    # cross validate hyperparameters (arbitrarily choose highest PC1 vs. PC2 interactions)
    gscv = GridSearchCV(pipe,cv=folds,param_grid=paramgrid,scoring='neg_mean_squared_error')
    gscv.fit(X,y)
    bestestimator = gscv.best_estimator_
    # cross validate scores
    cvobj = cross_validate(bestestimator,X,y,cv=folds,return_train_score=True,scoring=['r2','neg_mean_squared_error'])
    cv_trainr2,cv_valr2 = cvobj['train_r2'],cvobj['test_r2']
    cv_trainrmse,cv_valrmse =np.sqrt(-cvobj['train_neg_mean_squared_error']),np.sqrt(-cvobj['test_neg_mean_squared_error'])
    # enforce nonnegative score values
    cv_trainr2 = np.where(cv_trainr2<0,np.nan,cv_trainr2)
    cv_valr2 = np.where(cv_valr2<0,np.nan,cv_valr2)
    trainr2,valr2 = np.nanmean(cv_trainr2),np.nanmean(cv_valr2)
    trainrmse,valrmse = np.nanmean(cv_trainrmse),np.nanmean(cv_valrmse)
    print('\nDegree {} Train CV Mean R2 = {:,.3f}'.format(degbest,trainr2))
    print('Degree {} Validate CV Mean R2 = {:,.3f}'.format(degbest,valr2))
    print('\nDegree {} Train CV Mean RMSE = {:,.3f}'.format(degbest,trainrmse))
    print('Degree {} Validate CV Mean RMSE = {:,.3f}'.format(degbest,valrmse))
    # cv test yhat
    cv_testpred = cross_val_predict(bestestimator,Xtest,ytest,cv=folds).ravel()
    # cv residuals
    cv_testresids = (ytest.as_matrix()-cv_testpred.mean().ravel()).ravel()
    cv_testmse = np.square(cv_testresids).mean()
    cv_testrmse = np.sqrt(cv_testmse)
    # test cv r2score
    cv_testscore = cross_val_score(bestestimator,Xtest,ytest,cv=folds,scoring='r2')
    cv_testr2 = np.where(cv_testscore<0,np.nan,cv_testscore)
    testr2 = np.nanmean(cv_testr2)
    # enforce nonnegative test r2
    
    return degbest,cv_testrmse,testr2,bestestimator

# pca_regression(LinearRegression(),X,y,testsplit=.3,seed=0,paramgrid={'linearregression__normalize':[True,False],'linearregression__fit_intercept':[True,False]},folds=5)
pca_regression(Ridge(),X,y,testsplit=.3,seed=0,paramgrid={'ridge__normalize':[True,False],'ridge__fit_intercept':[True,False],'ridge__alpha':np.arange(0,.1,1e-3)},folds=5)
# pca_regression(Lasso(),X,y,testsplit=.3,seed=0,paramgrid={'lasso__normalize':[True,False],'lasso__fit_intercept':[True,False],'lasso__alpha':np.arange(.005,.01,1e-5)},folds=5)
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split,cross_validate,cross_val_predict,cross_val_score,validation_curve
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.model_selection import GridSearchCV

X,y = raw.drop('quality',axis=1),raw[['quality']]
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.3,random_state=0)

def optimal_estimator(estimator,X,y,folds=5,paramgrid={}):
    '''
    function which applies GridSearchCV
    estimator    : sklearn linear regularized model
    X,y          : features,response data, ndarrays
    folds        : folds in CV, int
    paramgrid    : key:values to use in GridSearchCV exhaustive search, dict
    '''
    # init GS object
    gscv = GridSearchCV(estimator,cv=folds,param_grid=paramgrid,scoring='neg_mean_squared_error')
    # fit gs obj to data
    gscv.fit(X,y)
    return  gscv.best_estimator_

def linear_estimator(estname,estimator,X,y,degrees=[1,2],seed=0,folds=10,paramgrid={},showcv=True):
    '''
    function which computes cross-validated linear model score
    estname     : name of linear estimator, str
    estimator   : sklearn linear estimator, sklearn linear obj
    X,y         : UCI wine quality features, response, ndarrays
    degrees     : polynomial feature degress arg, list of ints
    seed        : train_test_split random_state arg, int
    folds       : cv k number of folds, int
    paramgrid   : GridSearchCV hyperparameter list, list of flts
    showcv      : displays cv score array,bool
    '''
    # split data 
    Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.25,random_state=seed)
    # init best scores
    degbest,rmsebest,r2best,optimalestimator = 1,np.inf,0,None
    for i,deg in enumerate(degrees):
        print('-'*30+ 'Degree {} {}'.format(deg,estname) +'-'*30)
#         if str(estimator)[:16]!='LinearRegression':
#             print('regularized estimator specified:\n')
            # regular pipeline doesn't work
#             pipe = Pipeline(steps=[('poly',PolynomialFeatures(degree=deg)),
#                                    ('stdscaler',StandardScaler),
#                                    ('linear_est',estimator)])
        pipe = make_pipeline(PolynomialFeatures(degree=deg),
                             SimpleImputer(strategy='median'),
                             StandardScaler(with_std=True),
                             estimator)
        # optimal hyper on pipeline for regularized 
        gscv = GridSearchCV(pipe,cv=folds,param_grid=paramgrid,scoring='neg_mean_squared_error')
        # now fit gscv to data
        gscv.fit(Xtrain,ytrain)
        bestestimator = gscv.best_estimator_
        print('GridSearchCV.best_estimator_:\n{}'.format(bestestimator))
        print('\nGridSearchCV.best_score_:\n{}'.format(gscv.best_params_))
        # now use gscv optimized estimator to data
        cvresult = cross_validate(bestestimator,Xtrain,ytrain,
                                  cv=folds,return_train_score=True,
                                  scoring=('r2','neg_mean_squared_error'))
        # reassign negative scores to nan 
        cv_r2train = np.where(cvresult['train_r2']>=0,cvresult['train_r2'],np.nan)
        cv_r2val = np.where(cvresult['test_r2']>=0,cvresult['test_r2'],np.nan)
        # compute cv rmse
        cv_msetrain = cvresult['train_neg_mean_squared_error']
#         print(cv_msetrain)
        cv_rmsetrain = np.sqrt(-cv_msetrain)
        cv_mseval = cvresult['test_neg_mean_squared_error']
        cv_rmseval = np.sqrt(-cv_mseval) 
        if showcv:
            print('\nDegree {} {}-fold CV train rmse scores:\n{}\n'.format(deg,folds,cv_rmsetrain))
            print('\nDegree {} {}-fold CV train r2 scores:\n{}\n'.format(deg,folds,cv_r2train))
            print('Degree {} {}-fold CV validate r2 scores:\n{}'.format(deg,folds,cv_r2val))
        # compute mean cv r2 scores with np.nanmean to ignore nans in mean computation
        trainr2,valr2 = np.nanmean(cv_r2train),np.nanmean(cv_r2val)
        print('\nDegree {} Train CV Mean R2 = {:,.3f}'.format(deg,trainr2))
        print('Degree {} Validate CV Mean R2 = {:,.3f}'.format(deg,valr2))
        # compute mean cv rmse scores
        trainrmse,valrmse = np.mean(cv_rmsetrain),np.mean(cv_rmseval)
        print('\nDegree {} Train CV Mean RMSE = {:,.3f}'.format(deg,trainrmse))
        print('Degree {} Validate CV Mean RMSE = {:,.3f}'.format(deg,valrmse))
        # compute test yhat as cv'd 
        cv_testpred = cross_val_predict(bestestimator,Xtest,ytest).ravel()
        # prediction fit plot
        f,ax = plt.subplots(2,figsize=(16,16))
        ax[0].scatter(ytest,cv_testpred,edgecolor=(0,0,0))
        x = np.linspace(*ax[0].get_xlim())
        ax[0].plot(x,x,color='k',linestyle='dotted',label='identity')
        ax[0].set(title='Degree {} {} Prediction Fit'.format(deg,estname),
                 xlabel='test y',ylabel='CV yhat')
        ax[0].legend()
        # test residuals
        cv_testresids = (ytest.as_matrix()-cv_testpred.mean().ravel()).ravel()
#         print('cv_testresids.shape={}'.format(cv_testresids.shape))
#         print('cv_testpred.shape={}'.format(cv_testpred.shape))
        cv_testmse = np.square(cv_testresids).mean()
        cv_testrmse = np.sqrt(cv_testmse)
#         cv_testr2 = r2_score(ytest,cv_testpred)
        # cross_val_score instead of r2_score to control for negative scores
        cvscore = cross_val_score(bestestimator,Xtest,ytest,cv=folds,scoring='r2')
        cv_testr2 = np.where(cvscore<0,np.nan,cvscore)
        # np.nanmean instead of .mean() to ignore nan values
        cv_testr2 = np.nanmean(cv_testr2)
#         print('cv_testr2\n{}'.format(cv_testr2))
        # determine best result by degree
        if cv_testrmse<rmsebest:
            degbest,rmsebest,r2best,optimalestimator = deg,cv_testrmse,cv_testr2,bestestimator
        print('\nDegree {} CV test R2 = {}'.format(deg,cv_testr2))
        print('Degree {} CV test RMSE = {}'.format(deg,cv_testrmse))
        sns.regplot(cv_testresids,cv_testpred,ax=ax[1],lowess=False,
                    scatter_kws={'alpha':.8},
                    line_kws={'color':'red'})
        ax[1].set(title='Degree {} {} Residual Plot'.format(deg,estname),
                 xlabel='CV residuals',ylabel='CV yhat')
        ax[1].legend(['lowess'])
    return degbest,rmsebest,r2best,optimalestimator

randomseed=1
cvfolds=5
# narrow estimator hyper parameters
# linear_estimator('linreg',LinearRegression(),X,y,degrees=[1,2],seed=randomseed,folds=cvfolds,paramgrid={'linearregression__fit_intercept':[True,False]},showcv=False)    
# linear_estimator('ridge',Ridge(),X,y,degrees=[1],seed=randomseed,folds=cvfolds,paramgrid={'ridge__alpha':np.arange(65,75,1)})     # alpha=69
# linear_estimator('ridge',Ridge(),X,y,degrees=[2],seed=randomseed,folds=cvfolds,paramgrid={'ridge__alpha':np.arange(0,10,1)})   # alpha=6 
# linear_estimator('ridge',Ridge(),X,y,degrees=[3],seed=randomseed,folds=cvfolds,paramgrid={'ridge__alpha':np.arange(62,75,1)})   # alpha=67 
# linear_estimator('lasso',Lasso(),X,y,degrees=[1],seed=randomseed,folds=cvfolds,paramgrid={'lasso__alpha':np.arange(0.01,.017,1e-4)},showcv=False) # alpha=.0157    
# linear_estimator('lasso',Lasso(),X,y,degrees=[2],seed=randomseed,folds=cvfolds,paramgrid={'lasso__alpha':np.arange(0.0005,.01,1e-4)},showcv=False) # alpha=.0008    
# linear_estimator('lasso',Lasso(),X,y,degrees=[3],seed=randomseed,folds=cvfolds,paramgrid={'lasso__alpha':np.arange(0.002,.0025,1e-4)},showcv=False) # alpha=.0022    
# linear_estimator('knnreg',KNeighborsRegressor(),X,y,degrees=[1],seed=randomseed,folds=cvfolds,paramgrid={'kneighborsregressor__n_neighbors':np.arange(24,32,1)})  # n=28    
# linear_estimator('forestreg',RandomForestRegressor(),X,y,degrees=[1],seed=randomseed,folds=cvfolds,paramgrid={'randomforestregressor__n_estimators':np.arange(29,33,1),'randomforestregressor__min_samples_split':np.arange(3,7,1)}) # 31,4  # n=28    
# linear_estimator('treereg',DecisionTreeRegressor(),X,y,degrees=[1],seed=randomseed,folds=cvfolds,paramgrid={'decisiontreeregressor__max_depth':np.arange(2,7,1)}) # 4  
# linear_estimator('svr',SVR(),X,y,degrees=[1],seed=randomseed,folds=cvfolds,paramgrid={'svr__kernel':['rbf'],'svr__C':np.arange(.69,.72,1e-3)}) # rbf,C=.701    

randomseed=0
cvfolds=5

# bundle linear estimators to run through pipeline
estimatorlist = [
                 ('linreg',linear_estimator('linreg',LinearRegression(),X,y,degrees=[1,2],seed=randomseed,folds=cvfolds,paramgrid={'linearregression__fit_intercept':[True,False]},showcv=False)),    
                 ('ridge',linear_estimator('ridge',Ridge(),X,y,degrees=[1],seed=randomseed,folds=cvfolds,paramgrid={'ridge__alpha':np.arange(65,75,1)})),     # alpha=69
                 ('ridge',linear_estimator('ridge',Ridge(),X,y,degrees=[2],seed=randomseed,folds=cvfolds,paramgrid={'ridge__alpha':np.arange(0,10,1)})),   # alpha=6 
                 ('ridge',linear_estimator('ridge',Ridge(),X,y,degrees=[3],seed=randomseed,folds=cvfolds,paramgrid={'ridge__alpha':np.arange(64,75,1)})),   # alpha=67 
                 ('lasso',linear_estimator('lasso',Lasso(),X,y,degrees=[1],seed=randomseed,folds=cvfolds,paramgrid={'lasso__alpha':np.arange(0.01,.017,1e-4)},showcv=False)), # alpha=.0157    
                 ('lasso',linear_estimator('lasso',Lasso(),X,y,degrees=[2],seed=randomseed,folds=cvfolds,paramgrid={'lasso__alpha':np.arange(0.0005,.01,1e-4)},showcv=False)), # alpha=.0008    
                 ('lasso',linear_estimator('lasso',Lasso(),X,y,degrees=[3],seed=randomseed,folds=cvfolds,paramgrid={'lasso__alpha':np.arange(0.002,.0025,1e-4)},showcv=False)), # alpha=.0022    
                 ('knnreg',linear_estimator('knnreg',KNeighborsRegressor(),X,y,degrees=[1],seed=randomseed,folds=cvfolds,paramgrid={'kneighborsregressor__n_neighbors':np.arange(24,32,1)})),  # n=28    
                 ('forestreg',linear_estimator('forestreg',RandomForestRegressor(),X,y,degrees=[1],seed=randomseed,folds=cvfolds,paramgrid={'randomforestregressor__n_estimators':np.arange(29,33,1),'randomforestregressor__min_samples_split':np.arange(3,7,1)})), # 31,4  # n=28    
                 ('treereg',linear_estimator('treereg',DecisionTreeRegressor(),X,y,degrees=[1],seed=randomseed,folds=cvfolds,paramgrid={'decisiontreeregressor__max_depth':np.arange(2,7,1)})), # 4  
                 ('svr',linear_estimator('svr',SVR(),X,y,degrees=[1],seed=randomseed,folds=cvfolds,paramgrid={'svr__kernel':['rbf'],'svr__C':np.arange(.69,.72,1e-3)})) # rbf,C=.701    
                 ]

resultlist = []
# for i,est in enumerate(estimatorlist):
for tup in estimatorlist:
    resultlist.append((tup[0],tup[1]))
# append PCA reg separately    
resultlist.append(('pcareg',pca_regression(LinearRegression(),X,y,testsplit=.3,seed=randomseed,paramgrid={'linearregression__normalize':[False,True],'linearregression__fit_intercept':[True,False]},folds=cvfolds)))
resultlist.append(('pcaridge',pca_regression(Ridge(),X,y,testsplit=.3,seed=randomseed,paramgrid={'ridge__normalize':[False,True],'ridge__fit_intercept':[True,False],'ridge__alpha':np.arange(1e-1,.24,1e-3)},folds=cvfolds)))
resultsorted = sorted(resultlist,key=lambda x: x[1][1])
resultsorted
    
for i,result in enumerate(resultsorted):
    # index pipeline
    estname = result[0]
    polydegree = result[1][0]
    rmse = result[1][1]
    r2 = result[1][2]
    print('{}.{}\npoly={}\trmse={:,.10f}\tr2={:,.10f}\n'.format(i+1,estname,polydegree,rmse,r2))
