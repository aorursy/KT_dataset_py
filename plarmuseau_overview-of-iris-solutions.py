import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import datasets





def dddraw(X_reduced,name):

    import matplotlib.pyplot as plt

    from mpl_toolkits.mplot3d import Axes3D

    # To getter a better understanding of interaction of the dimensions

    # plot the first three PCA dimensions

    fig = plt.figure(1, figsize=(8, 6))

    ax = Axes3D(fig, elev=-150, azim=110)

    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y,cmap=plt.cm.Paired)

    titel="First three directions of "+name 

    ax.set_title(titel)

    ax.set_xlabel("1st eigenvector")

    ax.w_xaxis.set_ticklabels([])

    ax.set_ylabel("2nd eigenvector")

    ax.w_yaxis.set_ticklabels([])

    ax.set_zlabel("3rd eigenvector")

    ax.w_zaxis.set_ticklabels([])



    plt.show()

    

    



from sklearn.decomposition import PCA, FastICA,SparsePCA,NMF, LatentDirichletAllocation,FactorAnalysis

from sklearn.random_projection import GaussianRandomProjection,SparseRandomProjection

from sklearn.cluster import KMeans,Birch

import statsmodels.formula.api as sm

from scipy import linalg



iris = datasets.load_iris()

n_col=4

X = iris.data[:, :n_col]  # we only take the first two features.

Y = iris.target





names = [

         'PCA',

         'FastICA',

         'Gauss',

         'KMeans',

         'SparsePCA',

         'SparseRP',

         'Birch',

         'NMF',    

         'LatentDietrich',    

        ]



classifiers = [

    

    PCA(n_components=n_col),

    FastICA(n_components=n_col),

    GaussianRandomProjection(n_components=3),

    KMeans(n_clusters=3),

    SparsePCA(n_components=n_col),

    SparseRandomProjection(n_components=n_col, dense_output=True),

    Birch(branching_factor=20, n_clusters=300, threshold=0.5),

    NMF(n_components=n_col),    

    LatentDirichletAllocation(n_topics=n_col),

    

]

correction= [1,1,0,0,0,0,0,0,0]



temp=zip(names,classifiers,correction)

print(temp)



for name, clf,correct in temp:

    Xr=clf.fit_transform(X,Y)

    dddraw(Xr,name)

    res = sm.OLS(Y,Xr).fit()

    #print(res.summary())  # show OLS regression

    #print(res.predict(Xr).round()+correct)  #show OLS prediction

    print( name,'% errors', abs(res.predict(Xr).round()+correct-Y).sum()/len(Y)*100)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





iris_df=pd.DataFrame(iris.data)



iris_df['S1']=iris_df[0]*iris_df[1]

iris_df['S2']=iris_df[2]*iris_df[3]

iris_df['St']=iris_df['S1']*3+iris_df['S2']*3  # surface of flower

iris_df['S3']=iris_df[0]/iris_df[2]

iris_df['S4']=iris_df[1]/iris_df[3]

iris_df['Su']=iris_df['S3']+iris_df['S4']

iris_df['Ss']=iris_df[0]*iris_df[2]*3.14

iris_df['Scover']=iris_df['St']/iris_df['Ss']   # % surface of flower covering the outer diameter

iris_df['Sratio']=iris_df['S1']/iris_df['S2']

print(iris_df.sample(5))

from sklearn.decomposition import PCA, FastICA,SparsePCA

from sklearn.random_projection import GaussianRandomProjection

from sklearn.random_projection import SparseRandomProjection

from sklearn.cluster import KMeans

import statsmodels.formula.api as sm

from scipy import linalg



# import some data to play withk

iris = datasets.load_iris()

n_col=13

X = iris_df  # we only take the first two features.

Y = iris.target





names = [

         'PCA',

         'FastICA',

         'Gauss',

         'KMeans',

         'SparsePCA',

         'SparseRP',

         'Birch',

         'NMF',    

         'LatentDietrich',  ]  



classifiers = [

    

    PCA(n_components=n_col),

    FastICA(n_components=n_col),

    GaussianRandomProjection(n_components=10),

    KMeans(n_clusters=10),

    SparsePCA(n_components=n_col),

    SparseRandomProjection(n_components=n_col, dense_output=True),

    Birch(branching_factor=20, n_clusters=300, threshold=0.5),

    NMF(n_components=n_col),    

    LatentDirichletAllocation(n_topics=n_col),

    

]

correction= [1,1,0,0,0,0,0,0,]



temp=zip(names,classifiers,correction)

print(temp)



for name, clf,correct in temp:

    Xr=clf.fit_transform(X,Y)

    dddraw(Xr,name)

    res = sm.OLS(Y,Xr).fit()

    print( name,'% errors', abs(res.predict(Xr).round()+correct-Y).sum()/len(Y)*100)

from sklearn.linear_model import LogisticRegression,ElasticNetCV,HuberRegressor, Ridge, Lasso,LassoCV,Lars,BayesianRidge,SGDClassifier,LogisticRegressionCV,RidgeClassifier





# import some data to play with

iris = datasets.load_iris()

n_col=13

X = iris_df  # we only take the first two features.

Y = iris.target





names = [

         'ElasticNet',

         'HuberRegressor',

         'LogisticRegression',

         'LogisticRegressionCV',

         'Ridge',

         'Lasso',

         'LassoCV',

         'Lars',

         'BayesianRidge',

         'SGDClassifier',

         'RidgeClassifier'

         ]



classifiers = [

    ElasticNetCV(cv=5, random_state=0),

    HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=100,epsilon=2.95),

    LogisticRegression(),

    LogisticRegressionCV(),

    Ridge(fit_intercept=True, alpha=0.0, random_state=0, normalize=True),

    Lasso(alpha=0.01),

    LassoCV(),

    Lars(n_nonzero_coefs=6),

    BayesianRidge(),

    SGDClassifier(),

    RidgeClassifier()



]

correction= [0,0,0,0,0,0,0,0,0,0,0,0]



temp=zip(names,classifiers,correction)

print(temp)



for name, clf,correct in temp:

    regr=clf.fit(X,Y)

    print( name,'% errors', abs(regr.predict(X).round()+correct-Y).sum()/len(Y)*100)

from sklearn.cross_validation import train_test_split

import xgboost as xgb

dtrain = xgb.DMatrix(X, label=Y)

param = {

    'max_depth': 3,  # the maximum depth of each tree

    'eta': 0.3,  # the training step for each iteration

    'silent': 1,  # logging mode - quiet

    'objective': 'multi:softprob',  # error evaluation for multiclass training

    'num_class': 3}  # the number of classes that exist in this datset

num_round = 20  # the number of training iterations



bst = xgb.train(param, dtrain, num_round)

bst.dump_model('dumptree.raw.txt')

preds = bst.predict(dtrain)



print('% error',sum(  pd.DataFrame(preds.round()*[0,1,2]).sum(axis=1)

 - Y  ) ) 
