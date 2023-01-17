# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

camera=pd.read_csv('../input/camera_dataset.csv')

koloms=camera.columns

camera=pd.read_csv('../input/camera_dataset.csv',skiprows=[0])  #get rid of first row

camera.columns=koloms

print(camera.describe().T)

print(camera.head())
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=7, random_state=0).fit(camera.drop(['Model','Max resolution','Release date'],axis=1).fillna(value=0))

camera['group']=kmeans.labels_

print(camera)
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
def clust(x):

    kl=0

    if x<0.3:

        kl=1

    if x>0.29 and x<0.6:

        kl=2

    if x>0.59:

        kl=4

    return kl

camera.Price=np.log(camera.Price)

new_col= camera[['Release date','Max resolution','Price']].groupby('Release date').describe().fillna(method='bfill')

new_col.columns=['countt','meant','stdt','mint','p25t','p50t','p75t','maxt','countgt','meangt','stdgt','mingt','p25gt','p50gt','p75gt','maxgt']

new_col['efft']=new_col['stdt']/new_col['meant']

new_col['eff2t']=new_col['efft']*new_col['stdt']

new_col['clustt']=new_col['eff2t'].map(clust)

print(new_col.head())

camera=pd.merge(camera,new_col, how='outer', left_on='Release date',suffixes=('', '_c'), right_index=True)
# price (gt)  pixel( t) evolution its not exaclty the law of Moore, but its a 6x more resolution in 10years

new_col[['p25t','p25gt','p50t','p50gt','p75t','p75gt']].plot()

new_col[['maxt','maxgt']].plot(x='maxt',y='maxgt',kind='scatter')

new_col[['p50t','p50gt']].plot(x='p50t',y='p50gt',kind='scatter')
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.preprocessing import Normalizer

from sklearn.pipeline import make_pipeline



vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)

N_train = vectorizer.fit_transform(camera.Model)



n_comp=20         #variable to install



if True:

    print("Performing dimensionality reduction using LSA")

    svd = TruncatedSVD(n_components=n_comp)

    normalizer = Normalizer(copy=False)

    lsa = make_pipeline(svd, normalizer)

    N_train = lsa.fit_transform(N_train)



    explained_variance = svd.explained_variance_ratio_.sum()

    print("Explained variance of the SVD step: {}%".format(

        int(explained_variance * 100)))



    #print(N_train)

#N_train=pd.DataFrame(N_train)

#Append decomposition components to datasets  # to do in next part

for i in range(1, n_comp + 1):

    camera['txt_' + str(i)] = N_train[:,i - 1]
from sklearn.linear_model import OrthogonalMatchingPursuit,RANSACRegressor,LogisticRegression,ElasticNetCV,HuberRegressor, Ridge, Lasso,LassoCV,Lars,BayesianRidge,SGDClassifier,LogisticRegressionCV,RidgeClassifier



from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures



# import some data to play with

X = camera.drop(['Model','Price'],axis=1)

def rmsle(y_predicted, y_real):

    return np.sqrt(np.mean(np.power(np.log1p(y_predicted)-np.log1p(y_real), 2)))

def procenterror(y_predicted, y_real):

     return np.round( np.mean(np.abs(y_predicted-y_real) )/ np.mean(y_real) *100 ,1)



    

Y=np.round(camera['Price']*10)

X=X.fillna(value=0)  #nasty NaN

scaler = MinMaxScaler()

scaler.fit(X)

X=scaler.transform(X)

poly = PolynomialFeatures(2)

X=poly.fit_transform(X)





names = [

         #'ElasticNet',

         'HuberRegressor',

         'Ridge',

         'Lasso',

         'LassoCV',

         'Lars',

         #'BayesianRidge',

         'SGDClassifier',

         'RidgeClassifier',

         'LogisticRegression',

         'OrthogonalMatchingPursuit',

         #'RANSACRegressor',

         ]



classifiers = [

    #ElasticNetCV(cv=10, random_state=0),

    HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=100,epsilon=2.95),

    Ridge(fit_intercept=True, alpha=0.0, random_state=0, normalize=True),

    Lasso(alpha=0.05),

    LassoCV(),

    Lars(n_nonzero_coefs=10),

    #BayesianRidge(),

    SGDClassifier(),

    RidgeClassifier(),

    LogisticRegression(),

    OrthogonalMatchingPursuit(),

    #RANSACRegressor(),

]

correction= [0,0,0,0,0,0,0,0,0,0,0,0]



temp=zip(names,classifiers,correction)

print(temp)



for name, clf,correct in temp:

    regr=clf.fit(X,Y)

    #print( name,'% errors', abs(regr.predict(X)+correct-Y).sum()/(Y.sum())*100)

    print(name,'%error',procenterror(regr.predict(X),Y),'rmsle',rmsle(regr.predict(X),Y))
from sklearn.decomposition import PCA, FastICA,SparsePCA,NMF, LatentDirichletAllocation,FactorAnalysis

from sklearn.random_projection import GaussianRandomProjection,SparseRandomProjection

from sklearn.cluster import KMeans,Birch

import statsmodels.formula.api as sm

from scipy import linalg

from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures

def rmsle(y_predicted, y_real):

    return np.sqrt(np.mean(np.power(np.log1p(y_predicted)-np.log1p(y_real), 2)))

def procenterror(y_predicted, y_real):

     return ( np.mean(np.abs(y_predicted-y_real) )/ np.mean(y_real) *100 ).round()



n_col=7

X = camera.drop(['Model','Price'],axis=1)

def rmsle(y_predicted, y_real):

    return np.sqrt(np.mean(np.power(np.log1p(y_predicted)-np.log1p(y_real), 2)))

def procenterror(y_predicted, y_real):

     return np.round( np.mean(np.abs(y_predicted-y_real) )/ np.mean(y_real) *100 ,1)



    

Y=(camera['Price']*10)

X=X.fillna(value=0)  #nasty NaN

scaler = MinMaxScaler()

scaler.fit(X)

X=scaler.transform(X)

poly = PolynomialFeatures(2)

X=poly.fit_transform(X)





names = [

         'PCA',

         'FastICA',

         'Gauss',

         'KMeans',

         'SparsePCA',

         'SparseRP',

         'Birch',

         'NMF',    

       #  'LatentDietrich',    

        ]



classifiers = [

    

    PCA(n_components=n_col),

    FastICA(n_components=n_col),

    GaussianRandomProjection(n_components=3),

    KMeans(n_clusters=n_col),

    #SparsePCA(n_components=n_col),

    SparseRandomProjection(n_components=n_col, dense_output=True),

    Birch(branching_factor=10, n_clusters=3, threshold=0.5),

    NMF(n_components=n_col),    

  #  LatentDirichletAllocation(n_topics=n_col),

    

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

    #print('Ypredict',res.predict(Xr).round()+correct)  #show OLS prediction

    

    print('Ypredict',res.predict(Xr).round()+correct*Y.mean())  #show OLS prediction

    print(name,'%error',procenterror(res.predict(Xr)+correct*Y.mean(),Y),'rmsle',rmsle(res.predict(Xr)+correct*Y.mean(),Y)) #

    

    

    

    