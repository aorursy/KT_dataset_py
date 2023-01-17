import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

from sklearn.preprocessing import LabelEncoder



properties = pd.read_csv("../input/properties_2016.csv")



for c in properties.columns:

    properties[c]=properties[c].fillna(-1)

    if properties[c].dtype == 'object':

        lbl = LabelEncoder()

        lbl.fit(list(properties[c].values))

        properties[c] = lbl.transform(list(properties[c].values))

        

properties.describe().T
train = pd.read_csv("../input/train_2016_v2.csv")

import matplotlib.pyplot as plt

plt.hist(train['logerror'],200)

plt.title("Histogram of logerror")

plt.xlabel("Value")

plt.ylabel("Frequency")

plt.show()

train=pd.merge(train, properties, on='parcelid', how='left')
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
from collections import Counter

def todrop_col(df,tohold):

    # use todrop_col(dataframe,['listtohold'])

    # Categorical features

    df.replace([np.inf, -np.inf], np.nan).fillna(value=-1)

    

    cat_cols = []

    for c in df.columns:

        if df[c].dtype == 'object':

            cat_cols.append(c)

    #print('Categorical columns:', cat_cols)

    

    

    # Constant columns

    cols = df.columns.values    

    const_cols = []

    for c in cols:   

        if len(df[c].unique()) == 1:

            const_cols.append(c)

    #print('Constant cols:', const_cols)

    

    

    # Dublicate features

    d = {}; done = []

    cols = df.columns.values

    for c in cols:

        d[c]=[]

    for i in range(len(cols)):

        if i not in done:

            for j in range(i+1, len(cols)):

                if all(df[cols[i]] == df[cols[j]]):

                    done.append(j)

                    d[cols[i]].append(cols[j])

    dub_cols = []

    for k in d.keys():

        if len(d[k]) > 0: 

            # print k, d[k]

            dub_cols += d[k]        

    #print('Dublicates:', dub_cols)

    

    kolom=list(set(dub_cols+const_cols+cat_cols))

    kolom=[k for k in kolom if k not in tohold]

    

    return kolom



tohold=[]

print(todrop_col(train,tohold))

#print(todrop_col(properties,tohold))
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



n_col=50

X=train.drop(['logerror','assessmentyear', 'transactiondate'],axis=1)

def rmsle(y_predicted, y_real):

    return np.sqrt(np.mean(np.power(np.log1p(y_predicted)-np.log1p(y_real), 2)))

def procenterror(y_predicted, y_real):

     return np.round( np.mean(np.abs(y_predicted-y_real) )/ np.mean(y_real) *100 ,1)



    

Y=train['logerror'].fillna(value=0)

X=X.fillna(value=0)  #nasty NaN

scaler = MinMaxScaler()

scaler.fit(X)

X=scaler.transform(X)

#poly = PolynomialFeatures(2)

#X=poly.fit_transform(X)





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

    KMeans(n_clusters=n_col),

    SparsePCA(n_components=n_col),

    SparseRandomProjection(n_components=n_col, dense_output=True),

    Birch(branching_factor=10, n_clusters=7, threshold=0.5),

    NMF(n_components=n_col),    

    #LatentDirichletAllocation(n_topics=n_col),

    

]

correction= [1,1,0,0,0,0,0,0,0]



temp=zip(names,classifiers,correction)

print(temp)



for name, clf,correct in temp:

    Xr=clf.fit_transform(X,Y)

    dddraw(Xr,name)

    res = sm.OLS(Y,Xr).fit()

    print(res.summary())  # show OLS regression

    #print(res.predict(Xr).round()+correct)  #show OLS prediction

    #print('Ypredict',res.predict(Xr).round()+correct)  #show OLS prediction

    

    print('Ypredict',res.predict(Xr).round()+correct*Y.mean())  #show OLS prediction

    print(name,'%error',procenterror(res.predict(Xr)+correct*Y.mean(),Y),'rmsle',rmsle(res.predict(Xr)+correct*Y.mean(),Y)) #

    

    

    
from sklearn.linear_model import OrthogonalMatchingPursuit,RANSACRegressor,LogisticRegression,ElasticNetCV,HuberRegressor, Ridge, Lasso,LassoCV,Lars,BayesianRidge,SGDClassifier,LogisticRegressionCV,RidgeClassifier

from sklearn.svm import SVC

from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}



X=train.drop(['fireplaceflag', 'taxdelinquencyflag', 'propertycountylandusecode', 'propertyzoningdesc', 'assessmentyear', 'transactiondate', 'hashottuborspa'],axis=1)

Y=np.round(train['logerror'].fillna(value=0)*2)



X=X.replace([np.inf, -np.inf], np.nan).fillna(value=0)

scaler = MinMaxScaler()

scaler.fit(X)

X=scaler.transform(X)

#poly = PolynomialFeatures(2)

#X=poly.fit_transform(X)





names = [

         #677% 'ElasticNet',

         #timeout 'SVC',

         #98.61% 'kSVC',

         'KNN',

         'DecisionTree',

         #'RandomForestClassifier',

         #'GridSearchCV',

         #400%error 'HuberRegressor',

         #683% 'Ridge',

         #511% 'Lasso',

         #683% 'LassoCV',

         #681% 'Lars',

         'BayesianRidge',

         #415% 'SGDClassifier',

         #410%'RidgeClassifier',

         #406% 'LogisticRegression',

         #684% 'OrthogonalMatchingPursuit',

         #'RANSACRegressor',

         ]



classifiers = [

    #ElasticNetCV(cv=10, random_state=0),

    #SVC(),

    #SVC(kernel = 'rbf', random_state = 0),

    KNeighborsClassifier(n_neighbors = 1),

    DecisionTreeClassifier(),

    #RandomForestClassifier(n_estimators = 200),

    #GridSearchCV(SVC(),param_grid, refit = True, verbose = 1),

    #error HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=100,epsilon=2.95),

    #Ridge(fit_intercept=True, alpha=0.0, random_state=0, normalize=True),

    #Lasso(alpha=0.05),

    #LassoCV(),

    #Lars(n_nonzero_coefs=10),

    BayesianRidge(),

    #SGDClassifier(),

    #RidgeClassifier(),

    #LogisticRegression(),

    # OrthogonalMatchingPursuit(),

    #RANSACRegressor(),

]

correction= [0,0,0,0,0,0,0,0,0,0,0,0]



temp=zip(names,classifiers,correction)

print(temp)



for name, clf,correct in temp:

    regr=clf.fit(X,Y)

    #print( name,'% errors', abs(regr.predict(X)+correct-Y).sum()/(Y.sum())*100)

    print(name,'%error',procenterror(regr.predict(X),Y),'rmsle',rmsle(regr.predict(X),Y))

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score, precision_score, recall_score



    # Confusion Matrix

    print(name,'Confusion Matrix')

    print(confusion_matrix(Y, np.round(regr.predict(X) ) ) )

    print('--'*40)



    # Classification Report

    print('Classification Report')

    print(classification_report(Y,np.round( regr.predict(X) ) ))



    # Accuracy

    print('--'*40)

    logreg_accuracy = round(accuracy_score(Y, np.round( regr.predict(X) ) ) * 100,2)

    print('Accuracy', logreg_accuracy,'%')