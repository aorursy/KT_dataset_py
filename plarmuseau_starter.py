import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

disease = pd.read_csv('../input/U.S._Chronic_Disease_Indicators.csv')

print(disease.head())



#labelize

for c in disease.columns:

    disease[c]=disease[c].fillna(-1)

    if disease[c].dtype == 'object':

        lbl = LabelEncoder()

        lbl.fit(list(disease[c].values))

        disease[c] = lbl.transform(list(disease[c].values))

# Any results you write to the current directory are saved as output.
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

print(todrop_col(disease,tohold))

#print(todrop_col(properties,tohold))
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

from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures

import matplotlib.pyplot as plt



n_col=12

X = disease.drop(['StratificationCategoryID1', 'StratificationCategory3', 'Stratification3', 'StratificationCategoryID3', 'ResponseID', 'StratificationID3', 'Stratification2', 'StratificationCategory2'],axis=1) # we only take the first two features.



def rmsle(y_predicted, y_real):

    return np.sqrt(np.mean(np.power(np.log1p(y_predicted)-np.log1p(y_real), 2)))

def procenterror(y_predicted, y_real):

     return np.round( np.mean(np.abs(y_predicted-y_real) )/ np.mean(y_real) *100 ,1)



    

Y=disease['LocationDesc']

scaler = MinMaxScaler()

scaler.fit(X)

X=scaler.transform(X)

poly = PolynomialFeatures(2)

X=poly.fit_transform(X)







names = [

        # 'PCA',

         'FastICA',

       #  'Gauss',

         'KMeans',

         #'SparsePCA',

         #'SparseRP',

         #'Birch',

         #'NMF',    

         #'LatentDietrich',    

        ]



classifiers = [

    

    #PCA(n_components=n_col),

    FastICA(n_components=n_col),

    #GaussianRandomProjection(n_components=3),

    KMeans(n_clusters=24),

    #SparsePCA(n_components=n_col),

    #SparseRandomProjection(n_components=n_col, dense_output=True),

    #Birch(branching_factor=10, n_clusters=12, threshold=0.5),

    NMF(n_components=n_col),    

    #LatentDirichletAllocation(n_topics=n_col),

    

]

correction= [1,0,0,0,0,0,0,0,0]



temp=zip(names,classifiers,correction)

print(temp)



for name, clf,correct in temp:

    Xr=clf.fit_transform(X,Y)

    dddraw(Xr,name)

    res = sm.OLS(Y,Xr).fit()

    #print(res.summary())  # show OLS regression

    #print(res.predict(Xr).round()+correct)  #show OLS prediction

    #print('Ypredict',res.predict(Xr).round()+correct)  #show OLS prediction



    #print('Ypredict *log_sec',res.predict(Xr).round()+correct*Y.mean())  #show OLS prediction

    print(name,'%error',procenterror(res.predict(Xr)+correct*Y.mean(),Y),'rmsle',rmsle(res.predict(Xr)+correct*Y.mean(),Y))