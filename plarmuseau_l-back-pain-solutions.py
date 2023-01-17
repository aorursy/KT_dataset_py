import pandas as pd

import numpy as np

import seaborn as sns



import matplotlib.pyplot as plt

%matplotlib inline



# read data into dataset variable

train = pd.read_csv("../input/Dataset_spine.csv")





# Drop the unnamed column in place (not a copy of the original)#

train.drop('Unnamed: 13', axis=1, inplace=True)

train.columns = ['Pelvic Incidence','Pelvic Tilt','Lumbar Lordosis Angle','Sacral Slope','Pelvic Radius', 'Spondylolisthesis Degree', 'Pelvic Slope', 'Direct Tilt', 'Thoracic Slope', 'Cervical Tilt','Sacrum Angle', 'Scoliosis Slope','Outcome']

# Concatenate the original df with the dummy variables

#data = pd.concat([data, pd.get_dummies(data['Class_att'])], axis=1)



# Drop unnecessary label column in place. 

#data.drop(['Class_att','Normal'], axis=1, inplace=True)

new_col=train.describe().T

new_col['eff']=new_col['std']/new_col['mean']

new_col['eff2']=new_col['eff']*new_col['std']

print(new_col)
# Categorical features

cat_cols = []

for c in train.columns:

    if train[c].dtype == 'object':

        cat_cols.append(c)

print('Categorical columns:', cat_cols)



# Dublicate features

d = {}; done = []

cols = train.columns.values

for c in cols: d[c]=[]

for i in range(len(cols)):

    if i not in done:

        for j in range(i+1, len(cols)):

            if all(train[cols[i]] == train[cols[j]]):

                done.append(j)

                d[cols[i]].append(cols[j])

dub_cols = []

for k in d.keys():

    if len(d[k]) > 0: 

        # print k, d[k]

        dub_cols += d[k]        

print('Dublicates:', dub_cols)



# Constant columns

const_cols = []

for c in cols:

    if len(train[c].unique()) == 1:

        const_cols.append(c)

print('Constant cols:', const_cols)
def add_new_col(x):

    if x not in new_col.keys(): 

        # set n/2 x if is contained in test, but not in train 

        # (n is the number of unique labels in train)

        # or an alternative could be -100 (something out of range [0; n-1]

        return int(len(new_col.keys())/2)

    return new_col[x] # rank of the label



def clust(x):

    kl=0

    if x<0.75:

        kl=1

    if x>0.75 and x<4:

        kl=2

    if x>4:

        kl=4

    return kl



new_col= train[['Lumbar Lordosis Angle','Outcome']].groupby('Outcome').describe().fillna(method='bfill')

new_col.columns=['count','mean','std','min','p25','p50','p75','max']

new_col['eff']=new_col['std']/new_col['mean']

new_col['eff2']=new_col['eff']*new_col['std']

new_col['clust']=new_col['eff2'].map(clust)
print(new_col)

#train=pd.merge(train,new_col, how='inner', left_on='Outcome', right_index=True)

#sns.pairplot(train[['Pelvic Tilt','std','eff2','Outcome']],hue='Outcome')

#plt.show()



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

X = train.drop('Outcome',axis=1) # we only take the first two features.

le = preprocessing.LabelEncoder()

def rmsle(y_predicted, y_real):

    return np.sqrt(np.mean(np.power(np.log1p(y_predicted)-np.log1p(y_real), 2)))

def procenterror(y_predicted, y_real):

     return np.round( np.mean(np.abs(y_predicted-y_real) )/ np.mean(y_real) *100 ,1)



    

le.fit(train['Outcome'])

print(list(le.classes_))

Y=le.transform(train['Outcome'])

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

         'LatentDietrich',    

        ]



classifiers = [

    

    PCA(n_components=n_col),

    FastICA(n_components=n_col),

    GaussianRandomProjection(n_components=3),

    KMeans(n_clusters=24),

    SparsePCA(n_components=n_col),

    SparseRandomProjection(n_components=n_col, dense_output=True),

    Birch(branching_factor=10, n_clusters=12, threshold=0.5),

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

    #print('Ypredict',res.predict(Xr).round()+correct)  #show OLS prediction



    #print('Ypredict *log_sec',res.predict(Xr).round()+correct*Y.mean())  #show OLS prediction

    print(name,'%error',procenterror(res.predict(Xr)+correct*Y.mean(),Y),'rmsle',rmsle(res.predict(Xr)+correct*Y.mean(),Y))
from sklearn.linear_model import OrthogonalMatchingPursuit,RANSACRegressor,LogisticRegression,ElasticNetCV,HuberRegressor, Ridge, Lasso,LassoCV,Lars,BayesianRidge,SGDClassifier,LogisticRegressionCV,RidgeClassifier

from sklearn.preprocessing import MinMaxScaler



# import some data to play with

       # those ? converted to NAN are bothering me abit...        



from sklearn.linear_model import OrthogonalMatchingPursuit,RANSACRegressor,LogisticRegression,ElasticNetCV,HuberRegressor, Ridge, Lasso,LassoCV,Lars,BayesianRidge,SGDClassifier,LogisticRegressionCV,RidgeClassifier

from sklearn.svm import SVC

from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}



X = train.drop('Outcome',axis=1) # we only take the first two features.

le = preprocessing.LabelEncoder()

def rmsle(y_predicted, y_real):

    return np.sqrt(np.mean(np.power(np.log1p(y_predicted)-np.log1p(y_real), 2)))

def procenterror(y_predicted, y_real):

     return np.round( np.mean(np.abs(y_predicted-y_real) )/ np.mean(y_real) *100 ,1)



    

le.fit(train['Outcome'])

print(list(le.classes_))

Y=le.transform(train['Outcome'])

scaler = MinMaxScaler()

scaler.fit(X)

X=scaler.transform(X)

poly = PolynomialFeatures(2)

X=poly.fit_transform(X)





names = [

         'ElasticNet',

         'SVC',

         'kSVC',

         'KNN',

         'DecisionTree',

         'RandomForestClassifier',

         'GridSearchCV',

         'HuberRegressor',

         'Ridge',

         'Lasso',

         'LassoCV',

         'Lars',

         'BayesianRidge',

         'SGDClassifier',

         'RidgeClassifier',

         'LogisticRegression',

         'OrthogonalMatchingPursuit',

         #'RANSACRegressor',

         ]



classifiers = [

    ElasticNetCV(cv=10, random_state=0),

    SVC(),

    SVC(kernel = 'rbf', random_state = 0),

    KNeighborsClassifier(n_neighbors = 1),

    DecisionTreeClassifier(),

    RandomForestClassifier(n_estimators = 200),

    GridSearchCV(SVC(),param_grid, refit = True, verbose = 1),

    HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=100,epsilon=2.95),

    Ridge(fit_intercept=True, alpha=0.0, random_state=0, normalize=True),

    Lasso(alpha=0.05),

    LassoCV(),

    Lars(n_nonzero_coefs=10),

    BayesianRidge(),

    SGDClassifier(),

    RidgeClassifier(),

    LogisticRegression(),

    OrthogonalMatchingPursuit(),

    #RANSACRegressor(),

]

correction= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]



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
from tpot import TPOTClassifier

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X.astype(np.float64),

    Y.astype(np.float64), train_size=0.75, test_size=0.25)



tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)

tpot.fit(X_train, y_train)

print(tpot.score(X_test, y_test))

tpot.export('tpot_iris_pipeline.py')
from sklearn.cross_validation import train_test_split

import xgboost as xgb

dtrain = xgb.DMatrix(X, label=Y)

param = {

    'max_depth': 5,  # the maximum depth of each tree

    'eta': 0.1,  # the training step for each iteration

    'silent': 1,  # logging mode - quiet

    'objective': 'multi:softprob',  # error evaluation for multiclass training

    'num_class': 2}  # the number of classes that exist in this datset

num_round = 700  # the number of training iterations



bst = xgb.train(param, dtrain, num_round)

bst.dump_model('dumptree.raw.txt')

preds = bst.predict(dtrain)



print('% error',sum(  pd.DataFrame(preds.round()*[0,1]).sum(axis=1)

 - Y  ) ) 

print(pd.DataFrame(preds.round()*[0,1]))