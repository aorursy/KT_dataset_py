import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from math import sqrt

import matplotlib as plt

import matplotlib.pyplot as plt

import seaborn as sns 

from sklearn.preprocessing import normalize

from pandas import DataFrame

import warnings

warnings.filterwarnings('ignore')

% matplotlib inline

df=pd.read_csv('../input/unit_labour_costs_and_labour_productivity_(employment_based).csv')
#Using the string method contains we create a DF for the quarterly ULCs in Spain

Id1=df['Country'].str.contains('Belgium')

Id2=df['Subject'].str.contains('Unit Labour Costs')

Id3=df['Measure'].str.contains('Index')

BE=df[Id1&Id2&Id3]
# We won't need these columns, so we create a list of columns to drop

drop_col=['LOCATION','Country', 'SUBJECT', 'Subject', 'MEASURE', 'Measure',

       'FREQUENCY', 'Frequency', 'TIME', 'Unit Code', 'Unit',

       'PowerCode Code', 'PowerCode', 'Reference Period Code',

       'Reference Period', 'Flag Codes', 'Flags']

# We drop the columns

BE=BE.drop(drop_col, axis=1)

# We set the column time as our index

BE.set_index('Time',inplace=True)

# We drop our duplicated data

BE=BE[0:28]


# Let's plot the indexed quarterly evolution of Spain from 2010 to 2016



plt.rc('xtick', labelsize=24)                                           

plt.rc('ytick', labelsize=24)

ax=BE.plot(figsize=(20,10), kind='line', legend=False, use_index=True, grid=True, 

                 color='aqua')

plt.axhline(y=100)                          # we create a line for index ref. 2010=100

plt.xlabel('Quarterly evolution 2010-2016', size=22)                # x title label 

plt.ylabel('Unit Labour Costs - index ref. 2010', size=20)          # y title label 

plt.title('BELGIUM Unit Labour Costs ULC 2010-2016',size=26)          # plot title label   



# We create an arrow to indicate the timing of the competivity gain

bbox_props = dict(boxstyle="RArrow,pad=1", fc="cyan", ec="b", lw=2)

t = ax.text(10, 99, "Tax shift ? Less cost of labour ?", ha="center", va="center", rotation=10,

            size=16, bbox=bbox_props)
df1=pd.read_csv('../input/SNA_TABLE1_12082017110055171.csv')
# We use the same filtering method as before with the string function contains

m1=df1['Country'].str.contains('Belgium')

m2=df1['Transaction'].str.contains('Exports of goods and services')

m3=df1['Measure'].str.contains('Current prices')

m4=df1['Unit'].str.contains('Euro')

m5=df1['TRANSACT'].str.contains('P6')

BE_exp=df1[m1&m2&m3&m4&m5]
dcg=['LOCATION', 'Country', 'TRANSACT', 'Transaction', 'MEASURE', 'Measure',

       'TIME', 'Unit Code', 'Unit', 'PowerCode Code', 'PowerCode',

       'Reference Period Code', 'Reference Period', 'Flag Codes',

       'Flags']

BE_exp=BE_exp.drop(dcg, axis=1)

BE_exp.set_index('Year', inplace=True)
BE_exp.plot(figsize=(20,10), kind='line', legend=False, use_index=True, 

                    grid=True, color='aqua')



# we create a line for the 0% (Y=0

be_avg=BE_exp.mean()



plt.axhline(y=be_avg.item(), label='Exports mean 2010-2016')           

plt.xlabel('Years', size=26)                                          

plt.ylabel('Value in current prices - €', size=26)                     

plt.title('Belgian exports 2010-1016',size=30)                                                     

plt.legend(loc='upper left', prop={'size': 20}) 
pdCLU=pd.read_csv('../input/PDBI_I4_19082017122907545.csv')
# We create a list of columns which we won't need

dropCLU=['LOCATION', 'SUBJECT', 'Subject', 'MEASURE', 'Measure',

       'ACTIVITY', 'Activity', 'TIME', 'Unit Code', 'Unit',

       'PowerCode Code', 'PowerCode', 'Reference Period Code',

       'Reference Period', 'Flag Codes', 'Flags']
# We drop the list of superfluous columns

pdCLU.drop(dropCLU, axis=1, inplace=True)

# Now we can pivot the DataFrame  

pdCLU=pdCLU.pivot(index='Time', columns='Country', values='Value').transpose()
# As we have many missing values we will fill them with the closest available value

pdCLU=pdCLU.fillna(method='bfill', axis=1)

pdCLU=pdCLU.fillna(method='ffill', axis=1)



# Let's drop the EU indicators as they don' represent any country

dropIndex=['Euro area (19 countries)','European Union (28 countries)']

pdCLU.drop(dropIndex, axis=0, inplace=True)
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



n_col=3

X = pdCLU

print(pdCLU)



def rmsle(y_predicted, y_real):

    return np.sqrt(np.mean(np.power(np.log1p(y_predicted)-np.log1p(y_real), 2)))

def procenterror(y_predicted, y_real):

     return np.round( np.mean(np.abs(y_predicted-y_real) )/ np.mean(y_real) *100 ,1)



Y=pdCLU[2000]/pdCLU[2015]

X=X.fillna(value=0)       # those ? converted to NAN are bothering me abit...        

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

         #'SparsePCA',

         #'SparseRP',

         'Birch',

         'NMF',    

         'LatentDietrich',    

        ]



classifiers = [

    

    PCA(n_components=n_col),

    FastICA(n_components=n_col),

    GaussianRandomProjection(n_components=3),

    KMeans(n_clusters=24),

    #SparsePCA(n_components=n_col),

    #SparseRandomProjection(n_components=n_col, dense_output=True),

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



n_col=20

X = pdCLU

Y=np.round(pdCLU[2000]/pdCLU[2015]*10)

X=X.fillna(value=0)

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