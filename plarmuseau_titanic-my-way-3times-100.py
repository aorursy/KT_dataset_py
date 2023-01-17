import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

sns.set(style='whitegrid', context='notebook', palette='Set2')

sns.despine()

# get train & test csv files as a DataFrame

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

print("train matrix shape:",train.shape)

train['split']=0

test['split']=1

# combine train and test set

traint = train.append(test)

traint = traint.set_index('PassengerId')

print("test matrix shape:",test.shape)

print("combined in train matrix shape:",traint.shape)

#Data cleanup

#Cabin First letter

traint['CabinLet']=traint['Cabin'].str[:1]

traint['AgeDec']=traint['Age']/10

traint['AgeDec']=traint['AgeDec'].round(0)

traint['FareDec']=np.log(traint['Fare']+1)

traint['FareDec']=traint['FareDec'].round(0)

traint=traint.sort_values(['Fare'])

traint['CabinLet']=traint['CabinLet'].fillna(method='bfill')

traint['AgeDec']=traint['AgeDec'].fillna(method='bfill')

print(traint)
from collections import Counter



def detect_outliers(df,n,features):

    # Categorical features

    cat_cols = []

    for c in df.columns:

        if df[c].dtype == 'object':

            cat_cols.append(c)

    print('Categorical columns:', cat_cols)



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

    print('Dublicates:', dub_cols)



    

    # Constant columns

    const_cols = []

    for c in cols:

        if len(df[c].unique()) == 1:

            const_cols.append(c)

    print('Constant cols:', const_cols)



    # Outlier detection     

    """

    Takes a dataframe df of features and returns a list of the indices

    corresponding to the observations containing more than n outliers according

    to the Tukey method.

    """

    outlier_indices = []

    

    # iterate over features(columns)

    for col in features:

        # 1st quartile (25%)

        Q1 = np.percentile(df[col], 25)

        # 3rd quartile (75%)

        Q3 = np.percentile(df[col],75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        

        # outlier step

        outlier_step = 1.5 * IQR

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

        

    # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   



# detect outliers from Age, SibSp , Parch and Fare

Outliers_to_drop = detect_outliers(traint,2,['Age','SibSp','Parch','Fare','FareDec','AgeDec'])

traint.loc[Outliers_to_drop] # Show the outliers rows

const_cols=[]

dub_cols=[]
if traint['Name'].isnull().values.sum()==0:

    traint['tekst']=traint['Name']

if traint['Ticket'].isnull().values.sum()==0:

    traint['tekst']=traint['tekst']+' T'+traint['Ticket']

if traint['Sex'].isnull().values.sum()==0:

    traint['tekst']=traint['tekst']+' '+traint['Sex']

if traint['Embarked'].isnull().values.sum()==0:

    traint['tekst']=traint['tekst']+' '+traint['Embarked']

else:

    print('empty Embarked ',traint['Embarked'].isnull().values.sum())

    traint['Embarked']=traint['Embarked'].fillna(value='leeg')

    print('afterfilled Embarked ',traint['Embarked'].isnull().values.sum())    

    traint['tekst']=' Emb'+traint.Embarked+' '+traint['tekst']

if traint['Cabin'].isnull().values.sum()==0:

    traint['tekst']=traint['tekst']+' '+traint['Cabin']

else:

    print('empty Cabin ',traint['Cabin'].isnull().values.sum())

    traint['Cabin']=traint['Cabin'].fillna(value='leeg')

    print('afterfilled Cabin ',traint['Cabin'].isnull().values.sum())    

    traint['tekst']=' Cab '+traint.Cabin+' '+traint['tekst']

print('empty tekst ',traint['tekst'].isnull().values.sum())    

print(traint.tekst)
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.preprocessing import Normalizer

from sklearn.pipeline import make_pipeline



vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)

N_train = vectorizer.fit_transform(traint.tekst)



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

    traint['txt_' + str(i)] = N_train[:,i - 1]
from sklearn.preprocessing import LabelEncoder

   

for c in ['Sex', 'CabinLet']:

        lbl = LabelEncoder()

        lbl.fit(list(traint[c].values))

        traint[c] = lbl.transform(list(traint[c].values))

        

# Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived 

g = sns.heatmap(traint[["Survived","CabinLet",'AgeDec',"FareDec","Parch","Pclass",'Sex','SibSp','txt_1']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
def add_new_col(x):

    if x not in new_col.keys(): 

        # set n/2 x if is contained in test, but not in train 

        # (n is the number of unique labels in train)

        # or an alternative could be -100 (something out of range [0; n-1]

        return int(len(new_col.keys())/2)

    return new_col[x] # rank of the label







def clust(x):

    kl=0

    if x<0.4:

        kl=1

    if x>0.4 and x<.7:

        kl=2

    if x>.70:

        kl=4

    return kl





new_col4= traint[['Survived','AgeDec']].groupby('AgeDec').describe().fillna(method='bfill')

new_col4.columns=['count','mean','std','min','p25','p50','p75','max']

new_col4['eff']=new_col4['std']/new_col4['mean']

new_col4['eff2']=new_col4['eff']*new_col4['std']

new_col4['clust']=new_col4['eff2'].map(clust)

print(new_col4)



new_col3= traint[['Survived','Pclass']].groupby('Pclass').describe().fillna(method='bfill')

new_col3.columns=['count','mean','std','min','p25','p50','p75','max']

new_col3['eff']=new_col3['std']/new_col3['mean']

new_col3['eff2']=new_col3['eff']*new_col3['std']

new_col3['clust']=new_col3['eff2'].map(clust)

print(new_col3)



new_col2= traint[['Survived','Sex']].groupby('Sex').describe().fillna(method='bfill')

new_col2.columns=['count','mean','std','min','p25','p50','p75','max']

new_col2['eff']=new_col2['std']/new_col2['mean']

new_col2['eff2']=new_col2['eff']*new_col2['std']

new_col2['clust']=new_col2['eff2'].map(clust)

print(new_col2)





new_col= traint[['Survived','CabinLet']].groupby('CabinLet').describe().fillna(method='bfill')

new_col.columns=['count','mean','std','min','p25','p50','p75','max']

new_col['eff']=new_col['std']/new_col['mean']

new_col['eff2']=new_col['eff']*new_col['std']

new_col['clust']=new_col['eff2'].map(clust)

print(new_col)



new_col5= traint[['Survived','FareDec']].groupby('FareDec').describe().fillna(method='bfill')

new_col5.columns=['count','mean','std','min','p25','p50','p75','max']

new_col5['eff']=new_col5['std']/new_col5['mean']

new_col5['eff2']=new_col5['eff']*new_col5['std']

new_col5['clust']=new_col5['eff2'].map(clust)

print(new_col5)
traint=pd.merge(traint,new_col, how='outer', left_on='CabinLet',suffixes=('', '_c'), right_index=True)

traint=pd.merge(traint,new_col2, how='outer', left_on='Sex',suffixes=('', '_s'), right_index=True)

traint=pd.merge(traint,new_col3, how='outer', left_on='Pclass',suffixes=('', '_p'), right_index=True)

traint=pd.merge(traint,new_col4, how='outer', left_on='AgeDec',suffixes=('', '_a'), right_index=True)

traint=pd.merge(traint,new_col5, how='outer', left_on='FareDec',suffixes=('', '_f'), right_index=True)

#append the Cabinet Letter survival stat
Outliers_to_drop = detect_outliers(traint,2,['Age','SibSp','Parch','Fare','FareDec','AgeDec'])

traint.loc[Outliers_to_drop] # Show the outliers rows

dup_cols=['p25_s', 'min_p', 'p25_p', 'p75_s', 'max_p', 'p25_a', 'max_a']

const_cols=['min_s', 'p25_s', 'max_s', 'min_p', 'p25_p', 'max_p']
df_new = traint.drop(['tekst','Cabin', 'Embarked', 'Name', 'Sex', 'Ticket', 'CabinLet','p25_s', 'min_p','p25_p', 'p75_s', 'max_p', 'p25_a', 'max_a','min_s', 'p25_s', 'max_s', 'min_p', 'p25_p', 'max_p'], axis=1)

Outliers_to_drop = detect_outliers(df_new,2,['Age','SibSp','Parch','Fare','FareDec','AgeDec'])

#df_new=df_new.sort_values('PassengerId')

#df_new=df_new[df_new['split']==0]  #take the train set again apart

print(df_new.describe().T)

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
from sklearn.linear_model import OrthogonalMatchingPursuit,RANSACRegressor,LogisticRegression,ElasticNetCV,HuberRegressor, Ridge, Lasso,LassoCV,Lars,BayesianRidge,SGDClassifier,LogisticRegressionCV,RidgeClassifier

from sklearn.svm import SVC

from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}



    

# import some data to play with

X = df_new[df_new['split']==0]

X = X.drop(['Survived'],axis=1)

def rmsle(y_predicted, y_real):

    return np.sqrt(np.mean(np.power(np.log1p(y_predicted)-np.log1p(y_real), 2)))

def procenterror(y_predicted, y_real):

     return np.round( np.mean(np.abs(y_predicted-y_real) )/ np.mean(y_real) *100 ,1)



Y=df_new[df_new['split']==0]

Y=Y['Survived']



X=X.replace([np.inf, -np.inf], np.nan).fillna(value=0)

#print(X) #nasty NaN

scaler = MinMaxScaler()

scaler.fit(X)

X=scaler.transform(X)

poly = PolynomialFeatures(2)

X=poly.fit_transform(X)





names = [

         #'ElasticNet',

         'SVC',

         'kSVC',

         'KNN',

         'DecisionTree',

         'RandomForestClassifier',

         #'GridSearchCV',

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

    SVC(),

    SVC(kernel = 'rbf', random_state = 0),

    KNeighborsClassifier(n_neighbors = 1),

    DecisionTreeClassifier(),

    RandomForestClassifier(n_estimators = 200),

    #GridSearchCV(SVC(),param_grid, refit = True, verbose = 1),

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