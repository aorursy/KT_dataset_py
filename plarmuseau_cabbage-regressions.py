# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('../input/kaggle18011884/train_cabbage_price.csv')

train.avgPrice.plot()
train
test=pd.read_csv('../input/kaggle18011884/test_cabbage_price.csv')

test
train['yeargb']=[y[:6] for y in train['year'].astype('str')]

test['yeargb']=test['year']

train['month']=[np.int(y[4:6]) for y in train['year'].astype('str')]

test['month']=[np.int(y[-2:]) for y in test['year'].astype('str')]

train['year2']=[np.int(y[:4]) for y in train['year'].astype('str')]

test['year2']=[np.int(y[:4]) for y in test['year'].astype('str')]

traingb=train.groupby('yeargb').mean().reset_index()

import seaborn as sns

sns.heatmap(traingb.corr())
traingb.reset_index().append(test.reset_index())
mmean=traingb.drop('year',axis=1).groupby('month').mean()

mstd=traingb.drop('year',axis=1).groupby('month').std()

(mmean+mstd).plot()

(mmean-mstd).plot()
sns.heatmap(traingb.corr())
ymean=traingb.groupby('year2').mean()

ystd=traingb.groupby('year2').std()

ymean.avgPrice.plot()
pd.DataFrame(np.dot( np.array(ymean.avgPrice).reshape(-1,1)  , np.array(mmean.avgPrice).reshape(1,-1)) )/(traingb.avgPrice.mean()**2)
def clustertechniques2(dtrain,label,indexv):

    print('#encodings',dtrain.shape)

    cols=[ci for ci in dtrain.columns if ci not in [indexv,'index',label]]

    dtest=dtrain[dtrain[label].isnull()==True][[indexv,label]]

    print(dtest)



    print('encodings  after shape',dtrain.shape)

    #split data or use splitted data

    X_train=dtrain[dtrain[label].isnull()==False].drop([indexv,label],axis=1).fillna(0)

    Y_train=dtrain[dtrain[label].isnull()==False][label]

    X_test=dtrain[dtrain[label].isnull()==True].drop([indexv,label],axis=1).fillna(0)

    Y_test=dtrain[dtrain[label].isnull()==True][label].fillna(0)

    print(Y_test)

    for xi in range(len(Y_test)):

        Y_test.iloc[xi]=np.random.random((1,1))[0]

    print(Y_test)

    if len(X_test)==0:

        from sklearn.model_selection import train_test_split

        X_train,X_test,Y_train,Y_test = train_test_split(dtrain.drop(label,axis=1).fillna(0),dtrain[label],test_size=0.25,random_state=0)

    lenxtr=len(X_train)



    print('splitting data train test X-y',X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)

   





    import matplotlib.pyplot as plt 

    from sklearn import preprocessing

    scale = preprocessing.MinMaxScaler().fit(X_train)

    X_train = scale.transform(X_train)

    X_test = scale.transform(X_test)

        

    from sklearn.decomposition import PCA,TruncatedSVD,NMF,FastICA

    from umap import UMAP  # knn lookalike of tSNE but faster, so scales up

    from sklearn.manifold import TSNE #limit number of records to 100000



    clusters = [Dummy(1),

                PCA(n_components=0.7,random_state=0,whiten=True),

                FastICA(n_components=7,random_state=0),

                TruncatedSVD(n_components=5, n_iter=7, random_state=42),

                NMF(n_components=10,random_state=0),            

                UMAP(n_neighbors=5,n_components=10, min_dist=0.3,metric='minkowski'),

                TSNE(n_components=2,random_state=0)

                ] 

    clunaam=["raw",'PCA','tSVD','UMAP','tSNE']#,'ICA','tSVD','nmf','UMAP','tSNE']

    

    

    from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor

    from sklearn.svm import SVC, LinearSVC,NuSVC

    from sklearn.multiclass import OneVsRestClassifier

    from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor, RandomForestClassifier,ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier

    from sklearn.neural_network import MLPClassifier,MLPRegressor

    from sklearn.linear_model import PassiveAggressiveClassifier,Perceptron,SGDClassifier,LogisticRegression

    import xgboost as xgb

    from sklearn.tree import DecisionTreeClassifier

    from sklearn.naive_bayes import GaussianNB

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

    from sklearn.linear_model import ElasticNetCV,ridge_regression,HuberRegressor,LinearRegression,BayesianRidge,RANSACRegressor

    from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

    from sklearn.metrics import mean_squared_error,r2_score

    classifiers = [



                   #GradientBoostingRegressor(),

                   ExtraTreesRegressor(),

                   RandomForestRegressor(random_state=1, n_estimators=10),        

                   BayesianRidge(),

        

                   RANSACRegressor(),

                   KNeighborsRegressor(),

                   ElasticNetCV(cv=5, random_state=0),

                   HuberRegressor(),

                   LinearRegression(),

                  ]

    clanaam= ['xTreer','rFor','BaysR','Ransac','KNNr','elast','huber','linear',]

    from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

    

    results=[]





    #cluster data

    for clu in clusters:

        clunm=clunaam[clusters.index(clu)] #find naam

        X_total_clu = clu.fit_transform(np.concatenate( (X_train,X_test),axis=0))

        X_total_clu=np.concatenate((X_total_clu,np.concatenate( (X_train,X_test),axis=0)),axis=1)

        print(X_total_clu.shape)

        plt.scatter(X_total_clu[:lenxtr,0],X_total_clu[:lenxtr,1],c=Y_train.values,cmap='prism')

        plt.title(clu)

        plt.show()

        

        #classifiy 

        for cla in classifiers:

            import datetime

            start = datetime.datetime.now()

            clanm=clanaam[classifiers.index(cla)] #find naam

            

            print('    ',cla)

            cla.fit(X_total_clu,np.concatenate( (Y_train,Y_test)) )

            cla.fit(X_total_clu[:lenxtr],Y_train )

            

            #predict

            trainpredi=cla.predict(X_total_clu[:lenxtr])



            #print(classification_report(trainpredi,Y_train))            

            testpredi=cla.predict(X_total_clu[lenxtr:])  

            if classifiers.index(cla) in [0,2,3,4,5,7,8,9,10,11,12,13]:

                trainprediprob=cla.predict(X_total_clu[:lenxtr])

                testprediprob=cla.predict(X_total_clu[lenxtr:]) 

                

                plt.scatter(x=testprediprob, y=testpredi, marker='.', alpha=0.53)

                plt.show()            

            #testpredi=converging(pd.DataFrame(X_train),pd.DataFrame(X_test),Y_train,pd.DataFrame(testpredi),Y_test,clu,cla) #PCA(n_components=10,random_state=0,whiten=True),MLPClassifier(alpha=0.510,activation='logistic'))

            

            if len(dtest)==0:

                test_score=cla.score(X_total_clu[lenxtr:],Y_test)

                mse = mean_squared_error(testpredi,Y_test)

                train_score=cla.score(X_total_clu[:lenxtr],Y_train)



                li = [clunm,clanm,train_score,test_score,mse]

                results.append(li)

                r2s=r2_score(testpredi,Y_test)  

                print(r2s)



                plt.title(clanm+'test corr & mse:'+np.str(test_score)+' '+np.str(mse)+' and test confusionmatrix')

                plt.scatter(x=Y_test, y=testpredi, marker='.', alpha=1)

                plt.scatter(x=[np.mean(Y_test)], y=[np.mean(testpredi)], marker='o', color='red')

                plt.xlabel('Real test'); plt.ylabel('Pred. test')

                plt.show()





            else:

#                testpredlabel=le.inverse_transform(testpredi)  #use if you labellezid the classes 

                testpredlabel=testpredi

                print('train correl',r2_score(trainpredi,Y_train),'mse ',mean_squared_error(trainpredi,Y_train))

                submit = pd.DataFrame({'Id': dtest[indexv],'Expected': testpredlabel})

                submit['Expected']=submit['Expected'].astype('int')



                filenaam='subm_'+clunm+'_'+clanm+'.csv'

                submit.to_csv(path_or_buf =filenaam, index=False)

                

            print(clanm,'0 classifier time',datetime.datetime.now()-start)

            

    if len(dtest)==0:       

        print(pd.DataFrame(results).sort_values(3))

        submit=[]

    return submit



#Custom Transformer that extracts columns passed as argument to its constructor 

class Dummy( ):

    #Class Constructor 

    def __init__( self, feature_names ):

        self._feature_names = feature_names 

    

    #Return self nothing else to do here    

    def fit( self, X, y = None ):

        return self 

    

    #Method that describes what we need this transformer to do

    def fit_transform( self, X, y = None ):

        return X 







clustertechniques2(traingb.reset_index().append(test.reset_index()).drop('year',axis=1),'avgPrice','index') 