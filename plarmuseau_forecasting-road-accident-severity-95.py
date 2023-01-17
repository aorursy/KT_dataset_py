# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/road-accidents-incidence/Kaagle_Upload.csv", sep=",", decimal=",", engine='python') # Read the data from a csv
df
def clustertechniques2(dtrain,label,indexv):

    print('#encodings',dtrain.shape)

    cols=[ci for ci in dtrain.columns if ci not in [indexv,'index',label]]

    dtest=dtrain[dtrain[label].isnull()==True][[indexv,label]]

    print(dtest)



    if True:

        # One Hot Encode target mean()

        cols=[ci for ci in dtrain.columns if ci not in [indexv,'index',label]]

        coltype=dtrain.dtypes

        for ci in cols:

            if (coltype[ci]=="object"):

                codes=dtrain[[ci,label]].groupby(ci).mean().sort_values(label)

                codesdict=codes[label].to_dict()

                dtrain[ci]=dtrain[ci].map(codesdict)

                #if len(dtest)>0:  not necessary

                #    dtest[ci]=dtest[ci].map(codesdict)

    elif False:

        for i in cols:

            dtrain[i].fillna("Missing", inplace=True)

            dummies=pd.get_dummies(dtrain[i], prefix=i)

            dtrain=pd.concat([dtrain,dummies],axis=1)

            dtrain.drop([i],axis=1,inplace=True)

    print('encodings  after shape',dtrain.shape)

    #split data or use splitted data

    X_train=dtrain[dtrain[label].isnull()==False].drop([indexv,label],axis=1).fillna(0)

    Y_train=dtrain[dtrain[label].isnull()==False][label]

    X_test=dtrain[dtrain[label].isnull()==True].drop([indexv,label],axis=1).fillna(0)

    Y_test=np.random.random((X_test.shape[0],1))

    if len(X_test)==0:

        from sklearn.model_selection import train_test_split

        X_train,X_test,Y_train,Y_test = train_test_split(dtrain.drop(label,axis=1).fillna(0),dtrain[label],test_size=0.250,random_state=0)

    lenxtr=len(X_train)

    print('splitting data train test X-y',X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)

    

    

    import matplotlib.pyplot as plt    

    #COHEN STATS

    splitval=Y_train.mean()

    print('Cohen splitting on',splitval)

    group1, group2 = X_train[Y_train<splitval], X_train[Y_train>splitval]

    diff = group1.mean() - group2.mean()

    var1, var2 = group1.var(), group2.var()

    n1, n2 = group1.shape[0], group2.shape[0]

    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)

    d = diff / np.sqrt(pooled_var)

    #GRAPH

    features=[ci for ci in dtrain.columns if ci in d.index]

    d.reindex(d.abs().sort_values(ascending=False).nlargest(50).index)[::-1].plot.barh(figsize=(6, 10));plt.show()

    print('Features with the 30 largest effect sizes')

    significant_features = [f for f in features if np.abs(d.loc[f]) > 0.0025125]

    print('Significant features %d: %s' % (len(significant_features), significant_features))        

    X_train=X_train[significant_features]

    X_test=X_test[significant_features]







    from sklearn import preprocessing

    scale = preprocessing.MinMaxScaler().fit(X_train)

    X_train = scale.transform(X_train)

    X_test = scale.transform(X_test)



    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    from sklearn.neighbors import KNeighborsClassifier,NeighborhoodComponentsAnalysis

    from sklearn.decomposition import PCA,TruncatedSVD,NMF,FastICA

    from umap import UMAP  # knn lookalike of tSNE but faster, so scales up

    from sklearn.manifold import TSNE #limit number of records to 100000



    clusters = [#

                #

                PCA(n_components=0.7,random_state=0,whiten=True),       

                TruncatedSVD(n_components=5, n_iter=7, random_state=42),

                #UMAP(n_neighbors=5,n_components=10, min_dist=0.3,metric='minkowski'),

                Dummy(1),

                #FastICA(n_components=7,random_state=0),

                #NMF(n_components=10,random_state=0),            

                #TSNE(n_components=3,random_state=0)

                ] 

    clunaam=['PCA','tSVD','raw']#,'ICA','tSVD','nmf','UMAP','tSNE']

    

    

    from sklearn.neighbors import KNeighborsClassifier

    from sklearn.svm import SVC, LinearSVC,NuSVC

    from sklearn.multiclass import OneVsRestClassifier

    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier,ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier

    from sklearn.calibration import CalibratedClassifierCV

    from sklearn.neural_network import MLPClassifier,MLPRegressor

    from sklearn.linear_model import PassiveAggressiveClassifier,Perceptron,SGDClassifier,LogisticRegression

    import xgboost as xgb

    from sklearn.tree import DecisionTreeClassifier

    from sklearn.naive_bayes import GaussianNB

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

    

    classifiers = [#LinearDiscriminantAnalysis(n_components=2),

                    #NeighborhoodComponentsAnalysis(n_components=2,random_state=42),

                    LogisticRegression( solver="lbfgs",max_iter=500,n_jobs=-1),

                    #CalibratedClassifierCV(LogisticRegression( solver="lbfgs",max_iter=500,n_jobs=-1), method='isotonic', cv=5),

                    PassiveAggressiveClassifier(max_iter=50, tol=1e-3,n_jobs=-1),    

                    #KNeighborsClassifier(n_neighbors=5,n_jobs=-1), 

                    RandomForestClassifier(n_estimators=100, random_state=42,n_jobs=-1, oob_score=True),# method='isotonic', cv=5),

                    #ExtraTreesClassifier(n_estimators=10, max_depth=50, min_samples_split=5, min_samples_leaf=1, random_state=None, min_impurity_decrease=1e-7), 

                   #SVC(gamma='auto'),                   

                   #MLPClassifier(alpha=0.510,activation='logistic'),

                   #SGDClassifier(average=True,max_iter=100),

                   xgb.XGBClassifier(n_estimators=50, max_depth = 9, learning_rate=0.01, subsample=0.75, random_state=11),

                   ##SVC(kernel="rbf", C=0.025, probability=True),

                   #NuSVC(probability=True),

                   #CalibratedClassifierCV(DecisionTreeClassifier(), method='isotonic', cv=5),

                   #AdaBoostClassifier(),

                   GradientBoostingClassifier(),

                   GaussianNB(),

                   ##LinearDiscriminantAnalysis(),

                   ##QuadraticDiscriminantAnalysis()

                  ]

    clanaam= ['Logi','PasAggr','rFor','GrBo','Xgb','Gauss','QDis']#,'KNN','rFor','Xtr','MLP','Decis','LinDis','QuadDis']#['Logi','KNN','rFor','SVC','MLP','SGD']

    from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

    

    results=[]





    #cluster data

    for clu in clusters:

        clunm=clunaam[clusters.index(clu)] #find naam

        X_total_clu = clu.fit_transform(np.concatenate( (X_train,X_test),axis=0))

        #embed cluster with raw data

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

            #cla.fit(X_total_clu,np.concatenate( (Y_train,Y_test)) )

            cla.fit(X_total_clu[:lenxtr],Y_train )

            

            #predict

            #trainpredi=cla.predict(X_total_clu[:lenxtr])



            #embed prediction with data    

            if classifiers.index(cla) in [0]:#,1,2,3,4,5,6,7,8,9,10,11,12,13]:            

                totpredi=cla.predict_proba(X_total_clu)

                clus2=TruncatedSVD(n_components=5, n_iter=7, random_state=42).fit_transform(np.concatenate( (np.concatenate( (X_train,X_test),axis=0),totpredi),axis=1)  )

                X_total_clu=np.concatenate( (np.concatenate( (X_train,X_test),axis=0),clus2),axis=1)

                

            else:

                totpredi=cla.predict(X_total_clu)

                X_total_clu=np.concatenate( (X_total_clu,totpredi.reshape(-1,1)),axis=1)

            #X_total_clu = clu.fit_transform(X_total_clu)

            

            cla.fit(X_total_clu[:lenxtr],Y_train )

            

            trainpredi=cla.predict(X_total_clu[:lenxtr])            

            print(classification_report(trainpredi,Y_train))            

            testpredi=cla.predict(X_total_clu[lenxtr:])  

            if classifiers.index(cla) in [0]:#,2,3,4,5,7,8,9,10,11,12,13]:

                trainprediprob=cla.predict_proba(X_total_clu[:lenxtr])

                testprediprob=cla.predict_proba(X_total_clu[lenxtr:]) 

                plt.scatter(x=testprediprob[:,1], y=testpredi, marker='.', alpha=0.3)

                plt.show()            

            #testpredi=converging(pd.DataFrame(X_train),pd.DataFrame(X_test),Y_train,pd.DataFrame(testpredi),Y_test,clu,cla) #PCA(n_components=10,random_state=0,whiten=True),MLPClassifier(alpha=0.510,activation='logistic'))

            

            if len(dtest)==0:

                test_score=cla.score(X_total_clu[lenxtr:],Y_test)

                accscore=accuracy_score(testpredi,Y_test)

                

                train_score=cla.score(X_total_clu[:lenxtr],Y_train)



                li = [clunm,clanm,train_score,accscore]

                results.append(li)

                verhoud=len(Y_test)/len(Y_train)

                print(np.round( confusion_matrix(testpredi,Y_test)*verhoud / ( confusion_matrix(trainpredi,Y_train)+1 ) *100,0) )



                plt.title(clanm+'test accuracy versus unknown:'+np.str(test_score)+' '+np.str(accscore)+' and test confusionmatrix')

                plt.scatter(x=Y_test, y=testpredi, marker='.', alpha=1)

                plt.scatter(x=[np.mean(Y_test)], y=[np.mean(testpredi)], marker='o', color='red')

                plt.xlabel('Real test'); plt.ylabel('Pred. test')

                plt.show()

                #print(cla.coef_)



            else:

#                testpredlabel=le.inverse_transform(testpredi)  #use if you labellezid the classes 

                testpredlabel=testpredi

                print(confusion_matrix(trainpredi,Y_train))

                submit = pd.DataFrame({indexv: dtest[indexv],label: testpredlabel})

                submit[label]=submit[label].astype('int')



                filenaam='subm_'+clunm+'_'+clanm+'.csv'

                submit.to_csv(path_or_buf =filenaam, index=False)

                

            print(clanm,'0 classifier time',datetime.datetime.now()-start)

            

    if len(dtest)==0:       

        print(pd.DataFrame(results).sort_values(3))

        submit=[]

    return submit,significant_features#,cla.coef_



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

sub,sign=clustertechniques2(df,'accident_severity','accident_index') #total[len(train):].fillna(0)

#enc
