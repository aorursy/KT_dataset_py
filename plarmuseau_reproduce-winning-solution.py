import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
direct='3c-shared-task-influence'



train=pd.read_csv('../input/'+direct+'/train.csv')

train
stopwords=pd.read_csv('../input/smartstoplists/SmartStoplist.txt')

stopwords.columns=['word','njet']

stopwords
train.groupby('citation_influence_label').count()
test=pd.read_csv('../input/'+direct+'/test.csv')

test

def XTX(total,Qlabels,Slabels):

    from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

    from sklearn.metrics.pairwise import cosine_similarity,cosine_distances

    from sklearn.datasets import fetch_20newsgroups



    

    #enrich with external news data...

    placebo = fetch_20newsgroups()

    print("nr placebo text",len(placebo.data))



    total['Qtxt']=''

    for qi in Qlabels:

        total['Qtxt']=total['Qtxt']+' '+total[qi]

    total['Stxt']=''

    for qi in Slabels:

        total['Stxt']=total['Stxt']+' '+total[qi]        # error ' ' forgotten !

    cv = TfidfVectorizer(ngram_range=(1, 1),stop_words=list(stopwords.word.values) )

    

    Stfidf=cv.fit_transform(total['Stxt'].append(pd.Series(placebo.data)))    

    Qtfidf=cv.transform(total['Qtxt'].append(pd.Series(placebo.data)))  #newsdata added !



    print(Qtfidf.shape)#,Stfidf.shape)

    #words=cv.get_feature_names()



    #wv=TfidfVectorizer(ngram_range=(2,3),analyzer='char_wb')

    #wordcv=wv.fit_transform(words)



    from sklearn.decomposition import PCA,TruncatedSVD,NMF,FastICA

    #from sklearn.manifold import TSNE,Isomap,SpectralEmbedding,spectral_embedding,LocallyLinearEmbedding,MDS #limit number of records to 100000

    #Xi=Qtfidf.dot(Stfidf.T)

    #Xi= cosine_similarity(Qtfidf,Stfidf[:4000])

    

    #regress

    Xi=cosine_similarity(Stfidf[:4000])

    Xi=np.linalg.inv( Xi)

    print('XTXi',Xi.shape)

    Yi=cosine_similarity(Stfidf[:4000],Qtfidf[:4000])

    print('Yi',Yi.shape)

    Xi=Xi.dot(Yi)

    print('XYi',Xi.shape)

    # sparsity disappears with SVD

    Ut=Xi#Ut=TruncatedSVD(n_components=750).fit_transform(Xi)

    #Ut=np.hstack((Ut,PCA(n_components=300).fit_transform(Xi)))

    #Ut=np.hstack((Ut,FastICA(n_components=300).fit_transform(Xi)))

    #Ut=np.hstack((Ut,Isomap(n_components=10).fit_transform(Xi)))

    print('compressed',Ut.shape)

    return Ut



Xi=XTX(train.append(test),['citing_title','citing_author'],['cited_title','cited_author','citation_context'])
Xi=pd.DataFrame(Xi)

Xi['citation_influence_label']=train['citation_influence_label']

Xi['unique_id']=train['unique_id'].append(test['unique_id'],ignore_index=True)
def kluster1(data,grbvar,label,nummercl,level):

    from sklearn.cluster import KMeans

    from sklearn.metrics.pairwise import cosine_similarity

    from sklearn.metrics import confusion_matrix

    from sklearn.metrics import classification_report    

    from scipy import spatial    

    '''nummercl < ncol'''

    submit=data[data[label].isnull()==True][[grbvar,label]]

        

    

    print(label,data[label].unique())

    simdata=data[data[label].isnull()==False].drop([label],axis=1)



    #   Label encoding or remove string data

    ytrain=data[label]

    print(data[label].unique())    

    #find mean per label

    train_me=data.drop([grbvar],axis=1).groupby(label).mean()        

    #impute NAN

    kol=data.drop(label,axis=1).columns

    from sklearn.experimental import enable_iterative_imputer  

    from sklearn.impute import IterativeImputer

    if False: #len(data.dropna())>len(data[data[label]!=np.nan]):

        print('impute empty data')

        data = IterativeImputer(random_state=0).fit_transform(data.drop(label,axis=1))

    else:

        data=data.fillna(0)

    data = pd.DataFrame(data,columns=kol)   

    data[label]=ytrain.values

    

    

    #cosin similarity transform



    print(train_me)

    

    simdata=data[data[label].isnull()==False].drop([grbvar,label],axis=1)

    ytrain=data[data[label].isnull()==False][label]

    simtest=data[data[label].isnull()==True].drop([grbvar,label],axis=1)

    ytest=np.random.randint(0,1,size=(len(simtest), 1))  #fill data not used

    iddata=data[grbvar]

    #submit=data[data[label].isnull()==True][[grbvar,label]]

    print(submit.columns,submit.describe())

    if len(simtest)==0:   #randomsample if no empty label data

        simtest=data.sample(int(len(simdata)*0.2))

        ytest=simtest[label]

        simtest=simtest.drop([grbvar,label],axis=1)



    print(simdata.shape,simtest.shape,data.shape,ytrain.shape)

    #train_se=data.groupby('label').std()

    train_cs2=cosine_similarity(simdata,train_me)

    test_cs2=cosine_similarity(simtest,train_me)

    dicto={ np.round(i,1) : ytrain.unique()[i] for i in range(0, len(ytrain.unique()))} #print(clf.classes_)

    ypred=pd.Series(np.argmax(train_cs2,axis=1)).map(dicto)

    

    print('cosinesimilarity direction' ,classification_report(ytrain.values, ypred)  )

    

    trainmu=pd.DataFrame( simdata.values-simdata.values.mean(axis=1)[:,None])

    testmu=pd.DataFrame( simtest.values-simtest.values.mean(axis=1)[:,None])

    

    trainmu[label]=ytrain

    trainme2=trainmu.groupby(label).mean()    

    #spatial 0.79

    def verslag(titel,yval,ypred,ypred2):

        yval=pd.Series(yval)

        ypred=pd.Series(ypred)

        ypred2=pd.Series(ypred2)

        print(len(yval.dropna()),len(yval),len(ypred),len(ypred.dropna()))

        ypred=ypred.fillna(0)

        ypred2=ypred2.fillna(0)

        print(titel+'\n', classification_report(yval,ypred )  )

        submit[label]=[xp for xp in ypred2]

        submit[label]=submit[label].astype('int')

        #print(submit[[grbvar,label]])

        #submit[grbvar]=submit[grbvar].values#.astype('int')

        submit[[grbvar,label]].to_csv(titel+'submission.csv',index=False)

        print(titel,submit[[grbvar,label]].groupby(label).count() )

        return

    

    def adjcos_dist(size, matrix, matrixm):

        distances = np.zeros((len(matrix),size))

        M_u = matrix.mean(axis=1)

        m_sub = matrix - M_u[:,None]

        for first in range(0,len(matrix)):

            for sec in range(0,size):

                distance = spatial.distance.cosine(m_sub[first],matrixm[sec])

                distances[first,sec] = distance

        return distances



    trainsp2=adjcos_dist(len(trainme2),trainmu.drop(label,axis=1).values,trainme2.values)

    testsp2=adjcos_dist(len(trainme2),testmu.values,trainme2.values)

    

    print(trainsp2.shape,trainme2.shape,simdata.shape)

    verslag('cosinesimilarity distance', ytrain, pd.Series(np.argmin(trainsp2,axis=1)).map(dicto),pd.Series(np.argmin(testsp2,axis=1)).map(dicto)  )  



    return data







#train2=kluster2( train.append(test,ignore_index=True),'Id','Credit Default',len(train['Credit Default'].unique() ),1)

train2=kluster1(Xi[:4000],'unique_id','citation_influence_label',len(Xi['citation_influence_label'].unique() )-1,1)
def kluster2(data,grbvar,label,nummercl,level):

    '''nummercl < ncol'''

    submit=data[data[label].isnull()==True][[grbvar,label]]

        

    

    print(label,data[label].unique())

    from sklearn.cluster import KMeans

    from sklearn.metrics.pairwise import cosine_similarity

    from sklearn.metrics import confusion_matrix

    from sklearn.metrics import classification_report    

    from scipy import spatial

    import time

    import matplotlib.pyplot as plt

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    from sklearn.neighbors import KNeighborsClassifier

    from sklearn.decomposition import PCA,TruncatedSVD,NMF,FastICA

    from umap import UMAP  # knn lookalike of tSNE but faster, so scales up

    from sklearn.manifold import TSNE,Isomap,SpectralEmbedding,spectral_embedding,LocallyLinearEmbedding,MDS #limit number of records to 100000

    from sklearn.gaussian_process import GaussianProcessClassifier

    from sklearn.gaussian_process import GaussianProcessClassifier

    from sklearn.neighbors import KNeighborsClassifier

    from sklearn.neural_network import MLPClassifier

    from sklearn.ensemble import RandomForestClassifier

    from sklearn.linear_model import LogisticRegression

    from sklearn.svm import SVC,NuSVC

    import xgboost as xgb

    from lightgbm import LGBMClassifier,LGBMRegressor    

    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

    from sklearn.naive_bayes import GaussianNB

    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    from sklearn.linear_model import SGDClassifier

    simdata=data[data[label].isnull()==False].drop([label],axis=1)



    #   Label encoding or remove string data

    ytrain=data[label]

    if False: 

        from category_encoders.cat_boost import CatBoostEncoder

        CBE_encoder = CatBoostEncoder()

        cols=[ci for ci in data.columns if ci not in ['index',label]]

        coltype=data.dtypes

        featured=[ci for ci in cols]

        ytrain=data[label]

        CBE_encoder.fit(data[:len(simdata)].drop(label,axis=1), ytrain[:len(simdata)])

        data=CBE_encoder.transform(data.drop(label,axis=1))

        data[label]=ytrain

    print(data[label].unique())    

    #find mean per label

    train_me=data.drop([grbvar],axis=1).groupby(label).mean()        

    #impute NAN

    kol=data.drop(label,axis=1).columns

    from sklearn.experimental import enable_iterative_imputer  

    from sklearn.impute import IterativeImputer

    if False: #len(data.dropna())>len(data[data[label]!=np.nan]):

        print('impute empty data')

        data = IterativeImputer(random_state=0).fit_transform(data.drop(label,axis=1))

    else:

        data=data.fillna(0)

    data = pd.DataFrame(data,columns=kol)   

    data[label]=ytrain.values

    

    

    #cosin similarity transform



    print(train_me)

    

    simdata=data[data[label].isnull()==False].drop([grbvar,label],axis=1)

    ytrain=data[data[label].isnull()==False][label]

    simtest=data[data[label].isnull()==True].drop([grbvar,label],axis=1)

    ytest=np.random.randint(0,1,size=(len(simtest), 1))  #fill data not used

    iddata=data[grbvar]

    #submit=data[data[label].isnull()==True][[grbvar,label]]

    print(submit.columns,submit.describe())

    if len(simtest)==0:   #randomsample if no empty label data

        simtest=data.sample(int(len(simdata)*0.2))

        ytest=simtest[label]

        simtest=simtest.drop([grbvar,label],axis=1)



    print(simdata.shape,simtest.shape,data.shape,ytrain.shape)

    #train_se=data.groupby('label').std()

    train_cs2=cosine_similarity(simdata,train_me)

    test_cs2=cosine_similarity(simtest,train_me)

    dicto={ np.round(i,1) : ytrain.unique()[i] for i in range(0, len(ytrain.unique()))} #print(clf.classes_)

    ypred=pd.Series(np.argmax(train_cs2,axis=1)).map(dicto)

    

    print('cosinesimilarity direction' ,classification_report(ytrain.values, ypred)  )

    

    trainmu=pd.DataFrame( simdata.values-simdata.values.mean(axis=1)[:,None])

    testmu=pd.DataFrame( simtest.values-simtest.values.mean(axis=1)[:,None])

    

    trainmu[label]=ytrain

    trainme2=trainmu.groupby(label).mean()    

    #spatial 0.79

    def verslag(titel,yval,ypred,ypred2):

        yval=pd.Series(yval)

        ypred=pd.Series(ypred)

        ypred2=pd.Series(ypred2)

        print(len(yval.dropna()),len(yval),len(ypred),len(ypred.dropna()))

        ypred=ypred.fillna(0)

        ypred2=ypred2.fillna(0)

        print(titel+'\n', classification_report(yval,ypred )  )

        submit[label]=[xp for xp in ypred2]

        submit[label]=submit[label].astype('int')

        #print(submit[[grbvar,label]])

        #submit[grbvar]=submit[grbvar].values#.astype('int')

        submit[[grbvar,label]].to_csv(titel+'submission.csv',index=False)

        print(titel,submit[[grbvar,label]].groupby(label).count() )

        return

    

    def adjcos_dist(size, matrix, matrixm):

        distances = np.zeros((len(matrix),size))

        M_u = matrix.mean(axis=1)

        m_sub = matrix - M_u[:,None]

        for first in range(0,len(matrix)):

            for sec in range(0,size):

                distance = spatial.distance.cosine(m_sub[first],matrixm[sec])

                distances[first,sec] = distance

        return distances



    trainsp2=adjcos_dist(len(trainme2),trainmu.drop(label,axis=1).values,trainme2.values)

    testsp2=adjcos_dist(len(trainme2),testmu.values,trainme2.values)

    

    print(trainsp2.shape,trainme2.shape,simdata.shape)

    verslag('cosinesimilarity distance', ytrain, pd.Series(np.argmin(trainsp2,axis=1)).map(dicto),pd.Series(np.argmin(testsp2,axis=1)).map(dicto)  )  

    # blended with three classifiers random Forest

    classifier=[LGBMClassifier(),

                RandomForestClassifier(n_jobs=4),

                KNeighborsClassifier(n_neighbors=3),  #0.67

                xgb.XGBClassifier(n_estimators=50, max_depth = 9, learning_rate=0.01, subsample=0.75, random_state=11,n_jobs=4), #0.59        

                QuadraticDiscriminantAnalysis(),  #0.5        

                AdaBoostClassifier(), #0.46        

                PCA(n_components=nummercl*3,random_state=0,whiten=True),  #0.44

                TruncatedSVD(n_components=nummercl*3, n_iter=7, random_state=42), #0.44

                GaussianNB(),  #0.44

                LogisticRegression(n_jobs=4),        #use binary classification

                MLPClassifier(alpha=1, max_iter=1000), #0.37

                FastICA(n_components=nummercl,random_state=0),  #0.35  use when consecutiverelationship        

                #SVC(probability=True), #0.32        

                #Isomap(n_components=nummercl),



    ]

    simdata2=np.hstack((train_cs2,trainsp2))

    simtest2=np.hstack((test_cs2,testsp2))

    kol2=['x'+str(xi) for xi in range(nummercl)]+['y'+str(xi) for xi in range(nummercl)]

    for clf in classifier:

        #clf = RandomForestClassifier(n_jobs=4) #GaussianProcessClassifier()#

        print(simdata.shape,ytrain.shape,simtest.shape,data.shape,simdata2.shape,simtest2.shape)

        #print(simtest2)



        try:

            clf.fit(simdata, ytrain)

            train_tr=clf.predict_proba(simdata)

            test_tr=clf.predict_proba(simtest)

        except:

            clf.fit(simdata.append(simtest))

            train_tr=clf.transform(simdata)

            test_tr=clf.transform(simtest)

            

        #dicto={ i : clf.classes_[i] for i in range(0, len(clf.classes_) ) } #print(clf.classes_)

        ypred=pd.Series(np.argmax(train_tr,axis=1)).map(dicto)

        ypred2=pd.Series(np.argmax(test_tr,axis=1)).map(dicto)

        verslag('1'+str(clf)[:5]+'class' ,ytrain, ypred,ypred2  )

        simdata2=np.hstack((simdata2,train_tr))

        simtest2=np.hstack((simtest2,test_tr))

        kol2=kol2+[str(clf)[:3]+str(xi) for xi in range(train_tr.shape[1])]

    #concat data

    simdata=pd.DataFrame(simdata2,columns=kol2)

    simtest=pd.DataFrame(simtest2,columns=kol2)



    #plotimg2=pd.DataFrame(train_cs2,columns=['x'+str(xi) for xi in range(nummercl)])

    nummercl=3

    clusters = [

                PCA(n_components=nummercl*10,random_state=0,whiten=True),

                TruncatedSVD(n_components=nummercl*10, n_iter=7, random_state=42),

                FastICA(n_components=nummercl*50,random_state=0),

                Isomap(n_components=nummercl*30),

                #LocallyLinearEmbedding(n_components=nummercl),

                SpectralEmbedding(n_components=nummercl),

                #MDS(n_components=nummercl),

                TSNE(n_components=nummercl,random_state=0),

                UMAP(n_neighbors=3,n_components=nummercl, min_dist=0.3,metric='minkowski'),

                #grbvarNMF(n_components=nummercl,random_state=0),                

                ] 

    clunaam=['PCA','tSVD','ICA','Iso','Spectr','tSNE','UMAP','NMF']

    

    #clf = RandomForestClassifier()

    #from sklearn.linear_model import SGDClassifier

    #clf= SGDClassifier(max_iter=5)

    #classifier after clustering

    clf=xgb.XGBClassifier(n_estimators=50, max_depth = 9, learning_rate=0.01, subsample=0.75, random_state=11,n_jobs=4)

    clf=LGBMClassifier()

    clf.fit(simdata, ytrain)

        

    verslag('2xgb pure',ytrain,pd.Series(clf.predict(simdata)),pd.Series(clf.predict(simtest))  )

            

    for cli in clusters:

        print(cli)

        clunm=clunaam[clusters.index(cli)] #find naam

        

        if str(cli)[:3]=='NMF':

            maxmin=np.array([simdata.min(),simtest.min()])

            simdata=simdata-maxmin.min()+1

        svddata = cli.fit_transform(simdata.append(simtest))  #totale test

        

        km = KMeans(n_clusters=nummercl, random_state=0)

        km.fit_transform(svddata)

        cluster_labels = km.labels_

        cluster_labels = pd.DataFrame(cluster_labels, columns=[label])

        #print(cluster_labels.shape) # train+test ok

        pd.DataFrame(svddata[:len(simdata)]).plot.scatter(x=0,y=1,c=ytrain.values,colormap='viridis')

        clf.fit(svddata[:len(simdata)], ytrain)

        verslag('3'+clunm+'+lgbm reduced',ytrain,pd.Series(clf.predict(svddata[:len(simdata)])),pd.Series(clf.predict(svddata[len(simdata):])))        

        

    

        plt.show()



        #clusdata=pd.concat([pd.DataFrame(grbdata.reset_index()[grbvar]), cluster_labels], axis=1)

        #if len(grbdata)<3: 

        #    data['Clu'+clunm+str(level)]=cluster_labels.values

            

        #else:

        #    data=data.merge(clusdata,how='left',left_on=grbvar,right_on=grbvar)

        confmat=confusion_matrix ( ytrain,cluster_labels[:len(simdata)])

        dicti={}

        for xi in range(len(confmat)):

            #print(np.argmax(confmat[xi]),confmat[xi])

            dicti[xi]=np.argmax(confmat[xi])

        #print(dicti)

        #print('Correlation\n',confusion_matrix ( ytrain,cluster_labels[:len(ytrain)]))

        #print(clunm+'+kmean clusterfit', classification_report(ytrain.map(dicti), cluster_labels[:len(simdata)])  )   

        invdict = {np.round(value,1): key for key, value in dicti.items()}

        #print(invdict)

        #submit[label]=cluster_labels[len(simdata):].values

        #print(cluster_labels[len(simdata):])

        #print(submit.describe().T)

        #ytest=submit[label].astype('int')



        #submit[label]=ytest.map(invdict)#.astype('int')

        #submit[[grbvar,label]].to_csv('submit'+str(cli)[:5]+'kmean.csv',index=False)

        #print('kmean'+str(cli)[:10],submit[[grbvar,label]].groupby(label).count() )        

    return data







#train2=kluster2( train.append(test,ignore_index=True),'Id','Credit Default',len(train['Credit Default'].unique() ),1)

train2=kluster2(Xi[:4000],'unique_id','citation_influence_label',len(Xi['citation_influence_label'].unique() )-1,1)
def verslag(titel,label2,yval,ypred,ypred2,mytrain):

        from sklearn.metrics import classification_report    

        yval=pd.Series(yval)

        ypred=pd.Series(ypred)

        ypred2=pd.Series(ypred2)

        print('shape yval/dropna ypred/dropna',yval.dropna().shape,yval.shape,ypred.shape,ypred.dropna().shape)

        ypred=ypred.fillna(0)

        ypred2=ypred2.fillna(0)

        print(titel+'\n', classification_report(yval,ypred )  )

        #print(mytrain)

        vsubmit = pd.DataFrame({        label2[0]: mytrain[len(yval):][label2[0]].values,        label2[1]: ypred2    })

        #vsubmit = pd.DataFrame({        label2[0]: mytrain[len(yval):].reset_index().index,        label2[1]: ypred2    })

 

        

        #print(vsubmit)

        #print(label2,label2[0],label2[1 ],vsubmit.shape,vsubmit.head(3))

        vsubmit[label2[1]]=vsubmit[label2[1]].astype('int')#-1

        print('submission header',vsubmit.head())

        vsubmit[label2].to_csv(titel+'submission.csv',index=False)

        print(titel,vsubmit[label2].groupby(label2[1]).count() )

        return  



    

def ensemblecluster(data,grbvar,label,nummerclas,level):

    '''nummercl < ncol'''

    nummercl=nummerclas

    print('nummerclust',nummercl)

    from sklearn.cluster import KMeans

    from sklearn.metrics.pairwise import cosine_similarity

    from sklearn.metrics import confusion_matrix

    from sklearn.metrics import classification_report    

    from scipy import spatial

    import time

    import matplotlib.pyplot as plt

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    from sklearn.neighbors import KNeighborsClassifier

    from sklearn.decomposition import PCA,TruncatedSVD,NMF,FastICA

    from umap import UMAP  # knn lookalike of tSNE but faster, so scales up

    from sklearn.manifold import TSNE,Isomap,SpectralEmbedding,spectral_embedding,LocallyLinearEmbedding,MDS #limit number of records to 100000

    from sklearn.gaussian_process import GaussianProcessClassifier

    from sklearn.gaussian_process import GaussianProcessClassifier

    from sklearn.neighbors import KNeighborsClassifier

    from sklearn.neural_network import MLPClassifier

    from sklearn.ensemble import RandomForestClassifier

    from sklearn.linear_model import LogisticRegression

    from sklearn.svm import SVC,NuSVC

    import xgboost as xgb

    from lightgbm import LGBMClassifier,LGBMRegressor     

    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

    from sklearn.naive_bayes import GaussianNB

    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    from sklearn.linear_model import SGDClassifier

    from sklearn.multiclass import OneVsRestClassifier

    from sklearn.linear_model import Perceptron

    simdata=data[data[label].isnull()==False].drop([grbvar,label],axis=1)

    from sklearn.gaussian_process import GaussianProcessClassifier

    from sklearn.gaussian_process.kernels import RBF

    from sklearn.metrics import f1_score

    from sklearn.ensemble import VotingClassifier,BaggingClassifier

    

    #scaling

    from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,StandardScaler,RobustScaler,Normalizer,QuantileTransformer,PowerTransformer

    scalers=[Dummy(1),

            MinMaxScaler(),

            MaxAbsScaler(),

            StandardScaler(),

            RobustScaler(),

            Normalizer(),

            QuantileTransformer(output_distribution='uniform'),

            PowerTransformer(),

           ]

    # cluster techniques

    nummercl=int( (data.shape[1])**0.8)

    print('reduce data to',nummercl)

    clusters = [Dummy(1),

                PCA(n_components=nummercl,random_state=0,whiten=True),

                TruncatedSVD(n_components=nummercl, n_iter=7, random_state=42),

                FastICA(n_components=nummercl*2,random_state=0),

                #Isomap(n_components=nummercl*10),

                #LocallyLinearEmbedding(n_components=nummercl*2),

                SpectralEmbedding(n_components=nummercl*2),

                #MDS(n_components=nummercl),

                TSNE(n_components=3,random_state=0),

                UMAP(n_neighbors=nummercl*1,n_components=10, min_dist=0.3,metric='minkowski'),

                #NMF(n_components=nummercl,random_state=0),                

                ] 

    #classifier techniques

    classifiers=[LGBMClassifier(),

                 OneVsRestClassifier(LogisticRegression(),n_jobs=4),

                 

                VotingClassifier(estimators=[('rf', RandomForestClassifier(n_jobs=4)), ('xg',xgb.XGBClassifier(n_jobs=4)),('lr',LogisticRegression(n_jobs=4)) ,('kn', KNeighborsClassifier(n_neighbors=3))], voting='soft',n_jobs=4),

                BaggingClassifier(base_estimator=xgb.XGBClassifier(),n_jobs=4),

                LogisticRegression(n_jobs=4),

                #xgb.XGBClassifier(n_estimators=50, max_depth = 9, learning_rate=0.01, subsample=0.75, random_state=11,n_jobs=4),

                xgb.XGBClassifier( learning_rate=0.02, max_delta_step=0, max_depth=10, min_child_weight=0.1, missing=None, n_estimators=250, nthread=4,objective='binary:logistic', reg_alpha=0.01, reg_lambda = 0.01,scale_pos_weight=1, seed=0, silent=False, subsample=0.9),

                #GaussianProcessClassifier(kernel=1.0 * RBF(1.0), random_state=0),

                SVC(probability=True),

                KNeighborsClassifier(n_neighbors=3),

                #PassiveAggressiveClassifier(max_iter=50, tol=1e-3,n_jobs=-1),    

                #Perceptron(n_jobs=4),

                SGDClassifier(n_jobs=4),

                #MLPClassifier(alpha=1, max_iter=1000),

                AdaBoostClassifier(),

                GaussianNB(),

                #QuadraticDiscriminantAnalysis()

            ]    

    ydata=data[[grbvar,label]]

    

    if False:

        

        kolom=data.describe().T

        from sklearn.preprocessing import OneHotEncoder

        toencode=[ci for ci in data.columns if ci not in kolom.index]

        ohe=OneHotEncoder()

        data2=ohe.fit_transform(data[toencode].fillna('')).toarray()

        data=data.drop(toencode,axis=1)

        #from sklearn.preprocessing import OneHotEncoder

        #data2=OneHotEncoder().fit_transform(data[['Gender']]).toarray()

        tel=0

        for ci in ohe.get_feature_names():

            data[ci]=data2[:,tel]

            tel+=1

    #add random columns

    for xi in range(3):

        labnm='rand'+str(xi)

        data[labnm]=np.random.randint(0,1,size=(len(data), 1))

    kol=data.drop([grbvar,label],axis=1).columns    

    #check to impute

    #data=data.fillna(data.mean())

    null_columns=data.columns[data.isnull().any()]

    print('what is null \n',data[null_columns].isnull().sum()) 

    if len(null_columns)==0:

        from sklearn.model_selection import train_test_split

        simdata,simtest,ytrain,ytest=train_test_split(data.drop([grbvar,label],axis=1),data[[grbvar,label]],test_size=0.1)

        print('splitted',simdata.shape,simtest.shape,ytrain.shape,ytest.shape)

    else:

        simdata=data[data[label].isnull()==False].drop([grbvar,label],axis=1).fillna(data.mean())

        ytrain=ydata[ydata[label].isnull()==False][[grbvar,label]] # exceptional has to convert to integer for this

        simtest=data[data[label].isnull()==True].drop([grbvar,label],axis=1).fillna(data.mean())

        ytest=pd.DataFrame(np.random.randint(0,1,size=(len(simtest), 1)),columns=[label])  #fill data not used

    #iddata=data[grbvar]

    #submit=data[data[label].isnull()==True][[grbvar,label]]

    #ydata=ytrain.append(ytest)

    #data=

    #data=XTX(data,grbvar,)

    resul=[]

    

    def scalclusclas(scai,clusi,clasi,verbose):

        mdata = scai.fit_transform(simdata.append(simtest))

        svddata = clusi.fit_transform(mdata)

        naam=str(clasi)[:10]+str(clusi)[:7]+str(scai)[:10]        

        if verbose:

            clasi.fit(svddata[:len(simdata)],ytrain[label])

            train_tr=clasi.predict(svddata[:len(simdata)])  

            test_tr=clasi.predict(svddata[len(simdata):])

            f1sc=f1_score(ytrain[:len(simdata)][label],train_tr, average=None)

            f1te=f1_score(ytest[label],np.round(test_tr), average=None)

            

        else:

            pointer=int(len(simdata)*.2)

            clasi.fit(svddata[:len(simdata)-pointer],ytrain[:len(simdata)-pointer][label])

            train_tr=clasi.predict(svddata[:len(simdata)-pointer])  

            test_tr=clasi.predict(svddata[len(simdata)-pointer:len(simdata)])

            f1sc=f1_score(ytrain[:len(simdata)-pointer][label],train_tr, average=None)

            f1te=f1_score(ytrain[len(simdata)-pointer:len(simdata)][label],np.round(test_tr), average=None)

        resul=[naam]+[xi for xi in f1sc]+[xi for xi in f1te]

        if verbose:

            verslag('3_'+naam,[grbvar,label],ytrain[:len(simdata)][label],train_tr,test_tr,ydata)

        print(naam,resul)

        return resul

        

    print('________________find best scaler')

    resultsc=[]

    for scai in scalers:

        for clusi in clusters[:1]:

            for clasi in classifiers[:1]:

                resultsc.append(scalclusclas(scai,clusi,clasi,False))

    resultsc=pd.DataFrame(resultsc)

    resultsc['som']=resultsc.iloc[:,nummerclas+1:].sum(axis=1)/nummerclas  #nummerclas

    print(resultsc)

    maxscale=resultsc.sort_values('som')[-3:]

    print(maxscale)

    maxscale=[xi for xi in maxscale.index]

    print(maxscale)

    print('________________find best cluster')

    resultlu=[]    

    for scai in scalers[maxscale[2]:maxscale[2]+1]:

        for clusi in clusters:

            for clasi in classifiers[:1]:

                resultlu.append(scalclusclas(scai,clusi,clasi,False))

    resultlu=pd.DataFrame(resultlu)

    resultlu['som']=resultlu.iloc[:,nummerclas+1:].sum(axis=1)/nummerclas #nummerclas

    print(resultlu)

    maxclus=resultlu.sort_values('som')[-3:]

    print(maxclus)

    maxclus=[xi for xi in maxclus.index]

    print(maxclus)

    print('________________find best classifier')

    resultla=[]    

    for scai in scalers[maxscale[2]:maxscale[2]+1]:

        for clusi in clusters[maxclus[2]:maxclus[2]+1]:

            for clasi in classifiers:

                resultla.append(scalclusclas(scai,clusi,clasi,False))

    resultla=pd.DataFrame(resultla)

    resultla['som']=resultla.iloc[:,nummerclas+1:].sum(axis=1)/nummerclas #nummerclas

    maxclas=resultla.sort_values('som')[-3:]

    maxclas=[xi for xi in maxclas.index]

    print(maxclas)



   

    results=[]

    for scai in maxscale:

        for clusi in maxclus:

            for clasi in maxclas:

                results.append(scalclusclas(scalers[scai],clusters[clusi],classifiers[clasi],True))

              



    zoek=pd.DataFrame(results)

    zoek= zoek.append(resultsc).append(resultlu).append(resultla)

    zoek['som']=zoek.iloc[:,nummercl+1:].sum(axis=1)/nummercl

    

    print(zoek.sort_values(['som']))            

    



    return zoek





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

#zoek=ensemblecluster(mtotal[:len(train)],'Date','Category',len(train['Category'].unique() ),1)

#zoek=ensemblecluster(train,'index','Lable',len(total['Lable'].unique() ),1)

#

zoek=ensemblecluster(Xi[:4000],'unique_id','citation_influence_label',len(Xi['citation_influence_label'].unique() )-1,1)