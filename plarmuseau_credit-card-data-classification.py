# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
credit = pd.read_csv("../input/creditcard.csv")
credit.head()
credit.describe()
credit.info()
def Klasseer(Mtrain,Mtest,Mlabel,klas,rank,start):
    #data preparation
    #print(Mtotal)
    #Mtotal=Mtotal.fillna(-1)
    #print(Mtotal)
    #Mtrain=Mtotal[Mtotal[labelveld]!=-1]
    #Mtest=Mtotal[Mtotal[labelveld]==-1]
    #Mtest=Mtest.drop(labelveld,axis=1)
    Mlabel=pd.DataFrame( Mlabel,columns=['label'] )  #[:len(Mtrain)]
    #Mlabel=Mlabel.fillna(-1)  
    labelveld='label'
    print('shapes train',Mtrain.shape,'label',Mlabel.shape,'test',Mtest.shape)

    
    #totalA=Mtrain.append(Mtest)
    totalA=np.concatenate((Mtrain,Mtest), axis=0)
    predictionA=pd.DataFrame(Mlabel,columns=[labelveld])    
    #totalA=totalA.drop(labelveld,axis=1)
    #print(totalA.shape,predictionA.shape)
    #print(prediction)
    #faze 1
    # dimmension reduction
    from scipy.spatial.distance import cosine
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import Normalizer
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC, LinearSVC, NuSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,ExtraTreesClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.linear_model import OrthogonalMatchingPursuit,RANSACRegressor,LogisticRegression,ElasticNetCV,HuberRegressor, Ridge, Lasso,LassoCV,Lars,BayesianRidge,SGDClassifier,LogisticRegressionCV,RidgeClassifier,Perceptron

    import seaborn as sns
    import matplotlib.pyplot as plt
    
    
    for ira in range(rank-start,rank+1):
        print('****20% sample test==',ira)
        #Ulsa = lsa.fit_transform(Mtrain.values/255)  #train version
        #print(total)
        if ira!=0:
            if ira<len(totalA.T):
                print("lsa dimmension reduction")                
                svd = TruncatedSVD(ira)
                normalizer = Normalizer(copy=False)
                lsa = make_pipeline(svd, normalizer)
                UlsA = lsa.fit_transform(totalA) #total version
                explained_variance = svd.explained_variance_ratio_.sum()
                print("Explained variance of the SVD step knowledge transfer: {}%".format(
                    int(explained_variance * 100)))            
            else:
                print("no reduction")
                UlsA=totalA
        else:
            print("3D-SVD dimmension reduction")
            u,s,vh=np.linalg.svd(totalA)
            print(u.shape, s.shape, vh.shape)
            UlsA=np.reshape(u, (len(totalA),28))            
        #    UlsA = totalA
        #    print("no LSA reduction")
        print('ulsa',UlsA.shape)


        #faze2
        #training model

        #sample
        samlen=int(len(Mlabel)/5)
        X_train, X_test, y_train, y_test = train_test_split(UlsA[:samlen], Mlabel[:samlen],stratify=Mlabel[:samlen], test_size=0.25)
        print("test on 20% sample")
        
        if klas=='Logi':
            classifiers = [
    #    SVC(kernel="rbf", C=0.025, probability=True),  20%
    #    NuSVC(probability=True),
                LogisticRegression(),
                 ]
        if klas=='Quad':
            classifiers = [
                QuadraticDiscriminantAnalysis(),
                 ]           
        if klas=='Rand':
            classifiers = [
                RandomForestClassifier(84),
                 ]               
        if klas=='Extr':
            classifiers = [
                ExtraTreesClassifier(verbose=1,n_jobs=3),
                 ]             
        if klas=='Adab':
            classifiers = [
                AdaBoostClassifier(),
                 ]            
        if klas=='Deci':
            classifiers = [
                DecisionTreeClassifier(),
                 ]
        if klas=='Grad':
            classifiers = [
                GradientBoostingClassifier(),
                 ]            
        if klas=='KNN':
            classifiers = [
                KNeighborsClassifier(n_jobs=4),  
                 ]            
        if klas=='Line':
            classifiers = [
                LinearDiscriminantAnalysis(), 
                 ]  
        if klas=='Gaus':
            classifiers = [
                GaussianNB(),
                 ] 
        if klas=='Perc':
            classifiers = [
                Perceptron(),
                 ]      
        if klas=='Elas':
            classifiers = [
                ElasticNet(random_state=0),
                 ]                 
    # Logging for Visual Comparison
        log_cols=["Classifier", "Accuracy", "Log Loss"]
        log = pd.DataFrame(columns=log_cols)
    
        for clf in classifiers:
            clf.fit(X_train,y_train)
            name = clf.__class__.__name__
        
            print("="*30)
            print(name)
            
            #print('****Results****')
            train_predictions = clf.predict(X_test)
            acc = accuracy_score(y_test, train_predictions)
            print("Accuracy: {:.4%}".format(acc))
        
            train_predictions = clf.predict_proba(X_test)
            ll = log_loss(y_test, train_predictions)
            print("Log Loss: {}".format(ll))
            
            log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
            log = log.append(log_entry)
    
        print("="*30)

    print('*** train complete set==',UlsA[:len(Mlabel)].shape)
     
    clf.fit(UlsA[:len(Mlabel)],Mlabel)
    #on complete trainset

    #pr2=pd.DataFrame(clf.predict_proba(Ulsa),index=list(range(0,len(Ulsa),1)))

    predictionA=pd.DataFrame(clf.predict(UlsA),columns=['pred'],index=range(0,len(UlsA)))
    predictionA[labelveld]=Mlabel 
    print('predict',predictionA.shape)
    predictionA.fillna(-1)
    predictionA['diff']=0
    predictionA['next']=Mlabel
    #abs(prediction[labelveld]-prediction['pred
    collist=sorted( Mlabel.label.unique() )

    print(collist)
    if klas=='Logi':
        predictionA[collist] = pd.DataFrame(clf.predict_log_proba(UlsA))
    if klas!='Logi':
        print(UlsA.shape)
        temp=pd.DataFrame(clf.predict_proba(UlsA))
        print(temp.shape)
        predictionA[collist]=temp
    
    from sklearn.metrics import classification_report, confusion_matrix
    true_labels=predictionA[labelveld][:len(Mtrain)].values.astype('float32')
    predicted_labels = predictionA['pred'][:len(Mtrain)].values.astype('float32')

    cm = confusion_matrix(true_labels, predicted_labels,labels=collist)
    print(classification_report(true_labels, predicted_labels))
    print("Confusion matrix")
    print(cm)
    
    corr=predictionA.drop(['pred','diff'],axis=1).corr()
    f, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(abs(corr), mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax)
    predictionA=predictionA.fillna('0')
    #print('Prediction',prediction.head())
    pred2=predictionA.drop(['pred',labelveld,'diff','next'],axis=1)
    
    print(predictionA.shape)


    return predictionA #['next']
no_of_normal_transcations = len(credit[credit['Class']==1])
no_of_fraud_transcations = len(credit[credit['Class']==0])
print("no_of_fraud_transcations:",no_of_normal_transcations)
print("no_of_OK_transcations:", no_of_fraud_transcations)
credit1=credit[credit['Class']==1]
for xi in range(40):
    credit=credit.append(credit1)
X = credit.iloc[:, 1:29].values
y = credit.iloc[:, 30].values
no_of_normal_transcations = len(credit[credit['Class']==1])
no_of_fraud_transcations = len(credit[credit['Class']==0])
print("no_of_fraud_transcations:",no_of_normal_transcations)
print("no_of_OK_transcations:", no_of_fraud_transcations)
Klasseer(X,X[:1000],y,'KNN',22,0) 

pred=Klasseer(X,X[:300],y,'Extr',28,0) 