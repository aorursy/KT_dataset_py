# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", '../input']).decode("utf8"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/column_2C_weka.csv")
df.head()
df.describe().T
X = df.drop("class", axis=1)
y = df["class"]
le = LabelEncoder()
le.fit(y)
y = le.transform(y)
df1 = X
df1['class'] = y
df3C = pd.read_csv("../input/column_3C_weka.csv")
df3C.head()
df3C.info()
X3C = df3C.drop("class", axis=1)
y3C = df3C["class"]
le3C = LabelEncoder()
le3C.fit(y3C)
y3C = le3C.transform(y3C)
df3C1 = X3C
df3C1['class'] = y3C
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
            UlsA=np.reshape(u, (len(totalA),28*28))            
        #    UlsA = totalA
        #    print("no LSA reduction")
        print('ulsa',UlsA.shape)


        #faze2
        #training model

        #sample
        samlen=int(len(Mlabel)/1)
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
Klasseer(X3C,X3C,y3C,'KNN',7,0) 
Klasseer(X3C,X3C,y3C,'Grad',7,0)
Klasseer(X3C,X3C,y3C,'Extr',7,0)
Klasseer(X3C,X3C,y3C,'Rand',7,0)
Klasseer(X3C,X3C,y3C,'Adab',7,0) 
Klasseer(X3C,X3C,y3C,'Gaus',7,0)
Klasseer(X3C,X3C,y3C,'Deci',7,0) 
Klasseer(X3C,X3C,y3C,'Logi',7,0)
Klasseer(X3C,X3C,y3C,'Line',7,0) 
#Klasseer(X3C,X3C,y3C,'Quad',7,0) 