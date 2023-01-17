import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import pandas as pd
import random
import itertools
import seaborn as sns

sns.set(style = 'darkgrid')
% matplotlib inline
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
bc = pd.read_csv('../input/data.csv')
bc.head(1)
new_columns = bc.columns.values; new_columns[18] = 'concavepoints_worst'; bc.columns = new_columns
#Volume worst estimation of cancer create a estimate worst 3D structure
bc = bc.drop("Unnamed: 32",1)
temp = np.log(bc.radius_worst*bc.area_worst)
bc['Volume_ln'] = temp.values
#
temp = np.log(bc.concavepoints_worst*bc.concavity_worst*bc.compactness_worst+1)
bc['Concave_ln'] = temp.values
#cancer fractal- symmetry  divided by volume
temp = -np.log(bc.fractal_dimension_worst*bc.symmetry_worst/np.log(bc.radius_mean*bc.area_mean))
bc['FractVol_ln'] = temp.values
# all unrelated
temp = np.log(bc.radius_worst*bc.perimeter_worst*bc.concavepoints_worst+1)
bc['RaPeCo_ln'] = temp.values
bcs = pd.DataFrame(preprocessing.scale(bc.ix[:,2:36]))
bcs.columns = list(bc.ix[:,2:36].columns)
bcs['diagnosis'] = bc['diagnosis']
from pandas.tools.plotting import scatter_matrix
mbc = pd.melt(bcs, "diagnosis", var_name="measurement")
fig, ax = plt.subplots(figsize=(10,5))
p = sns.violinplot(ax = ax, x="measurement", y="value", hue="diagnosis", split = True, data=mbc, inner = 'quartile', palette = 'Set2');
p.set_xticklabels(rotation = 90, labels = list(bcs.columns));
sns.swarmplot(x = 'diagnosis', y = 'Volume_ln',palette = 'Set2', data = bcs);
sns.swarmplot(x = 'diagnosis', y = 'RaPeCo_ln',palette = 'Set2', data = bcs);
sns.jointplot(x = bc['RaPeCo_ln'], y = bc['Volume_ln'], stat_func=None, color="#4CB391", edgecolor = 'w', size = 6);
X = bcs.ix[:,0:30]

y = bcs['diagnosis']
class_names = list(y.unique())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=42)
svc = SVC(kernel = 'linear',C=.1, gamma=10, probability = True)
svc.fit(X,y)
y_pred = svc.fit(X_train, y_train).predict(X_test)
t = pd.DataFrame(svc.predict_proba(X_test))
svc.score(X_train,y_train), svc.score(X_test, y_test)
mtrx = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision = 2)

plt.figure()
plot_confusion_matrix(mtrx,classes=class_names,title='Confusion matrix, without normalization')

plt.figure()
plot_confusion_matrix(mtrx, classes=class_names, normalize = True, title='Normalized confusion matrix')

plt.show()
def Klasseer(Mtrain,Mtest,Mlabel,klas,rank,start):
    #data preparation
    #print(Mtotal)
    #Mtotal=Mtotal.fillna(-1)
    #print(Mtotal)
    #Mtrain=Mtotal[Mtotal[labelveld]!=-1]
    #Mtest=Mtotal[Mtotal[labelveld]==-1]
    #Mtest=Mtest.drop(labelveld,axis=1)
    Mlabel=pd.DataFrame( Mlabel,columns=['diagnosis'] )  #[:len(Mtrain)]
    #Mlabel=Mlabel.fillna(-1)  
    labelveld='diagnosis'
    print('shapes train',Mtrain.shape,labelveld,Mlabel.shape,'test',Mtest.shape)

    
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
                ElasticNetCV(random_state=0),
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
    collist=sorted( Mlabel[labelveld].unique() )

    print(collist)
    if klas=='Logi':
        predictionA[collist] = pd.DataFrame(clf.predict_log_proba(UlsA))
    if klas!='Logi':
        print(UlsA.shape)
        temp=pd.DataFrame(clf.predict_proba(UlsA))
        print(temp.shape)
        predictionA[collist]=temp
    
    from sklearn.metrics import classification_report, confusion_matrix
    true_labels=predictionA[labelveld][:len(Mtrain)] #.values.astype('float32')
    predicted_labels = predictionA['pred'][:len(Mtrain)] #.values.astype('float32')

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
#bcs = pd.DataFrame(preprocessing.scale(bc.ix[:,2:36]))
#bcs.columns = list(bc.ix[:,2:36].columns)
#bcs['diagnosis'] = bc['diagnosis']
#print(bc.describe().T)
Klasseer(bc.iloc[:,2:36],bc.iloc[:,2:36],bc['diagnosis'],'Logi',11,0) 
Klasseer(bc.iloc[:,2:36],bc.iloc[:,2:36],bc['diagnosis'],'KNN',11,0) 
Klasseer(bc.iloc[:,2:36],bc.iloc[:,2:36],bc['diagnosis'],'Quad',11,0) 
Klasseer(bc.iloc[:,2:36],bc.iloc[:,2:36],bc['diagnosis'],'Gaus',11,0) 
Klasseer(bc.iloc[:,2:36],bc.iloc[:,2:36],bc['diagnosis'],'Line',11,0) 
Klasseer(bc.iloc[:,2:36],bc.iloc[:,2:36],bc['diagnosis'],'Grad',11,0) 
Klasseer(bc.iloc[:,2:36],bc.iloc[:,2:36],bc['diagnosis'],'Adab',11,0) 
Klasseer(bc.iloc[:,2:36],bc.iloc[:,2:36],bc['diagnosis'],'Rand',11,0) 
#Klasseer(bc.iloc[:,2:36],bc.iloc[:,2:36],bc['diagnosis'],'Elas',11,0) 
Klasseer(bc.iloc[:,2:36],bc.iloc[:,2:36],bc['diagnosis'],'Extr',11,0) 