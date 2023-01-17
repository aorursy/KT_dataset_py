import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
train=pd.read_csv("../input/heart.csv")
train.describe().T


from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC,SVR
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import PassiveAggressiveClassifier,Perceptron,LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier,MLPRegressor,BernoulliRBM
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline, make_union
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin
from sklearn.svm import LinearSVR,SVC
from sklearn.utils import check_array


class StackingEstimator(BaseEstimator, TransformerMixin):
    
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self
    def transform(self, X):
        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))

        # add class prodiction as a synthetic feature
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

        return X_transformed
    
Classifiers = [
               #Perceptron(n_jobs=-1),
               #SVR(kernel='rbf',C=1.0, epsilon=0.2),
               CalibratedClassifierCV(LinearDiscriminantAnalysis(), cv=4, method='sigmoid'),    
               #OneVsRestClassifier( SVC(    C=50,kernel='rbf',gamma=1.4, coef0=1,cache_size=3000,)),
               KNeighborsClassifier(10),
               DecisionTreeClassifier(),
               RandomForestClassifier(n_estimators=200),
               ExtraTreesClassifier(n_estimators=250,random_state=0), 
               OneVsRestClassifier(ExtraTreesClassifier(n_estimators=10)) , 
               MLPClassifier(alpha=0.510,activation='logistic'),
               LinearDiscriminantAnalysis(),
               #OneVsRestClassifier(GaussianNB()),
               AdaBoostClassifier(),
               #GaussianNB(),
               QuadraticDiscriminantAnalysis(),
               SGDClassifier(average=True,max_iter=100),
               XGBClassifier(max_depth=5, base_score=0.005),
               LogisticRegression(C=1.0,multi_class='multinomial',penalty='l2', solver='saga',n_jobs=-1),
               LabelPropagation(n_jobs=-1),
               LinearSVC(),
               #MultinomialNB(alpha=.01),    
                   make_pipeline(
                    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
                    StackingEstimator(estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=3, max_features=0.55, min_samples_leaf=18, min_samples_split=14, subsample=0.7)),
                    AdaBoostClassifier()
                ),

              ]
def klasseer(e_,mtrain,mtest,veld,idvld,thres,probtrigger):
    # e_ total matrix without veld, 
    # veld the training field
    #thres  threshold to select features
    label = mtrain[veld]
    # select features find most relevant ifo threshold
    clf = ExtraTreesClassifier(n_estimators=100)
    ncomp=e_.shape[1]-2
    model = SelectFromModel(clf, prefit=True,threshold =(thres)/100)
       # SVD
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=ncomp, n_iter=7, random_state=42)
    e_=svd.fit_transform(e_)
    
       #tsne not used
    from sklearn.manifold import TSNE
    #e_=TSNE(n_components=3).fit_transform(e_)
    #from sklearn.metrics.pairwise import cosine_similarity
    
       #robustSVD not used
    #A_,e1_,e_,s_=robustSVD(e_,140)
    clf = clf.fit( e_[:len(mtrain)], label)
    New_features = model.transform( e_[:len(mtrain)])
    Test_features= model.transform(e_[-len(mtest):])
    pd.DataFrame(New_features).plot.scatter(x=0,y=1,c=mtrain[veld]+1)
    pd.DataFrame(np.concatenate((New_features,Test_features))).plot.scatter(x=0,y=1,c=['r' for x in range(len(mtrain))]+['g' for x in range(len(mtest))])    

    print('Model with threshold',thres/100,New_features.shape,Test_features.shape,e_.shape)
    print('____________________________________________________')
    
    Model = []
    Accuracy = []
    for clf in Classifiers:
        #train
        fit=clf.fit(New_features,label)
        pred=fit.predict(New_features)
        Model.append(clf.__class__.__name__)
        Accuracy.append(accuracy_score(mtrain[veld],pred))
        #predict
        sub = pd.DataFrame({idvld: mtest[idvld],veld: fit.predict(Test_features)})
        sub.plot(x=idvld,kind='kde',title=clf.__class__.__name__ +str(( mtrain[veld]==pred).mean()) +'prcnt') 
        sub2=pd.DataFrame(pred,columns=[veld])
        #estimate sample if  accuracy
        if veld in mtest.columns:
            print( clf.__class__.__name__ +str(round( accuracy_score(mtrain[veld],pred),2)*100 )+'prcnt accuracy versus unknown',(sub[veld]==mtest[veld]).mean() )
        #write results
        klassnaam=clf.__class__.__name__+".csv"
        sub.to_csv(klassnaam, index=False)
        if probtrigger:
            pred_prob=fit.predict_proba(Test_features)
            sub=pd.DataFrame(pred_prob)
    return sub
train=train.reset_index()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('target',axis=1),train['target'], test_size=0.3, random_state=42)
totaal=(X_train.append(X_test)).fillna(0).values

subx=klasseer(totaal,(X_train.T.append(y_train.T)).T,(X_test.T.append(y_test.T)).T,'target','index',3,False)