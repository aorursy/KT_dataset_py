# Importing libraries
import re
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.options.display.max_columns = 50
%matplotlib inline
warnings.filterwarnings("ignore")
sns.set_style("darkgrid")
# Importing the dataset
df = pd.read_csv('../input/Kaagle_Upload.csv', low_memory=False)[:152000]
# Giving the dimension information
print('Dataframe dimensions:', df.shape)

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
               #CalibratedClassifierCV(LinearDiscriminantAnalysis(), cv=4, method='sigmoid'),    
               #OneVsRestClassifier( SVC(    C=50,kernel='rbf',gamma=1.4, coef0=1,cache_size=3000,)),
               #KNeighborsClassifier(10),
               #DecisionTreeClassifier(),
               RandomForestClassifier(n_estimators=200),
               ExtraTreesClassifier(n_estimators=250,random_state=0), 
               #OneVsRestClassifier(ExtraTreesClassifier(n_estimators=10)) , 
               #MLPClassifier(alpha=0.510,activation='logistic'),
               #LinearDiscriminantAnalysis(),
               #OneVsRestClassifier(GaussianNB()),
               #AdaBoostClassifier(),
               #GaussianNB(),
               #QuadraticDiscriminantAnalysis(),
               #SGDClassifier(average=True,max_iter=100),
               XGBClassifier(max_depth=5, base_score=0.005),
               #LogisticRegression(C=1.0,multi_class='multinomial',penalty='l2', solver='saga',n_jobs=-1),
               #LabelPropagation(n_jobs=-1),
               #LinearSVC(),
               #MultinomialNB(alpha=.01),    
               #    make_pipeline(
               #     StackingEstimator(estimator=LassoLarsCV(normalize=True)),
               #     StackingEstimator(estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=3, max_features=0.55, min_samples_leaf=18, min_samples_split=14, subsample=0.7)),
               #     AdaBoostClassifier()
               # ),

              ]


import numpy as np
from numpy.linalg import norm, svd
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
#svd = TruncatedSVD(n_components=int(disc.shape[1]-1), n_iter=7, random_state=42)


def robustSVD(X,n_comp,lmbda=.01, tol=1e-3, maxiter=100, verbose=True):
    
    svd = TruncatedSVD(n_components=n_comp, n_iter=10, random_state=42)
    """
    Inexact Augmented Lagrange Multiplier
    """
    Y = X
    norm_two = norm(Y.ravel(), 2)
    norm_inf = norm(Y.ravel(), np.inf) / lmbda
    dual_norm = np.max([norm_two, norm_inf])
    Y = Y / dual_norm
    A = np.zeros(Y.shape)
    E = np.zeros(Y.shape)
    dnorm = norm(X, 'fro')
    mu = 1.25 / norm_two
    rho = 1.5
    sv = 10.
    n = Y.shape[0]
    itr = 0
    while True:
        Eraw = X - A + (1 / mu) * Y
        Eupdate = np.maximum(Eraw - lmbda / mu, 0) + np.minimum(Eraw + lmbda / mu, 0)
        #U, S, V = svd(X - Eupdate + (1 / mu) * Y, full_matrices=False)
        X_n=svd.fit_transform(X - Eupdate + (1 / mu) * Y)
        S=svd.singular_values_
        if itr ==0:
            pd.DataFrame(svd.explained_variance_ratio_*100).plot()
        svp = (S > 1 / mu).shape[0]
        print(S.sum())
        if svp < sv:
            sv = np.min([svp + 1, n])
        else:
            sv = np.min([svp + round(.05 * n), n])
        #Aupdate = np.dot(np.dot(U[:, :svp], np.diag(S[:svp] - 1 / mu)), V[:svp, :])
        Aupdate = svd.inverse_transform(X_n)
        A = Aupdate
        E = Eupdate
        Z = X - A - E
        Y = Y + mu * Z
        mu = np.min([mu * rho, mu * 1e7])
        itr += 1
        
        if ((norm(Z, 'fro') / dnorm) < tol) or (itr >= maxiter):
            u=svd.components_
            pd.DataFrame(svd.explained_variance_ratio_*100).plot()
            pd.DataFrame(A[:33,:20]).plot()
            pd.DataFrame(E[:33,:20]).plot()
            break
    if verbose:
        print("Finished at iteration %d" % (itr))  
    return A, E, X_n, S

# the description explains nothing... bizar ?
### so its clear that police reports don't explain the cause of the accident.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.fillna(' ').reset_index(),df['accident_severity'], test_size=0.3, random_state=42)

def classy(mtrain,mtest,mkolom,veld,idvld,thres,probtrigger):
    mtrain=mtrain[mkolom+[veld]]
    mtest=mtest[mkolom+[veld]]
    from sklearn import preprocessing
    from sklearn.metrics import roc_curve, auc,recall_score,precision_score,average_precision_score
    

    print('train',mtrain.shape,'test',mtest.shape)
    # Label Encoding - Target 
    print ("Label Encode strings... ")
    from sklearn.preprocessing import LabelEncoder
    #encode train veld
    lb = LabelEncoder()
    label = lb.fit_transform(mtrain[veld])
    #encode all categorial fields
    def label_encoding(col):
        le = LabelEncoder()
        le.fit(list(mtrain[col].values) + list(mtest[col].values))
        #print(mtrain[col])
        mtrain[col] = le.transform(mtrain[col].astype(str))
        mtest[col] = le.transform(mtest[col].astype(str))
    features=mtrain.columns
    num_cols = mtrain._get_numeric_data().columns 
    cat_cols = list(set(features) - set(num_cols))
    for col in cat_cols:
        print(col)
        label_encoding(col)
        
    e_=mtrain[mkolom].append(mtest[mkolom])
    #scale
    from sklearn.preprocessing import MinMaxScaler
    mms = MinMaxScaler()
    e_ = mms.fit_transform(e_)
    #svd
    svdcol=mtrain.shape[1]-35  #number of  components for SVD   
    from sklearn.decomposition import TruncatedSVD
    from scipy.sparse.linalg import svds, eigs
    #from scipy.sparse import vstack
    #e_, s, vt = svds(e_, k=svdcol)  #vstack for sparse matrix
    e_,e1_,u_,s_=robustSVD(e_,svdcol)
    e_,s,vt =np.linalg.svd(e_,full_matrices=False)
    vt=pd.DataFrame(vt,index=mkolom)
    #print(vt)
    featx=pd.DataFrame(s,columns=['sing'])
    featx['descr']=''
    featx['perc']=featx['sing']/featx ['sing'].sum()*100
    
    for xi in vt.columns:
        featx.iat[xi,1]=list(vt.sort_values(xi,ascending=False)[:3].index)
    print(featx)
    #svd = TruncatedSVD(n_components=svdcol, n_iter=7, random_state=42)
    #e_=svd.fit_transform(e_)  #sparse
    #A_,er_,e_,s_=robustSVD(e_.values,svdcol+1)  #robust
    #e_=svd.fit_transform(e_)
    print('SVD',e_.shape)#,e_)
    
    #
    #find most relevant features
    clf = ExtraTreesClassifier(n_estimators=100)
    model = SelectFromModel(clf, prefit=True,threshold =(thres)/100)    
    #clf = clf.fit( e_[:len(mtrain)], label)
    #New_features = model.transform( e_[:len(mtrain)])
    #Test_features= model.transform(e_[-len(mtest):])
    #New_features= e_[:len(mtrain)]
    #Test_features=e_[-len(mtest):]
 
    pd.DataFrame(e_[:len(mtrain)]).plot.scatter(x=0,y=1,c=mtrain[veld]+1,title='Train SVD classes')
    pd.DataFrame(np.concatenate((e_[-len(mtest):],e_[:len(mtrain)]))).plot.scatter(x=0,y=1,c=[1 for x in range(len(mtest))]+[2 for x in range(len(mtrain))],colormap='viridis',title='Train-2 versus Test-1 SVD')
    print('Model with threshold',thres/100,e_[:len(mtrain)].shape,e_[-len(mtest):].shape,e_.shape)
    print('____________________________________________________')
    
    Model = []
    Accuracy = []
    for clf in Classifiers:
        fitI=clf.fit(e_[:len(mtrain)],label)
        pred=fitI.predict(e_[:len(mtrain)])

        #print(pred_prob)
        Model.append(clf.__class__.__name__)
        Accuracy.append( (mtrain[veld]==pred).mean() )
        #print('Accuracy of '+clf.__class__.__name__ +' is '+str(accuracy_score(train[veld],pred)))
        #prediction of test
        predicty=lb.inverse_transform(fitI.predict(e_[-len(mtest):]))
        print(predicty)
        sub = pd.DataFrame({idvld: mtest[idvld],veld: predicty})
        #sub.plot(x=idvld,kind='kde',title=clf.__class__.__name__ +str((label==pred).mean()) +'prcnt') 
        #score = average_precision_score(label==label, pred==label)  # works only binary
        #print('train score: {:.6f}'.format(score))
        feat=pd.DataFrame(featx.descr)
        feat['importance']=clf.feature_importances_
        feat=feat.sort_values(by=['importance'],ascending=False)
        print(feat.head())
        feat.plot.barh(x=0,y='importance')
        sub2=pd.DataFrame(pred,columns=[veld])
        if veld in mtest.columns:
            print( clf.__class__.__name__ +str((label==pred).mean() )+' accuracy versus unknown',(sub[veld]==mtest[veld]).mean() )
        else:
            print( clf.__class__.__name__ +str((label==pred).mean() ) )
        klassnaam=clf.__class__.__name__+".csv"
        sub.to_csv(klassnaam, index=False)
        if probtrigger:
            pred_prob=fitI.predict_proba(Test_features)
            sub=pd.DataFrame(pred_prob)
    return sub
#kolom=[x for x in X_train.columns if x not in ['Fatal','Date Of Stop','Model','Time Of Stop','Description','Geolocation','Location','Latitude','Longitude']]
kolom=[x for x in X_train.columns if x not in ['accident_severity']]
subx=classy(X_train,X_test,kolom,'accident_severity','index',0.3,False)
df['accident_severity']