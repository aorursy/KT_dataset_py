import numpy as np
import pandas as pd
import os
import xgboost as xgb
train = pd.read_csv('../input/train/train.csv')
test = pd.read_csv('../input/test/test.csv')
train.shape,test.shape
sub = pd.read_csv('../input/test/sample_submission.csv')

y = train.AdoptionSpeed.values
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
vectorizer = CountVectorizer() #min_df=0.01)
#vectorizer =TfidfVectorizer(min_df=0.001)
X = vectorizer.fit_transform(train.Description.append(test.Description).fillna(' '))
#X=pd.DataFrame(X.todense(),columns=vectorizer.get_feature_names())
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=10, n_iter=7, random_state=42)
u=svd.fit_transform(X)  
vt=svd.fit_transform(X.T)  
Xi=np.linalg.pinv(X*vt)
#X=np.dot(X,vt)
X.shape,vt.shape,Xi.shape
def cohen_effect_size(X, y):
    """Calculates the Cohen effect size of each feature.
    
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target vector relative to X
        Returns
        -------
        cohen_effect_size : array, shape = [n_features,]
            The set of Cohen effect values.
        Notes
        -----
        Based on https://github.com/AllenDowney/CompStats/blob/master/effect_size.ipynb
    """
    print(X.shape,y.shape,y.mean())
    medi=y.mean()
    group1, group2 = X[y<medi], X[y>=medi]
    diff = group1.mean() - group2.mean()
    var1, var2 = group1.var(), group2.var()
    n1, n2 = group1.shape[0], group2.shape[0]
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    d = diff / np.sqrt(pooled_var)
    return d
excluded_feats = ['Name', 'RescuerID', 'Description', 'PetID','AdoptionSpeed'] #['SK_ID_CURR']
features = [f_ for f_ in train.columns if f_ not in excluded_feats]

print('Number of features %d  ' % (len(features)),train.shape,train.AdoptionSpeed.shape) 
#effect_sizes = cohen_effect_size(Xtrain[:len(ytrain)], ytrain)
effect_sizes = cohen_effect_size(train[features],train.AdoptionSpeed.values)
effect_sizes.reindex(effect_sizes.abs().sort_values(ascending=False).nlargest(50).index)[::-1].plot.barh(figsize=(6, 10));
print('Features with the 30 largest effect sizes')
significant_features = [f for f in features if np.abs(effect_sizes.loc[f]) > 0.1]
print('Significant features %d: %s' % (len(significant_features), significant_features))

excluded_feats = ['Name', 'RescuerID', 'Description', 'PetID','AdoptionSpeed'] #['SK_ID_CURR']
features = [f_ for f_ in train.columns if f_ not in excluded_feats]
features = [f_ for f_ in pd.DataFrame(u).columns if f_ not in excluded_feats]
print('Number of features %d  ' % (len(features)),u[:len(train)].shape,train.AdoptionSpeed.shape) 
#effect_sizes = cohen_effect_size(Xtrain[:len(ytrain)], ytrain)
effect_sizes = cohen_effect_size(pd.DataFrame(u[:len(train)]),train.AdoptionSpeed.values)
effect_sizes.reindex(effect_sizes.abs().sort_values(ascending=False).nlargest(50).index)[::-1].plot.barh(figsize=(6, 10));
print('Features with the 30 largest effect sizes')
significant_features = [f for f in features if np.abs(effect_sizes.loc[f]) > 0.1]
print('Significant features %d: %s' % (len(significant_features), significant_features))

excluded_feats = ['Name', 'RescuerID', 'Description', 'PetID','AdoptionSpeed'] #['SK_ID_CURR']
features = [f_ for f_ in train.columns if f_ not in excluded_feats]
features = [f_ for f_ in pd.DataFrame(Xi.T).columns if f_ not in excluded_feats]
print('Number of features %d  ' % (len(features)),u[:len(train)].shape,train.AdoptionSpeed.shape) 
#effect_sizes = cohen_effect_size(Xtrain[:len(ytrain)], ytrain)
effect_sizes = cohen_effect_size(pd.DataFrame(Xi.T[:len(train)]),train.AdoptionSpeed.values)
effect_sizes.reindex(effect_sizes.abs().sort_values(ascending=False).nlargest(50).index)[::-1].plot.barh(figsize=(6, 10));
print('Features with the 30 largest effect sizes')
significant_features = [f for f in features if np.abs(effect_sizes.loc[f]) > 0.1]
print('Significant features %d: %s' % (len(significant_features), significant_features))

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, VotingClassifier,ExtraTreesClassifier,GradientBoostingRegressor, AdaBoostClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import PassiveAggressiveClassifier,Perceptron,LogisticRegression, RidgeClassifier,SGDClassifier,ElasticNetCV, LassoLarsCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier,MLPRegressor,BernoulliRBM
from sklearn.svm import SVC,LinearSVC,SVR
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import make_pipeline, make_union
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin
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
               Perceptron(n_jobs=-1),
               RidgeClassifier(tol=1e-2, solver="lsqr"),
               #SVR(kernel='rbf',C=1.0, epsilon=0.2),
               CalibratedClassifierCV(LinearDiscriminantAnalysis(), cv=4, method='sigmoid'),    
               #OneVsRestClassifier( SVC(    C=50,kernel='rbf',gamma=1.4, coef0=1,cache_size=3000,)),
               KNeighborsClassifier(10),
               DecisionTreeClassifier(),
               #RandomForestClassifier(n_estimators=200),
               ExtraTreesClassifier(n_estimators=250,random_state=0), 
               OneVsRestClassifier(ExtraTreesClassifier(n_estimators=10)) , 
               MLPClassifier(alpha=0.510,activation='logistic'),
               LinearDiscriminantAnalysis(),
               OneVsRestClassifier(GaussianNB()),
               AdaBoostClassifier(),
               GaussianNB(),
               QuadraticDiscriminantAnalysis(),
               SGDClassifier(average=True,max_iter=100),
               LogisticRegression(C=1.0,multi_class='multinomial',penalty='l2', solver='saga',n_jobs=-1),
               XGBClassifier(max_depth=5, base_score=0.005),
               #LabelPropagation(n_jobs=-1),
               #LinearSVC(),
               #MultinomialNB(alpha=.01),    

              ]
def klasseer(e_,mtrain,label,mtest,veld,idvld,thres,probtrigger):
    # e_ total matrix without veld, 
    # veld the training field
    #thres  threshold to select features
    velden=[v for v in e_.columns if v not in [veld,idvld]]
    #label = mtrain[veld]
    #print(e_.shape,velden)
    e_=e_.loc[:,velden]
    print(e_.shape)
    # select features find most relevant ifo threshold
    #clf = ExtraTreesClassifier(n_estimators=100)
    ncomp=e_.shape[1]-18
    #model = SelectFromModel(clf, prefit=True,threshold =(thres)/100)
       # SVD
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import Normalizer
    svd = TruncatedSVD(n_components=ncomp, n_iter=7, random_state=42)
    e_=svd.fit_transform(Normalizer().fit_transform(e_))
    print('SVD expl var',svd.explained_variance_[:10])
       #tsne not used
    from sklearn.manifold import TSNE
    e_=TSNE(n_components=3,perplexity=5).fit_transform(e_)
    #from sklearn.metrics.pairwise import cosine_similarity
    
       #robustSVD not used
    #A_,e1_,e_,s_=robustSVD(e_,140)
    #clf = clf.fit( e_[:len(mtrain)], label)
    #New_features = model.transform( e_[:len(mtrain)])
    #Test_features= model.transform(e_[-len(mtest):])
    New_features=e_[:len(mtrain)]
    Test_features=e_[len(mtrain):]
    pd.DataFrame(New_features).plot.scatter(x=0,y=1,c=label,colormap='Spectral')
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
        Accuracy.append(accuracy_score(label,pred))
        #predict
        sub = pd.read_csv('../input/test/sample_submission.csv')
        #sub = pd.DataFrame({'PetID': mtest[idvld],veld: fit.predict(Test_features)})
        sub[veld]=fit.predict(Test_features)
        #sub.plot(x=idvld,kind='kde',title=clf.__class__.__name__ +str(( mtrain[veld]==pred).mean()) +'prcnt') 
        sub2=pd.DataFrame(pred,columns=[veld])
        #estimate sample if  accuracy
        if veld in mtest.columns:
            print( clf.__class__.__name__ +str(round( accuracy_score(label,pred),2)*100 )+'prcnt accuracy versus unknown',(sub[veld]==mtest[veld]).mean() )
        else:
            print(clf.__class__.__name__ +str(round( accuracy_score(label,pred),2)*100 ))
        #write results
        klassnaam=clf.__class__.__name__+".csv"
        #sub.to_csv(klassnaam, index=False)
        sub.to_csv('submission.csv', index=False)
        if probtrigger:
            pred_prob=fit.predict_proba(Test_features)
            sub=pd.DataFrame(pred_prob)
    return sub
excluded_feats = ['Name', 'RescuerID', 'Description', 'PetID','AdoptionSpeed'] #['SK_ID_CURR']
features = [f_ for f_ in train.columns if f_ not in excluded_feats]
features0=features+['PetID','AdoptionSpeed']
features1=features+['PetID']
print(features0)
totaal=(train.append(test)).fillna(0)
cotrain=pd.DataFrame( np.hstack((u[:len(train)],Xi.T[:len(train)],train[features1])) )
cotest=pd.DataFrame( np.hstack((u[len(train):],Xi.T[len(train):],test[features1])) )
totaal=pd.DataFrame( np.hstack((u,Xi.T,train[features].append(test[features]).fillna(0) )))
subx=klasseer(totaal,cotrain,train.AdoptionSpeed.values,cotest,'AdoptionSpeed',36,3,False)

train = train.drop(['Name', 'RescuerID', 'Description', 'PetID', 'AdoptionSpeed'], axis=1).values
test = test.drop(['Name', 'RescuerID', 'Description', 'PetID'], axis=1).values
clf = xgb.XGBClassifier(n_estimators=100, nthread=-1, max_depth=8, learning_rate=0.05)
clf.fit(np.hstack((u[:len(train)],Xi[:len(train)],train)), y)
preds = clf.predict(np.hstack((u[len(train):],Xi[len(train):],test)))

#sample = pd.read_csv('../input/test/sample_submission.csv')
#sample.AdoptionSpeed = preds
#sample.to_csv('submission.csv', index=False)
