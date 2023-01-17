# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#train = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')[:200000]

def loaddata():

    from sklearn import preprocessing



    train_tr = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID')

    test_tr = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')



    train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID')

    test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv', index_col='TransactionID')

    print('files read and merging')

    #sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')

    

    train = train_identity.merge(train_tr, how='left', left_index=True, right_index=True)

    test = test_identity.merge(test_tr, how='left', left_index=True, right_index=True)

    del train_tr, train_identity, test_tr, test_identity

    

    total=train.append(test)

    del test

    print('train,total',train.shape,total.shape)

    y_label = train['isFraud']

    del train

    coltype=total.dtypes

    # Drop target, fill in NaNs

    print('labelencoding')

    # Label Mean Encoding

    #sparse>> total = pd.get_dummies(total, columns=total.columns, sparse=True).sparse.to_coo().tocsr()



    for ci in total.columns:

        if (coltype[ci]=="object"):

            #print(ci)

            codes=total[[ci,'isFraud']].groupby(ci).mean().sort_values("isFraud")

            codesdict=codes.isFraud.to_dict()

            #print(codes)

            total[ci]=total[ci].map(codesdict) #l_enc.transform(totaal[ci])            

            

    #fill target with 50%

    total.drop('isFraud',axis=1)

    total['isFraud']=np.concatenate((y_label,[0.5 for x in range(len(total)-len(y_label))]),axis=0)

    return total.astype('float32').fillna(0),y_label



total=loaddata()

label=total[1]

total[0].shape,label.shape

total[0].head()
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

def sigmoid(z):

    return 1 / (1 + np.exp(-z))



class classGS():  

      

    #defining constructor  

    def __init__(self, clf=[KNeighborsClassifier(10)],thres=0.1,probtrigger=False,ncomp=5,neighU=5,ncompU=5,midiU=0.3,veld='label',idvld='index',lentrain=30000,scale=True):  

        self.clf2=clf

        self.thres=thres

        self.probtrigger=probtrigger

        self.ncomp=ncomp

        self.neighU=neighU

        self.ncompU=ncompU

        self.midiU=midiU

        self.lentrain=lentrain

        self.veld=veld

        self.idvld=idvld

        self.scale=scale

        

    def get_params(self, deep=True):

        # suppose this estimator has parameters "alpha" and "recursive"

        return {"clf":self.clf2,'thres':self.thres,'probtrigger':self.probtrigger,'ncomp':self.ncomp,'neighU':self.neighU,'ncompU':self.ncompU,'midiU':self.midiU,'lentrain':self.lentrain}



    def set_params(self, **parameters):

        for parameter, value in parameters.items():

            setattr(self, parameter, value)

        return self

    

    def fit(self, e__,mtrain_):

        # e__= pd.dataframe with veld and ivld as labels

        # mtrain = pd.DataFrame() with veld as label field 

        from umap import UMAP

        from sklearn.decomposition import PCA

        from sklearn import preprocessing

        min_max_scaler = preprocessing.MinMaxScaler()





        #klasseerGS(e_,mtrain,mtest,veld,idvld,thres,probtrigger,ncomp,neighU,ncompU,midiU):

        mtest=pd.DataFrame( e__[self.lentrain:].index,columns=[self.idvld] )

        mtest[self.veld]=mtrain_[self.lentrain:][self.veld]

        label = mtrain_[:self.lentrain][self.veld]

        print('1:',e__.shape,mtrain_.shape,label.shape,self.lentrain,e__[self.lentrain:].shape)

        if self.scale:

            e__ = min_max_scaler.fit_transform(e__)

            e__ = sigmoid(e__) 

        

        e__ = PCA(n_components=self.ncomp).fit_transform(e__)

        e__ = UMAP(n_neighbors=self.neighU,n_components=self.ncompU, min_dist=self.midiU,metric='minkowski').fit_transform(e__)

        self.e__=e__

        

        pd.DataFrame(e__[:self.lentrain]).plot(x=0,y=1,c=mtrain_[:self.lentrain][self.veld]+1,kind='scatter',title='classesplot',colormap ='jet')

        pd.DataFrame(e__).plot.scatter(x=0,y=1,c=['r' for x in range(self.lentrain)] +['g' for x in range(len(e__[self.lentrain:]))])   

        print('2 Model with threshold',self.thres/1000,mtrain_[:self.lentrain].shape,e__.shape,self.ncomp,self.neighU,self.ncompU,self.midiU,)

    

        for clf2 in self.clf2:

            #train

            fit=clf2.fit(e__[:self.lentrain],label)

            print(fit)



            #if clf.__class__.__name__=='DecisionTreeClassifier':

                #treeprint(clf)

            pred=fit.predict(e__)

            #Model.append(self.clf2.__class__.__name__)

            #Accuracy.append(accuracy_score(mtrain[:self.lentrain][self.veld],pred))

            #predict

            print('3:',self.idvld,self.veld,len(mtest))

            self.sub = pd.DataFrame({self.idvld: mtest[self.idvld],self.veld: pred[-len(mtest):]})

            self.sub.plot(x=self.idvld,kind='kde',title=clf2.__class__.__name__ +str(( mtrain_[:self.lentrain][self.veld]==pred[:self.lentrain]).mean()) +'prcnt') 

            sub2=pd.DataFrame(pred,columns=[self.veld])



            #estimate sample if  accuracy

            if False:# self.veld in mtest.columns:

                print( clf2.__class__.__name__ +str(round( accuracy_score(mtrain_[:self.lentrain][self.veld],pred[:self.lentrain]),2)*100 )+'prcnt accuracy versus unknown',(mtrain_[self.lentrain:][self.veld]==pred[self.lentrain:]).mean() )

                from sklearn.metrics import confusion_matrix

                print(confusion_matrix(mtrain_[self.lentrain:][self.veld],pred[self.lentrain:]))

                #write results

            if self.probtrigger:

                pred_prob=fit.predict_proba(e__[-len(mtest):])

                sub=pd.DataFrame(pred_prob)        

        #defining class methods  

            self.f1score=((mtrain_[self.veld]==pred[:len(mtrain_)]).mean())

            print('4:f1',self.f1score)

            self.treshold_=pred



            print(self.sub.shape)

        return self

        

    def _meaning(self, _e1):

        # returns True/False according to fitted classifier

        # notice underscore on the beginning

        print('meaning')

        return( True if _e1 >= self.treshold_ else True )



    def predict(self, e__, mtrain_):

        try:

            getattr(self, "treshold_")

        except AttributeError:

            raise RuntimeError("You must train classifer before predicting data!")

        print('predict',e__.shape,mtrain_.shape)

        return(self.treshold_)



    def score(self, e__, mtrain_):

        # counts number of values bigger than mean

        print('score',self.e__.shape,mtrain_.shape,self.f1score)

        return(self.f1score) 

  
    cGS=classGS(ncomp=30,midiU=0.1,ncompU=7,neighU=5,lentrain=144233,veld='isFraud',scale=True)

    result=cGS.fit(total[0].drop('isFraud',axis=1).reset_index(),pd.DataFrame( total[0]['isFraud'][:144233] ))
from __future__ import division, print_function



import numpy as np



try:

    from pylab import plt

except ImportError:

    print('Unable to import pylab. R_pca.plot_fit() will not work.')



try:

    # Python 2: 'xrange' is the iterative version

    range = xrange

except NameError:

    # Python 3: 'range' is iterative - no need for 'xrange'

    pass





class R_pca:



    def __init__(self, D, mu=None, lmbda=None):

        self.D = D

        self.S = np.zeros(self.D.shape)

        self.Y = np.zeros(self.D.shape)



        if mu:

            self.mu = mu

        else:

            self.mu = np.prod(self.D.shape) / (4 * self.frobenius_norm(self.D))



        self.mu_inv = 1 / self.mu



        if lmbda:

            self.lmbda = lmbda

        else:

            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))



    @staticmethod

    def frobenius_norm(M):

        return np.linalg.norm(M, ord='fro')



    @staticmethod

    def shrink(M, tau):

        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))



    def svd_threshold(self, M, tau):

        U, S, V = np.linalg.svd(M, full_matrices=False)

        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))



    def fit(self, tol=None, max_iter=1000, iter_print=100):

        iter = 0

        err = np.Inf

        Sk = self.S

        Yk = self.Y

        Lk = np.zeros(self.D.shape)



        if tol:

            _tol = tol

        else:

            _tol = 1E-7 * self.frobenius_norm(self.D)



        while (err > _tol) and iter < max_iter:

            Lk = self.svd_threshold(

                self.D - Sk + self.mu_inv * Yk, self.mu_inv)

            Sk = self.shrink(

                self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)

            Yk = Yk + self.mu * (self.D - Lk - Sk)

            err = self.frobenius_norm(self.D - Lk - Sk)

            iter += 1

            if (iter % iter_print) == 0 or iter == 1 or iter > max_iter or err <= _tol:

                print('iteration: {0}, error: {1}'.format(iter, err))



        self.L = Lk

        self.S = Sk

        return Lk, Sk



    def plot_fit(self, size=None, tol=0.1, axis_on=True):



        n, d = self.D.shape



        if size:

            nrows, ncols = size

        else:

            sq = np.ceil(np.sqrt(n))

            nrows = int(sq)

            ncols = int(sq)

        nrows=10

        ncols =1

        ymin = np.nanmin(self.D)

        ymax = np.nanmax(self.D)

        print('ymin: {0}, ymax: {1}'.format(ymin, ymax))



        numplots = np.min([n, nrows * ncols])

       

        plt.figure()



        for n in range(numplots):

            plt.subplot(nrows, ncols, n + 1)

            plt.ylim((ymin - tol, ymax + tol))

            plt.plot(self.L[n, :] + self.S[n, :], 'r')

            plt.plot(self.L[n, :], 'b')

            if not axis_on:

                plt.axis('off')
# use R_pca to estimate the degraded data as L + S, where L is low rank, and S is sparse

rpca = R_pca(total[0].values)

L, S = rpca.fit(max_iter=1000, iter_print=100)

total[0][total[0].isFraud==1]
Spd=pd.DataFrame(S[:,:-1])

Spd['isFraud']=total[0]['isFraud'].values



cGS=classGS(ncomp=30,midiU=0.1,ncompU=7,neighU=5,lentrain=144233,veld='isFraud',scale=True)

result=cGS.fit(Spd.drop('isFraud',axis=1).reset_index(),pd.DataFrame( total[0]['isFraud'][:144233] ))

predi=result.predict(Spd.drop('isFraud',axis=1).reset_index(),pd.DataFrame( total[0]['isFraud'][:144233] ) )

predi
test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv', index_col='TransactionID')
test_identity['isFraud']=predi[144233:]
test_tr = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')


submit=test_tr[['TransactionDT']].merge(test_identity[['isFraud']], how='left', left_index=True, right_index=True)

subm = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')
submit[['isFraud']].reset_index().fillna(0).to_csv('sample_submission.csv',index=False)