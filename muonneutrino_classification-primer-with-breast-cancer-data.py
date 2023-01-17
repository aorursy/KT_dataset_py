import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report





%matplotlib inline
data = pd.read_csv('../input/data.csv',index_col=0)

data = data.drop('Unnamed: 32',axis=1)

data.diagnosis = 1 * (data.diagnosis=='M')

data.head()
def plot_var(i,X):

    name = X.columns[i]

    plt.hist(X.iloc[:,i],bins=20)

    plt.xlabel(name)

    plt.ylabel('Number of Entries')

    plt.show()



for i in range(data.shape[1]):

    plot_var(i,data)
data.info()
data.describe()
def analyze(mod,X1,y1,X2,y2):

    mod.fit(X1,y1) 

    y1_pred = mod.predict(X1)

    y2_pred = mod.predict(X2)

    rep1 = classification_report(y1,y1_pred)

    rep2 = classification_report(y2,y2_pred)

    print('Training classification report:')

    print(rep1)

    print('Validation classification report:')

    print(rep2)
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



y = data['diagnosis']

X = data.drop('diagnosis',axis=1)

print(X.head())

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=124,

                                                 test_size=0.3,stratify=y)
lr = LogisticRegression(C=1e10)

# the large C removes regularization so that we can more easily compare the different models

# that we are using in this notebook without having regularization complicating our discussions.

analyze(lr,X_train,y_train,X_test,y_test)
from sklearn import metrics



def analyze_roc(mod,X1,y1,X2,y2):

    ytrain_pred = mod.predict_proba(X1)

    ytest_pred = mod.predict_proba(X2)

    # Results have probabilities for results [0,1]



    roc_train = metrics.roc_curve(y1,ytrain_pred[:,1])

    roc_test = metrics.roc_curve(y2,ytest_pred[:,1])



    roc_auc_train = metrics.roc_auc_score(y1,ytrain_pred[:,1])

    roc_auc_test = metrics.roc_auc_score(y2,ytest_pred[:,1])



    print('ROC Area-under-curve for training set: {:0.3}'.format(roc_auc_train))

    print('ROC Area-under-curve for validation set: {:0.3}'.format(roc_auc_test))





    fig = plt.figure(1,figsize=[6,6])

    plt.plot(roc_train[0],roc_train[1],label='Test set',c='b')

    plt.plot(roc_test[0],roc_test[1],label='Validation set',c='r')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.legend(loc='lower right')

    plt.show()

    

def analyze_prec_recall(mod,X1,y1,X2,y2):

    ytrain_pred = mod.predict_proba(X1)

    ytest_pred = mod.predict_proba(X2)

    # Results have probabilities for results [0,1]



    roc_train = metrics.precision_recall_curve(y1,ytrain_pred[:,1])

    roc_test = metrics.precision_recall_curve(y2,ytest_pred[:,1])



    roc_auc_train = metrics.average_precision_score(y1,ytrain_pred[:,1])

    roc_auc_test = metrics.average_precision_score(y2,ytest_pred[:,1])



    print('Ave. precision score for training set: {:0.3}'.format(roc_auc_train))

    print('Ave. precision score for validation set: {:0.3}'.format(roc_auc_test))





    fig = plt.figure(1,figsize=[6,6])

    plt.step(roc_train[1],roc_train[0],label='Test set',c='b')

    plt.step(roc_test[1],roc_test[0],label='Validation set',c='r')

    plt.xlabel('Recall')

    plt.ylabel('Precision')

    plt.legend(loc='lower left')

    plt.show()

    

analyze_roc(lr,X_train,y_train,X_test,y_test)

analyze_prec_recall(lr,X_train,y_train,X_test,y_test)
st_sc = StandardScaler()



X_train = st_sc.fit_transform(X_train)

X_test = st_sc.transform(X_test)
from sklearn.decomposition import PCA



pca = PCA()

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)
exp_var = pca.explained_variance_ratio_.cumsum()

print(exp_var)
analyze(lr,X_train[:,:],y_train,X_test[:,:],y_test)
analyze_roc(lr,X_train,y_train,X_test,y_test)

analyze_prec_recall(lr,X_train,y_train,X_test,y_test)
nfeat = 15



features = np.array([x for x in range(1,nfeat+1)])

train_score = np.zeros(nfeat)

val_score = np.zeros(nfeat)

train_score_2 = np.zeros(nfeat)

val_score_2 = np.zeros(nfeat)



for i in range(1,nfeat+1):

    #print('First {} PCA variable(s)\n\n'.format(i))

    lr.fit(X_train[:,:i],y_train)

    #print(i)

    train_prob = lr.predict_proba(X_train[:,:i])[:,1]

    test_prob = lr.predict_proba(X_test[:,:i])[:,1]

    #print(test_prob)

    

    train_score[i-1] = metrics.roc_auc_score(y_train,train_prob)

    val_score[i-1] = metrics.roc_auc_score(y_test,test_prob)

    

    train_score_2[i-1] = metrics.average_precision_score(y_train,train_prob)

    val_score_2[i-1] = metrics.average_precision_score(y_test,test_prob)

    

fig = plt.figure(1,figsize=[12,6])

ax = fig.add_subplot(121)

ax.plot(features,train_score,label='Training set',c='b')

ax.plot(features,val_score,label='Validation set',c='r')



ax.set_xlabel('Number of Features')

ax.set_ylabel('ROC AUC Score')

ax.set_title('Validation Plot for # of PCA Variables')

ax.legend(loc='lower right')

ax.set_ylim([0.9,1])

ax.grid(alpha=0.2,linestyle='-')



ax = fig.add_subplot(122)

ax.plot(features,train_score_2,label='Training set',c='b')

ax.plot(features,val_score_2,label='Validation set',c='r')



ax.set_xlabel('Number of Features')

ax.set_ylabel('Average Precision Score')

ax.set_title('Validation Plot for # of PCA Variables')

ax.legend(loc='lower right')

ax.set_ylim([0.9,1])

ax.grid(alpha=0.2,linestyle='-')



plt.show()
from sklearn import model_selection

from sklearn.pipeline import Pipeline

import warnings



# Get lots of warnings from exp for entries with very low probabilities

warnings.filterwarnings('ignore')



# Need to make my classifier to deal with shape of predict_proba output



class MyClassifier(LogisticRegression):

    def __init__(self,C=1e10,penalty='l2'):

        super(MyClassifier,self).__init__(C=C,penalty=penalty)

        

    def predict_proba(self,X):

        return super(MyClassifier,self).predict_proba(X)[:,1]

                                            

mylr = MyClassifier(C=1e10)

pca5 = PCA(n_components=5)



# Want to test all steps in the pipeline

my_pipeline = Pipeline([['Scaler',st_sc],['PCA',pca5],['LogReg',mylr]])

    

# Make scorers to pass to the learning_curve function

roc_scorer = metrics.make_scorer(metrics.roc_auc_score,needs_proba=True)

prec_scorer = metrics.make_scorer(metrics.average_precision_score,needs_proba=True)



# Create the learning curves

roc_size,roc_train,roc_test = model_selection.learning_curve(my_pipeline,X.as_matrix(),y,cv=8,

                                scoring = roc_scorer,train_sizes=np.linspace(0.1,1.0,20))

prec_size,prec_train,prec_test = model_selection.learning_curve(my_pipeline,X.as_matrix(),y,cv=8,

                                scoring = prec_scorer,train_sizes=np.linspace(0.1,1.0,20))



# Plot the curves

fig = plt.figure(1,figsize=[12,6])

ax = fig.add_subplot(121)



# Means and standard errors

roc_mean_train = np.mean(roc_train,axis=1)

roc_std_train = np.std(roc_train,axis=1)/np.sqrt(8)

roc_mean_test = np.mean(roc_test,axis=1)

roc_std_test = np.std(roc_test,axis=1)/np.sqrt(8)



ax.plot(roc_size,roc_mean_train,label='Training set',c='b')

ax.fill_between(roc_size,roc_mean_train-roc_std_train,

                roc_mean_train+roc_std_train,color='b',alpha=0.3)



ax.plot(roc_size,roc_mean_test,label='Validation set',c='r')

ax.fill_between(roc_size,roc_mean_test-roc_std_test,

                roc_mean_test+roc_std_test,color='r',alpha=0.3)



ax.set_xlabel('Number of Entries')

ax.set_ylabel('ROC AUC Score')

ax.set_title('Learning Curve for 5 PCA Features')

ax.legend(loc='lower right')

ax.set_ylim([0.9,1])

ax.grid(alpha=0.2,linestyle='-')





ax = fig.add_subplot(122)



# Means and standard errors

prec_mean_train = np.mean(prec_train,axis=1)

prec_std_train = np.std(prec_train,axis=1)/np.sqrt(8)

prec_mean_test = np.mean(prec_test,axis=1)

prec_std_test = np.std(prec_test,axis=1)/np.sqrt(8)



ax.plot(prec_size,prec_mean_train,label='Training set',c='b')

ax.fill_between(prec_size,prec_mean_train-prec_std_train,

                prec_mean_train+prec_std_train,color='b',alpha=0.3)



ax.plot(prec_size,prec_mean_test,label='Validation set',c='r')

ax.fill_between(prec_size,prec_mean_test-prec_std_test,

                prec_mean_test+prec_std_test,color='r',alpha=0.3)



ax.set_xlabel('Number of Entries')

ax.set_ylabel('Average Precision Score')

ax.set_title('Learning Curve for 5 PCA Features')

ax.legend(loc='lower right')

ax.set_ylim([0.9,1])

ax.grid(alpha=0.2,linestyle='-')



plt.show()



warnings.filterwarnings('default')
from sklearn.model_selection import validation_curve

warnings.filterwarnings('ignore')



# Transform the full dataset with my pca transformation

st_sc.fit(X)

X_feat = st_sc.transform(X)

pca.fit(X_feat)

X_feat = pca.transform(X_feat)



mylr = MyClassifier(C=1e10,penalty='l1')



c_val = [0.01,0.1,1,10,100,1000,10000]



roc_train,roc_test = model_selection.validation_curve(mylr,X_feat,y,'C',c_val,cv=8,

                                scoring = roc_scorer)

prec_train,prec_test = model_selection.validation_curve(mylr,X_feat,y,'C',c_val,cv=8,

                                scoring = prec_scorer)



# Plot the curves

fig = plt.figure(1,figsize=[12,6])

ax = fig.add_subplot(121)



# Means and standard errors

roc_mean_train = np.mean(roc_train,axis=1)

roc_std_train = np.std(roc_train,axis=1)/np.sqrt(8)

roc_mean_test = np.mean(roc_test,axis=1)

roc_std_test = np.std(roc_test,axis=1)/np.sqrt(8)



ax.plot(c_val,roc_mean_train,label='Training set',c='b')

ax.fill_between(c_val,roc_mean_train-roc_std_train,

                roc_mean_train+roc_std_train,color='b',alpha=0.3)



ax.plot(c_val,roc_mean_test,label='Validation set',c='r')

ax.fill_between(c_val,roc_mean_test-roc_std_test,

                roc_mean_test+roc_std_test,color='r',alpha=0.3)



ax.set_xlabel('Inverse Regularization Strength')

ax.set_ylabel('ROC AUC Score')

ax.set_title('Validation Curve for L1 Regression')

ax.legend(loc='lower right')

ax.set_ylim([0.9,1])

ax.grid(alpha=0.2,linestyle='-')

ax.set_xscale('log')



ax = fig.add_subplot(122)



# Means and standard errors

prec_mean_train = np.mean(prec_train,axis=1)

prec_std_train = np.std(prec_train,axis=1)/np.sqrt(8)

prec_mean_test = np.mean(prec_test,axis=1)

prec_std_test = np.std(prec_test,axis=1)/np.sqrt(8)



ax.plot(c_val,prec_mean_train,label='Training set',c='b')

ax.fill_between(c_val,prec_mean_train-prec_std_train,

                prec_mean_train+prec_std_train,color='b',alpha=0.3)



ax.plot(c_val,prec_mean_test,label='Validation set',c='r')

ax.fill_between(c_val,prec_mean_test-prec_std_test,

                prec_mean_test+prec_std_test,color='r',alpha=0.3)



ax.set_xlabel('Inverse Regularization Strength')

ax.set_ylabel('Average Precision Score')

ax.set_title('Validation Curve for L1 Regression')

ax.legend(loc='lower right')

ax.set_ylim([0.9,1])

ax.set_xscale('log')

ax.grid(alpha=0.2,linestyle='-')



plt.show()



warnings.filterwarnings('default')
lr_l1 = LogisticRegression(C=10,penalty='l1')

lr_l1.fit(X_feat,y)

print('C=10')

print(np.where(lr_l1.coef_[0]==0)[0])



lr_l1 = LogisticRegression(C=1,penalty='l1')

lr_l1.fit(X_feat,y)

print('C=1')

print(np.where(lr_l1.coef_[0]==0)[0])



lr_l1 = LogisticRegression(C=0.1,penalty='l1')

lr_l1.fit(X_feat,y)

print('\nC=0.1')

print(np.where(lr_l1.coef_[0]==0)[0])

from sklearn.feature_selection import RFE



st_sc_rfe = StandardScaler()



X_feat_rfe = st_sc_rfe.fit_transform(X_feat)



rfe = RFE(lr,n_features_to_select=5)

X_feat_rfe = rfe.fit_transform(X_feat_rfe,y)



print(rfe.ranking_)