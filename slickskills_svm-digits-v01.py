%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train_df=pd.read_csv('../input/train.csv')

test_df=pd.read_csv('../input/test.csv')
train_df.info()

test_df.info()

test_df.head()
def display(data, indices=[1], label=False):

    im_ar=[]

    labels=[]

    ncols=10

    nrows=int((len(indices)-1e-6)/ncols)+1

    fig, axes=plt.subplots(nrows, ncols)

    axes_=axes.reshape(nrows*ncols,)

    for i,ax in zip(indices, axes_):

        img= data.ix[i,:].reshape(28,28)

        img=255-img.reshape(28,28)

        ax.imshow(img,cmap='Greys_r')

        im_ar.append(img)

        ax.set_xticks([])

        ax.set_yticks([])



display(train_df.drop(['label'], axis=1), indices=range(0,100,1))
from sklearn import decomposition

ncomps=20

img_df=train_df.drop('label', axis=1)

pca = decomposition.PCA(n_components=ncomps)

pca.fit(img_df)
pca_score = pca.explained_variance_ratio_

V = pca.components_

n_pca=list(range(ncomps))

plt.bar(n_pca, pca_score)

plt.ylabel('Explained Variance')

plt.xlabel('PCA Component')

plt.title('Number of Components :'+ str(len(n_pca))+ ', Cumulative Variance: '+ str( '{:.2f}'.format(np.sum(pca_score))))
#Display PCA variances



pca_comps=pd.DataFrame(V)

display(pca_comps, range(ncomps))
# Starting to differentiate using PCA components

img_pca=pca.transform(img_df)

def pca_to_df(img_pca):

    pca_df=pd.DataFrame()

    for n in range(img_pca.shape[1]):

        var='pca_'+str(n+1)

        pca_df[var]=img_pca.transpose()[n]

    return pca_df

pca_df=pca_to_df(img_pca)

pca_df['label']=train_df['label']

sns.lmplot(x='pca_1', y='pca_2', data=pca_df[:5000], hue='label', fit_reg=False)
pca_df.head()
from sklearn.cross_validation import train_test_split

def accuracy(ytest, ypred):

    pred=pd.DataFrame()

    pred['label']=ytest

    pred['Predicted']=ypred

    pred['Accuracy']=1*(pred.label==pred.Predicted)

    pred = pred.groupby('label').mean()

    return pred.drop(['Predicted'], axis=1)



def extract_pca(data_df):

    """Returns X and Y after PCA"""

    data_pca=pca.transform(data_df.drop(['label'], axis=1))

    X=pca_to_df(data_pca)

    X.index=data_df.index

    Y=data_df['label']

    return X,Y



def train_test(data=train_df):

    m=list(range(len(data)))

    tr, te=train_test_split(m)

    train_split, test_split=data.ix[tr, :], data.ix[te, :]

    return train_split, test_split





def predict(clf, train_df=train_df, test_df=test_df, pca=True):

    """Returns predicted values"""

    if pca:

        Xtrain, Ytrain =train_test_pca(train_df)

        Xtest, Ytest = train_test_pca(test_df)

    else:

        Xtrain, Ytrain = train_df.drop(['label'], axis=1), train_df['label']

        Xtest, Ytest = test_df.drop(['label'], axis=1), test_df['label']

    clf.fit(Xtrain, Ytrain)

    return clf.predict(Xtest)

    
#Cross Validation splitting

train_split, test_split = train_test(train_df)

Xtrain,Ytrain=extract_pca(train_split)

Xtest, Ytest = extract_pca(test_split)
# Cross validation to determine optimum k

from sklearn.neighbors import KNeighborsClassifier

ks=[1,5,11,21,51,101]

test_accuracy=[]

for k in ks:

    neigh = KNeighborsClassifier(n_neighbors=k)

    neigh.fit(Xtrain, Ytrain)

    ypred = neigh.predict(Xtest)

    pred=accuracy(Ytest, ypred)

    test_accuracy.append(pred.mean())
# k=5 turns out to be the best

plt.plot(ks[:5], test_accuracy[:5])
#Testing k=5

k=5

neigh = KNeighborsClassifier(n_neighbors=k)

neigh.fit(Xtrain, Ytrain)

ypred = neigh.predict(Xtest)
#Results for k=5

pred=accuracy(Ytest, ypred)

print (pred)

print (pred.mean())
from sklearn.svm import SVC
#Try with 1000 samples first

train_split, test_split = train_test(train_df[:2000])

Xtrain,Ytrain=extract_pca(train_split)

Xtest, Ytest = extract_pca(test_split)
ac=[]

for c in [ 0.1, 1, 10, 100]:

    for g in [5e-7, 1e-6, 2e-6]:

        clf=SVC(C=c, gamma=g)

        clf.fit(Xtrain,Ytrain)

        svm_pred=clf.predict(Xtest)

        ac.append([c,g,accuracy(Ytest, svm_pred).mean()])

ac
# Now try all samples

train_split, test_split = train_test(train_df)

Xtrain,Ytrain=extract_pca(train_split)

Xtest, Ytest = extract_pca(test_split)

svm=SVC(C=1, gamma=1e-6)

svm.fit(Xtrain,Ytrain)

svm_pred=svm.predict(Xtest)
svm_accuracy=accuracy(Ytest, svm_pred)

print (svm_accuracy)

print (svm_accuracy.mean())
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=200)

clf.fit(Xtrain,Ytrain)

rand_pred=clf.predict(Xtest)
clf
rf_accuracy=accuracy(Ytest, rand_pred)

print (rf_accuracy)

print (rf_accuracy.mean())
test_pca=pca.transform(test_df)

test_pca=pca_to_df(test_pca)
test_pred=svm.predict(test_pca)
test_df['Label']=test_pred
display(test_df.drop(['Label'], axis=1), indices=range(0,100,1))
test_df['ImageId']=test_df.index+1
test_df[['ImageId','Label']].to_csv('submission.csv', index=False)