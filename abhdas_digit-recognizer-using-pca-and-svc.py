import numpy as np

import scipy as sp

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D



from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
train = pd.read_csv('../input/train.csv')

train.shape
#Using first 4000 datapoints for now, because entire data set is too large

train_trunc = train.head(4000)



X = train_trunc.drop('label', axis = 1)

y = train_trunc['label']

y.hist()

plt.xlabel('Digits')

plt.ylabel('Number of Samples')

plt.title('Distribution of Digits in subset of dataset')

plt.show()
#histogram shows balanced classes in the first 4000 data points used.

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=55)



#scaling is optional as all pixel values are in the same range

ss = StandardScaler()

ss.fit(X_train)

X_train = ss.transform(X_train)

X_test = ss.transform(X_test)
#Dimensionality reduction using PCA

pca = PCA(n_components = 100)

pca.fit(X_train)

X_train_pca = pca.transform(X_train)

X_test_pca = pca.transform(X_test)



plt.plot(pca.explained_variance_ratio_*100)

plt.xlabel('$n$-th PCA component')

plt.ylabel('Percentage of Explained Variance')

plt.title('Percentage of Explained Variance per PCA component')

plt.show()

plt.plot(pca.explained_variance_ratio_.cumsum()*100)

plt.xlabel('$n$-th PCA component')

plt.ylabel('Percentage of Explained Variance')

plt.title('Cumulative Percentage of Explained Variance')

plt.show()

print('Total Explained Variance: %2.2f %%'%(pca.explained_variance_ratio_.sum()*100))
#Reducing from 784 to 100 features using PCA gives us ~ 80% of explained variance

#To visualize separation of classes, t-SNE is used

tsne = TSNE(n_components = 2)

X_train_tsne_pca = tsne.fit_transform(X_train_pca)
#Plotting the classes after t-SNE

plt.figure(figsize=(11,11))

colors = ['r','g','b','y','m','c','gray','navy','tan','brown']

for i in range(10):

    plt.scatter(X_train_tsne_pca[y_train[:] == i,0],X_train_tsne_pca[y_train[:] == i,1],color=colors[i],marker='.',

               alpha=0.75,s=100)

leg = {'0':'r', 1:'g',2:'b',3:'y',4:'m',5:'c',6:'gray',7:'navy',8:'tan',9:'brown'}

plt.xlabel('t-SNE-1',fontsize=15)

plt.ylabel('t-SNE-2',fontsize=15)

plt.title('Distribution of Classes after t-SNE on PCA reduced dataset',fontsize=15)

plt.legend(leg,fontsize=15)

plt.show()
#To improve class separation, dimensions are further reduced using LDA



lda = LinearDiscriminantAnalysis(n_components = 9)

lda.fit(X_train_pca,y_train)



X_train_lda = lda.transform(X_train_pca)

X_test_lda = lda.transform(X_test_pca)



plt.plot(lda.explained_variance_ratio_*100)

plt.xlabel('$n$-th LDA component')

plt.ylabel('Percentage of Explained Variance')

plt.title('Percentage of Explained Variance per LDA component')

plt.show()

plt.plot(lda.explained_variance_ratio_.cumsum()*100)

plt.xlabel('$n$-th LDA component')

plt.ylabel('Percentage of Explained Variance')

plt.title('Cumulative Percentage of Explained Variance')

plt.show()

print('Total Explained Variance: %2.2f %%'%(lda.explained_variance_ratio_.sum()*100))
#To visualize separation of classes, t-SNE is used

tsne = TSNE(n_components = 2)

X_train_tsne_lda = tsne.fit_transform(X_train_lda)
#Plotting the classes after t-SNE

plt.figure(figsize=(11,11))

colors = ['r','g','b','y','m','c','gray','navy','tan','brown']

for i in range(10):

    plt.scatter(X_train_tsne_lda[y_train[:] == i,0],X_train_tsne_lda[y_train[:] == i,1],color=colors[i],marker='.',

               alpha=0.75,s=100)

leg = {'0':'r', 1:'g',2:'b',3:'y',4:'m',5:'c',6:'gray',7:'navy',8:'tan',9:'brown'}

plt.xlabel('t-SNE-1',fontsize=15)

plt.ylabel('t-SNE-2',fontsize=15)

plt.title('Distribution of Classes after t-SNE on PCA followed by LDA reduced dataset',fontsize=15)

plt.legend(leg,fontsize=15)

plt.show()
#fitting a SVC model with RBF kernel with the LDA reduced dataset

param_grid = {'C':np.logspace(-4,2,base=10,num=7),'gamma':np.logspace(-6,-1,base=10,num=6)}

cv_svc_rbf_lda = GridSearchCV(SVC(kernel='rbf'),param_grid = param_grid,cv=5)

cv_svc_rbf_lda.fit(X_train_lda,y_train)

d = cv_svc_rbf_lda.best_params_

print('Optimum C: ',d.get('C'))

print('Optimum gamma: ', d.get('gamma'))

print('Test Score: %0.3f ' %cv_svc_rbf_lda.score(X_test_lda,y_test))

print('Train Score: %0.3f' %cv_svc_rbf_lda.score(X_train_lda,y_train))
#fitting a SVC model with RBF kernel with the PCA reduced dataset

param_grid = {'C':np.logspace(-4,2,base=10,num=7),'gamma':np.logspace(-6,-1,base=10,num=6)}

cv_svc_rbf_pca = GridSearchCV(SVC(kernel='rbf'),param_grid = param_grid,cv=5)

cv_svc_rbf_pca.fit(X_train_pca,y_train)

d = cv_svc_rbf_pca.best_params_

print('Optimum C: ',d.get('C'))

print('Optimum gamma: ', d.get('gamma'))

print('Test Score: %0.3f ' %cv_svc_rbf_pca.score(X_test_pca,y_test))

print('Train Score: %0.3f' %cv_svc_rbf_pca.score(X_train_pca,y_train))
'''

Although LDA gives better class separation (visualized by t-SNE), PCA is performing better.

For the final model, SVC is used with a RBF kernel with C = 10.0, gamma = 0.001

The whole dataset is used.

'''

X = train.drop('label', axis = 1)

y = train['label']

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=55)



ss = StandardScaler()

ss.fit(X_train)

X_train = ss.transform(X_train)

X_test = ss.transform(X_test)



pca = PCA(n_components = 145) 

pca.fit(X_train)

X_train_pca = pca.transform(X_train)

X_test_pca = pca.transform(X_test)



plt.plot(pca.explained_variance_ratio_*100)

plt.xlabel('$n$-th PCA component')

plt.ylabel('Percentage of Explained Variance')

plt.title('Percentage of Explained Variance per PCA component')

plt.show()

plt.plot(pca.explained_variance_ratio_.cumsum()*100)

plt.xlabel('$n$-th PCA component')

plt.ylabel('Percentage of Explained Variance')

plt.title('Cumulative Percentage of Explained Variance')

plt.show()

print('Total Explained Variance: %2.2f %%'%(pca.explained_variance_ratio_.sum()*100))
#Here 145 PCA components used to get Explained Variance ~ 80%

svc = SVC(kernel='rbf',C = 10.0, gamma=0.001)

svc.fit(X_train_pca,y_train)

y_predict = svc.predict(X_test_pca)



print('Test Score: %0.3f ' %svc.score(X_test_pca,y_test))

print('Train Score: %0.3f ' %svc.score(X_train_pca,y_train))



confusion = pd.DataFrame(confusion_matrix(y_test,y_predict),index = [i for i in '0123456789'], columns =

                        [i for i in '0123456789'])

plt.figure(figsize=(15,15))

sns.heatmap(confusion,annot=True)

plt.xlabel('Predicted Label',fontsize=15)

plt.ylabel('True Label',fontsize=15)

plt.title('Confusion Matrix - Test Set',fontsize=15)

plt.show()
test = pd.read_csv('../input/test.csv')

test = ss.transform(test)

test_pca = pca.transform(test)

y_submit = svc.predict(test_pca)

submission = pd.DataFrame()

submission['ImageId'] = np.linspace(1,test.shape[0],num=test.shape[0],dtype=int)

submission['Label'] = y_submit

submission.to_csv('submission.csv',index=False)