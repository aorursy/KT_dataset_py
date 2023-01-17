# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt



import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'

matplotlib.rcParams['font.family'] = 'sans-serif'

matplotlib.rcParams['font.size'] = 12
# preprocessing libraries

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder



# model selection libraries

from sklearn.model_selection import train_test_split



# data decomposition libraries

from sklearn.decomposition import PCA



# machine learning libraries

from sklearn.tree import DecisionTreeClassifier



# postprocessing and checking-results libraries

from sklearn.metrics import balanced_accuracy_score

from sklearn.metrics import confusion_matrix
def plotConfusionMatrix(dtrue,dpred,classes,title = 'Confusion Matrix',\

                        width = 0.75,cmap = plt.cm.Blues):

  

    cm = confusion_matrix(dtrue,dpred)

    cm = cm.astype('float') / cm.sum(axis = 1)[:,np.newaxis]



    fig,ax = plt.subplots(figsize = (np.shape(classes)[0] * width,\

                                       np.shape(classes)[0] * width))

    im = ax.imshow(cm,interpolation = 'nearest',cmap = cmap)



    ax.set(xticks = np.arange(cm.shape[1]),

           yticks = np.arange(cm.shape[0]),

           xticklabels = classes,

           yticklabels = classes,

           title = title,

           aspect = 'equal')

    

    ax.set_ylabel('True',labelpad = 20)

    ax.set_xlabel('Predicted',labelpad = 20)



    plt.setp(ax.get_xticklabels(),rotation = 90,ha = 'right',

             va = 'center',rotation_mode = 'anchor')



    fmt = '.2f'



    thresh = cm.max() / 2.0



    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j,i,format(cm[i,j],fmt),ha = 'center',va = 'center',

                    color = 'white' if cm[i,j] > thresh else 'black')

    plt.tight_layout()

    plt.show()
df = pd.read_csv('../input/nutrient-analysis-of-pizzas/Pizza.csv')

print(df)
X = df.drop(['brand','id'],axis = 1)

y = df['brand']
print(np.asarray(X.columns))
scaler = StandardScaler().fit(X)

Xnorm = scaler.transform(X)

print(Xnorm)
pca = PCA(n_components = 0.98)

pca.fit(Xnorm)



Xpca = pca.transform(Xnorm)

ypca = LabelEncoder().fit_transform(y)



n_components = pca.explained_variance_ratio_.shape[0]



print('Cumulative variance ratio: {}'.format(np.cumsum(pca.explained_variance_ratio_)))

print('Principal components: {:d}'.format(n_components))
fig,ax = plt.subplots(figsize = (6.0,5.0))

ax.bar(np.arange(n_components),pca.explained_variance_ratio_,color = 'orange',edgecolor = 'black')

ax.set_ylabel('Variance Ratio',labelpad = 10)

ax.set_xticks(np.arange(n_components))

ax.set_xticklabels(['PC-{:d}'.format(i + 1) for i in np.arange(0,n_components)])

bx = ax.twinx()

bx.scatter(np.arange(n_components),np.cumsum(pca.explained_variance_ratio_),\

           s = 100,c = 'cyan',edgecolor = 'black',lw = 0.75)

bx.set_ylabel('Cumulative Variance Ratio',labelpad = 10)

plt.tight_layout()

plt.show()
fig,ax = plt.subplots(1,3,figsize = (15.0,5.0))



ax[0].scatter(Xpca[:,0],Xpca[:,1],c = ypca,s = 100,cmap = plt.cm.plasma,edgecolor = 'black',lw = 0.25)

ax[0].set_xlabel('PC-1')

ax[0].set_ylabel('PC-2')



ax[1].scatter(Xpca[:,0],Xpca[:,2],c = ypca,s = 100,cmap = plt.cm.plasma,edgecolor = 'black',lw = 0.25)

ax[1].set_xlabel('PC-1')

ax[1].set_ylabel('PC-3')



ax[2].scatter(Xpca[:,1],Xpca[:,2],c = ypca,s = 100,cmap = plt.cm.plasma,edgecolor = 'black',lw = 0.25)

ax[2].set_xlabel('PC-2')

ax[2].set_ylabel('PC-3')



plt.tight_layout()



plt.show()
Xtrain,Xtest,ytrain,ytest = train_test_split(Xpca,ypca,test_size = 0.30,random_state = 21)

clf = DecisionTreeClassifier(random_state = 21).fit(Xtrain,ytrain)



ypred = clf.predict(Xtest)



print('Model accuracy: {:.3f}'.format(balanced_accuracy_score(ytest,ypred)))
plotConfusionMatrix(ytest,ypred,classes = y.unique(),cmap = plt.cm.binary)