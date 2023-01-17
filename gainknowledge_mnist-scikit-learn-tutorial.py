import matplotlib as mpl
import matplotlib.pyplot as plt

#%config InlineBackend.figure_formats = {'pdf',}
%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
%time dfTrain = pd.read_csv('../input/train.csv')
dfTrain.head()
y = dfTrain['label'].values.flatten() 
y
x = dfTrain.drop(['label'],axis=1).values 
x
#x = x/255.0
x
from sklearn.model_selection import train_test_split
x_train,x_dev, y_train,  y_dev = train_test_split(x,y,random_state=42)
def displayData(X,Y):
    # set up array
    fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(15,15))
    fig.suptitle( "Display randomly images of the training data set")
    # loop over randomly drawn numbers
    for i in range(10):
        for j in range(10):
            ind = np.random.randint(X.shape[0])
            tmp = X[ind,:].reshape(28,28)
            ax[i,j].set_title("Label: {}".format(Y[ind]))
            ax[i,j].imshow(tmp, cmap='gray_r') # display it as gray colors.
            plt.setp(ax[i,j].get_xticklabels(), visible=False)
            plt.setp(ax[i,j].get_yticklabels(), visible=False)
    
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

displayData(x_train,y_train)    
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
ax.hist(y,bins=[0,1,2,3,4,5,6,7,8,9,10], edgecolor="b", histtype="bar",align='left')
ax.set_title('Histogram: Training data set')
ax.set(xlabel='Number', ylabel='Frequency')
ax.xaxis.set_ticks([0,1,2,3,4,5,6,7,8,9] );
ax.axhline(y=(y.size/10), label="average frequency",linestyle='dashed',   color='r')
ax.legend()
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
proj = pca.fit_transform(x)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
ax.set_title('PCA of dataset n_components=2')
ax.scatter(proj[:,0],proj[:,1],c=y, label='number')
ax.legend()
%time dfTest = pd.read_csv('../input/test.csv')
dfTest.head()
dfTrain.describe()
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import scipy

clf = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(1000,),  random_state=1 ) 
best_model = clf.fit(x/255.0, y)
dfTrain['label_predicted'] =  best_model.predict(x/255.0)
dfTrain['valid_predicted'] = dfTrain.apply(lambda row: row['label_predicted']==row['label'], axis=1)
print('Training Accuracy: {:3.2f} %'.format(best_model.score(x,y)*100))
y_test_pred = best_model.predict(x)

from sklearn.metrics import confusion_matrix
import seaborn as sn
confusion_matrix = confusion_matrix(y, y_test_pred)
fig, ax = plt.subplots(figsize=(10,10))

sn.heatmap(confusion_matrix, annot=True, ax=ax, fmt='g',vmin=0)
ax.set_ylabel("true label")
ax.set_xlabel("predicted label")
def displayData(labeld_as, index,X,Y,Y_Pred):
    nImages = index.values.size 
    nRows = 1+ nImages//10
    # set up array
    fig, ax = plt.subplots(nrows=nRows, ncols=10,squeeze=False,figsize=(10+9*.5, 1+nRows+0.5*nRows))
    fig.suptitle( "Labeld as {} predicted otherwise".format(labeld_as))
    # loop over randomly drawn numbers
    for i in range(nRows):
        for j in range(10):
            pos = i*10+j
            if (pos<nImages):
                tmp = X[index[pos],:].reshape(28,28)
                ax[i,j].set_title("I:{}, P:{}".format(index[pos],Y_Pred[index[pos]]))
                ax[i,j].imshow(tmp, cmap='gray_r') # display it as gray colors.
                plt.setp(ax[i,j].get_xticklabels(), visible=False)
                plt.setp(ax[i,j].get_yticklabels(), visible=False)

    fig.subplots_adjust(hspace=0.5, wspace=0.5)
   
for i in range(10):
    index = dfTrain[(dfTrain['valid_predicted']==False) & (dfTrain['label']==i)].index
    if (index.size>0):
        displayData(i,index,x,y,y_test_pred)
# load the test data set.
dfTest = pd.read_csv('../input/test.csv')
x_test = dfTest.values
y_test = best_model.predict(x_test/255.)
dfExport = pd.DataFrame( {'ImageId':range(1,y_test.size+1),'Label': y_test})
dfExport.to_csv('prediction.csv',index=False)   
