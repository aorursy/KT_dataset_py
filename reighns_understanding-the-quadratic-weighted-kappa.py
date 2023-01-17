import numpy as np

import pandas as pd

from sklearn.metrics import confusion_matrix

from sklearn.metrics import cohen_kappa_score, make_scorer

import itertools

import sklearn

import seaborn as sns

import matplotlib.pyplot as plt

import random



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
y_true = pd.Series(['cat',  'cat', 'dog', 'cat',   'cat',  'cat', 'pig',  'pig', 'hen', 'pig'], name = 'Actual')

y_pred   = pd.Series(['bird', 'hen', 'pig','bird',  'bird', 'bird', 'pig', 'pig', 'hen', 'pig'], name = 'Predicted')



print("Ground truth:\n{}".format(y_true))

print("-"*40)

print("Predicted Values:\n{}".format(y_pred))
classes= ['bird','cat','dog','hen', 'pig']



# thank you https://datascience.stackexchange.com/questions/40067/confusion-matrix-three-classes-python 

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()

    

cnf_matrix = confusion_matrix(y_true, y_pred,labels=['bird','cat','dog','hen', 'pig'],)

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=['bird','cat','dog','hen', 'pig'],

                      title='Confusion matrix C, without normalization')
train = pd.read_csv("/kaggle/input/prostate-cancer-grade-assessment/train.csv")

y_true = train.isup_grade

y_true



y_pred = np.random.choice(6, 10616, replace=True)

y_pred = pd.Series(y_pred)

y_pred
cm = confusion_matrix(y_true, y_pred,labels=[0,1,2,3,4,5],)

plot_confusion_matrix(cm, classes=[0,1,2,3,4,5],

                      title='Confusion matrix C, without normalization')
# We construct the weighted matrix starting from a zero matrix, it is like constructing a 

# list, we usually start from an empty list and add things inside using loops.



def weighted_matrix(N):

    weighted = np.zeros((N,N)) 

    for i in range(len(weighted)):

        for j in range(len(weighted)):

            weighted[i][j] = float(((i-j)**2)/(N-1)**2) 

    return weighted

        

print(weighted_matrix(5))
## dummy example 

actual = pd.Series([2,2,2,3,4,5,5,5,5,5]) 

pred   = pd.Series([2,2,2,3,2,1,1,1,1,3]) 

C = confusion_matrix(actual, pred)



N=5

act_hist=np.zeros([N])

for item in actual: 

    act_hist[item - 1]+=1

    

pred_hist=np.zeros([N])

for item in pred: 

    pred_hist[item - 1]+=1

    



print(f'Actuals value counts:{act_hist}, \nPrediction value counts:{pred_hist}')
E = np.outer(act_hist, pred_hist)/10





E

C
# Method 1

# apply the weights to the confusion matrix

weighted = weighted_matrix(5)

num = np.sum(np.multiply(weighted, C))

# apply the weights to the histograms

den = np.sum(np.multiply(weighted, E))



kappa = 1-np.divide(num,den)

kappa
# Method 2



num=0

den=0

for i in range(len(weighted)):

    for j in range(len(weighted)):

        num+=weighted[i][j]*C[i][j]

        den+=weighted[i][j]*E[i][j]



weighted_kappa = (1 - (num/den)); weighted_kappa
# Method 3: Just use sk learn library



cohen_kappa_score(actual, pred, labels=None, weights= 'quadratic', sample_weight=None)