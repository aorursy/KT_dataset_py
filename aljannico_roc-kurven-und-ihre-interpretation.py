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

# Read some arff data



def read_data(filename):

    f = open(filename)

    data_line = False

    data = []

    for l in f:

        l = l.strip() # get rid of newline at the end of each input line

        if data_line:

            content = [float(x) for x in l.split(',')]

            if len(content) == 3:

                data.append(content)

        else:

            if l.startswith('@DATA'):

                data_line = True

    return data



d = read_data("../input/train.arff") # our data



import matplotlib.pyplot as plt

def plot_datapoints(data):

    x1,y1 = zip(*[(x,y) for x,y,c in data if c == 1])

    x2,y2 = zip(*[(x,y) for x,y,c in data if c == -1])

    plt.plot(x1, y1, 'o')

    plt.plot(x2, y2, 'o')



plot_datapoints(d);
# We may want to split the data in training and validation sets

# (depends on your choice to an approach)

np.random.seed(1000) # für 1001 gibt es einen interessanten ROC-Verlauf



val_size = 100 # Size of the validation set

l = len(d)



# np.random.choice: https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.choice.html

# Watch it: default is "replace=True", das führt dazu, dass mit Zurücklegen gezogen wird,

# also manche Indices mehrfach gezogen werden können. Das wollen wir hier nicht!

valid_idx = np.random.choice(l,val_size,replace=False) # wähle aus den 400 denkbaren indices 100 aus

train_idx = set(range(l)) - set(valid_idx)



train = np.array([d[i] for i in train_idx])

valid = np.array([d[i] for i in valid_idx])



# You will want to use sklearn methods to split data: 

# from sklearn.model_selection import train_test_split

# And you may want to scale the data (useless here)

# from sklearn.preprocessing import StandardScaler



trainX = train[...,0:2]

trainY = train[...,2:3]

validX = valid[...,0:2]

validY = valid[...,2:3]
## Implementieren wir ein einfaches kNN

from scipy.spatial.distance import cdist, squareform

M = cdist(validX,trainX, 'euclidean') # Determine a matrix of pairwise euclidian distances



# Prüfen wir mal stichprobenartig, ob die Einträge auch stimmen

import math

def dist(i,j,X1,X2):

    '''

    Für eine Kontrollrechnung implementieren wir kurz

    Euclidische Distanz "zu Fuss"

    '''

    return math.sqrt( (X1[i][0]-X2[j][0])**2 + (X1[i][1]-X2[j][1])**2 )





## Einige Tests

print(validX[1],trainX[2],dist(1,2,validX,trainX),"M[1,2]=",M[1,2])

print(trainX[34],trainX[212],dist(34,212,validX,trainX),"M[34,212]=",M[34,212])

### Geht wohl ;)



# Pick a slice

test = M[34,...] # row 34

print("Element 212 in row 34",test[212])
# Ignore the threshold parameter for a moment

def predict_class(N,Y,threshold):

    '''

    Computes the class occuring most in N.

    N is a list of (index,distance)-pairs where

    index points to samples in Y

    '''    

    predict = { -1 : 0, 1 : 0}

    for i,_ in N:

        predict[int(Y[i])] += 1

    if threshold != None: # ignore for now

        return 1 if predict[1]/(predict[1]+predict[-1]) >= threshold else -1

    else:

        if predict[-1] == predict[1]:

            return np.random.choice([1,-1])

        return 1 if predict[1] > predict[-1] else -1



def kNN(M,X,Y,k=4,threshold=None):

    # Produce a prediction vector 

    predict = []

    for i in range(len(X)):

        M_i = [(i,v) for i,v in enumerate(M[i,...])]

        neighbors = sorted(M_i,key=lambda x: x[1]) # heap would be better here, we need only the k best

        neighbors = neighbors[0:k]

        predict.append(predict_class(neighbors,Y,threshold))

    return predict

                       

prediction = kNN(M,validX,trainY)



def accuracy(predY,trueY):

    l = len(predY)

    correct = 0.

    for i in range(l):

        if predY[i] == trueY[i]:

            correct += 1.

    return correct/l



print(accuracy(prediction,validY))



## Do this for a number of k's



testK = []

for k in range(len(trainX)):

    testK.append((k,accuracy(kNN(M,validX,trainY,k=k),validY)))

testK = np.array(testK)

print(testK)
# plotting the points  

plt.plot(testK[...,0],testK[...,1]);



# Finding the best k

sortedK = sorted(testK,key=lambda x: x[1],reverse=True)

print("The 5 best k: ",sortedK[0:5])
def matrix(predY,trueY):

    result = { 'TP':0, 'FP':0, 'FN':0, 'TN':0 }

    l = len(predY)

    for i in range(l):

        if predY[i] == trueY[i]: # correct prediction

            if predY[i] == 1:

                result['TP'] += 1

            else:

                result['TN'] += 1

        else: # wrong prediction

            if predY[i] == 1:

                result['FP'] += 1

            else:

                result['FN'] += 1

    return result



matrixK = []

for k in range(len(trainX)):

    matrixK.append((k,matrix(kNN(M,validX,trainY,k=k),validY)))

print(matrixK)
## ROC-Graph für die kNN-Klassifizierer zeichnen

x,y = [],[]

for elem in matrixK:

    k,m = elem # mit k könnten wir den Punkt labeln, aber zu voll

    FPR = m['FP']/(m['FP']+m['TN'])

    TPR = m['TP']/(m['TP']+m['FN'])

    x.append(FPR)

    y.append(TPR)



def show_ROC(x,y,fmt=False):

    if fmt: 

        plt.plot(x, y, fmt)

    else:

        plt.plot(x, y, 'o', color='black')

    plt.xlim(0, 1.0)

    plt.ylim(0, 1.0)

    plt.xlabel('FPR')

    plt.ylabel('TPR');

    plt.title('ROC-Graph');

    plt.plot([0,1],[0,1]); # Diagonale zeigen



show_ROC(x,y)
ths=np.linspace(0,1.01,num=101,endpoint=True) # Thresholds to try

matrixK120 = []

testK120 = []

for th in ths:

    res = kNN(M,validX,trainY,k=120,threshold=th)

    matrixK120.append((th,matrix(res,validY))) # collect TP,FP,TN,FN for thresholds

    testK120.append((th,accuracy(res,validY))) # collect accuracies for thresholds

print(matrixK120)

testK120 = np.array(testK120)

print(testK120)
## ROC-Graph für den kNN-Klassifizierer mit k=120 zeichnen

x,y = [],[]

for elem in matrixK120:

    k,m = elem # mit k könnten wir den Punkt labeln, aber zu voll

    FPR = m['FP']/(m['FP']+m['TN'])

    TPR = m['TP']/(m['TP']+m['FN'])

    x.append(FPR)

    y.append(TPR)



show_ROC(x,y,fmt='og-') # show ROC

plt.show() # close first graph



# plotting threshold vs. accuracy for k=120

plt.title("Accuracy vs. threshold")

plt.xlim(0, 1.0)

plt.ylim(0, 1.0)

plt.xlabel('Threshold')

plt.ylabel('Accuracy');

plt.plot(testK120[...,0],testK120[...,1]);


#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

##

#

#