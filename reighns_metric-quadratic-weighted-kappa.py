import numpy as np

import pandas as pd

from sklearn.metrics import confusion_matrix

from sklearn.metrics import cohen_kappa_score, make_scorer



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
actuals = pd.Series(['cat',  'cat', 'dog', 'cat',   'cat',  'cat', 'pig',  'pig', 'hen', 'pig'], name = 'Actual')

preds   = pd.Series(['bird', 'hen', 'pig','bird',  'bird', 'bird', 'pig', 'pig', 'hen', 'pig'], name = 'Predicted')

C = confusion_matrix(actuals, preds); 



C



#print(metrics.classification_report(actuals,preds, digits=5))
# Here is the actual vs pred grades/class



actual = pd.Series([2,2,2,3,4,5,5,5,5,5]) 

pred   = pd.Series([2,2,2,3,2,1,1,1,1,3]) 
C = confusion_matrix(actual, pred); 



C
weighted = np.zeros((5,5)) # We construct the weighted matrix starting from a zero matrix, it is like constructing a 

                           # list, we usually start from an empty list and add things inside using loops.



for i in range(len(weighted)):

    for j in range(len(weighted)):

        weighted[i][j] = float(((i-j)**2)/16) 

        

weighted
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