import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
# define the actual and predicted labels
X_actual = pd.Series([1,1,1,2,3,4,4,4,4,4]) 
X_pred   = pd.Series([1,1,1,2,3,4,4,4,4,0]) 
# create the histogram matrix O
O = confusion_matrix(X_actual, X_pred)
O
N = len(X_actual)
N
w = np.zeros((5,5))
w
for i in range(len(w)):
    for j in range(len(w)):
        w[i][j] = float(((i-j)**2)/((N-1)**2))
w
N = 5

# calculation of actual histogram vector
X_actual_hist=np.zeros([N]) 
for i in X_actual: 
    X_actual_hist[i]+=1    

print('Actuals value counts : {}'.format(X_actual_hist))
# calculation of predicted histogram vector
X_pred_hist=np.zeros([N]) 
for i in X_pred: 
    X_pred_hist[i]+=1    

print('Predicted value counts : {}'.format(X_pred_hist))
E = np.outer(X_actual_hist, X_pred_hist)
E
E = E/E.sum()
E.sum()
O = O/O.sum()
O.sum()
E
O
Num=0
Den=0

for i in range(len(w)):
    for j in range(len(w)):
        Num+=w[i][j]*O[i][j]
        Den+=w[i][j]*E[i][j]
        
Res = Num/Den
 
QWK = (1 - Res)
print('The QWK value is {}'.format(round(QWK,4)))