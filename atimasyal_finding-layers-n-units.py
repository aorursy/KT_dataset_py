import numpy as np
import pandas as pd
import math
import random
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import fbeta_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve, matthews_corrcoef, cohen_kappa_score
def f(s):
    if s=="negative":
        return 0
    else:
        return 1
    
def f1(s):
    if s=="M":
        return 2
    else:
        return 1
df=pd.read_csv("../input/project/glass4.csv")
df["Class"]=df.Class.apply(f)
#df["Sex"]=df.Sex.apply(f)
N=df.shape[0]
M=df.shape[1]
x=df.values[:, :M-1]
y=df.values[:, M-1]
scaler=preprocessing.StandardScaler()
x[0], y[0], x.shape
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
#scaler.fit(x_train)
#scaler.transform(x_train)
#scaler.transform(x_test)
x.shape, len(x_train)
N=x.shape[0]
M=x.shape[1]
N,M
UNITS_col = []
UNITS = pd.DataFrame(columns = UNITS_col)
row_index = 0
for c in range(2,M):
  for d in range (2, c):
   for e in range (2, d):
    clf = MLPClassifier(activation='logistic', solver='lbfgs', random_state=1, alpha=0.05, learning_rate_init=0.5, hidden_layer_sizes=(c,d,e), max_iter=1000)
    clf.fit(x_train, y_train)
    
    y_test_pred=clf.predict(x_test)
    fpstd2, tpstd2, thstd2 = roc_curve(y_test, y_test_pred)
    fb_std2 = fbeta_score(y_test, y_test_pred, average='binary', beta=2) #binary bcos f2 of positive class is needed
    mccs2 = matthews_corrcoef(y_test, y_test_pred)
    kappas2 = cohen_kappa_score(y_test, y_test_pred)
    
    
    UNITS.loc[row_index,'Units'] = str(c) + ', ' + str(d) + ', '+ str(e)
    UNITS.loc[row_index, 'Precission'] = precision_score(y_test, y_test_pred)
    UNITS.loc[row_index, 'Recall'] = recall_score(y_test, y_test_pred)
    UNITS.loc[row_index, 'AUC'] = auc(fpstd2, tpstd2)
    UNITS.loc[row_index, 'FBeta'] = fb_std2
    UNITS.loc[row_index, 'Algo MCC'] = mccs2
    UNITS.loc[row_index, 'Algo Kappa'] = kappas2
    row_index+=1

    
    sum0=0
    sum1=0
    tot0=0
    tot1=0
    for i in range(len(x_test)):
        if(y_test[i]==0):
            tot0+=1
            if(y_test_pred[i]==y_test[i]):
                sum0+=1
        else:
            tot1+=1
            if(y_test_pred[i]==y_test[i]):
                sum1+=1
    print(c, d, len(x_test), tot0, tot1, 'std')
    print(c, d, sum0+sum1, sum0, sum1)
    #print(precision_score(y_train, y_train_pred), recall_score(y_train, y_train_pred), auc(fpstd1, tpstd1), fb_std1, mccs1, kappas1, )
    #print(c, precision_score(y_test, y_test_pred), recall_score(y_test, y_test_pred), auc(fpstd2, tpstd2), fb_std2, mccs2, kappas2)
UNITS
#def color_negative_red(val):
#    color = 'lawngreen' 
#    return 'background-color: %s' % color
#UNITS = UNITS.style.applymap(color_negative_red, subset=pd.IndexSlice[[6],])
#UNITS
