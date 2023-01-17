import  numpy as np

from sklearn.metrics import confusion_matrix

import pandas as pd

import matplotlib.pyplot as plt

import itertools

%matplotlib inline
cm  = np.array([[ 5284.0,  62.0, 167.0, 26.0],

                                              [  60.0,  1553.0,  11.0, 76.0],

                                              [  193.0,  21.0, 980.0,  118.0],

                                              [  27.0,  108.0,  82.0, 3754.0]])

FP = cm.sum(axis=0) - np.diag(cm)  

FN = cm.sum(axis=1) - np.diag(cm)

TP = np.diag(cm)

TN = cm.sum() - (FP + FN + TP)



# Sensitivity, hit rate, recall, or true positive rate

TPR = TP/(TP+FN)

print("TPR",TPR)

# Specificity or true negative rate

TNR = TN/(TN+FP) 

print("TNR",TNR)

# Precision or positive predictive value

PPV = TP/(TP+FP)

print("PPV",PPV)

# Negative predictive value

NPV = TN/(TN+FN)

print("NPV",NPV)

# Fall out or false positive rate

FPR = FP/(FP+TN)

print("FPR",FPR)

# False negative rate

FNR = FN/(TP+FN)

print("FNR",FNR)

# False discovery rate

FDR = FP/(TP+FP)

print("FDR",FDR)



# Overall accuracy

ACC = (TP+TN)/(TP+FP+FN+TN)

print("Accuracy",ACC)
