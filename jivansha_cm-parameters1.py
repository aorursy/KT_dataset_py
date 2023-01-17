import  numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import itertools
%matplotlib inline
cm=np.array([
[  2952.0,   8.0,    31.0,    5.0], 
[  21.0,   882.0,   2.0,   33.0], 
[  40.0,  1.0,  617.0,  16.0], 
[ 8.0,  25.0,   74.0,  1963.0]])


FP = cm.sum(axis=0) - np.diag(cm)  
TP = np.diag(cm)
TN = cm.sum() - (FP + TP + TN)
FN = cm.sum(axis=1) - TP
print('TP - ', TP)
print('TN - ', TN)
print('FP - ', FP)
print('FN - ', FN)
tp_sum = TP.sum()
tn_sum = TN.sum()
fn_sum = FN.sum()
fp_sum = FP.sum()
print(tp_sum, tn_sum,fn_sum,fp_sum)
sensitivity = tp_sum/(tp_sum + fn_sum)
print('sensitivity - ',sensitivity)
specificity = tn_sum/(tn_sum + fp_sum)
print('specificity  - ', specificity)
confusion_matrix =np.array([
[  2971.0,   13.0,    34.0,    2.0], 
[  3.0,   865.0,   2.0,   20.0], 
[  26.0,  1.0,  608.0,  49.0], 
[ 3.0,  29.0,   9.0,  2043.0]])
classes = ['CNV','DME','DRUSEN','NORMAL']

fig, ax = plt.subplots()
im = ax.imshow(confusion_matrix, cmap='BuGn')
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)

for i in range(len(classes)):
    for j in range(len(classes)):
        text = ax.text(j, i, confusion_matrix[i, j],ha="center", va="center", color="orange")

ax.set_title("Confusion Matrix- 7 layered CNN for modified images")
fig.tight_layout()
plt.show()

