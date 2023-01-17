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



%%javascript

MathJax.Hub.Config({

    TeX: { equationNumbers: { autoNumber: "AMS" } }

});
N = 11592



TP = 0.694*np.sqrt((384)*(267)*(N-384)*(N-267))/(N-384)

TP = int(TP)

print(TP)
FP = int(384- TP)

FN = 267-223

TN = int(N-TP-FN-FP)



print(FP,TN,FN)

print('sanity check : total number of examples = TP+FP+TN+FN = ',TP+FP+FN+TN)
def neg_precision(y_true, y_pred):



    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    y_pred_neg = 1 - y_pred_pos



    y_pos = K.round(K.clip(y_true, 0, 1))

    y_neg = 1 - y_pos



    tn = K.sum(y_neg * y_pred_neg)

    fn = K.sum(y_pos * y_pred_neg)

    numerator = tn

    denominator = (tn + fn)



    return numerator / (denominator + 1e-15)
tp,fp,tn,fn = TP,FP,TN,FN



def mcc(tp,fp,tn,fn):

    return (tp*tn - fp*fn)/((tp+fp)*(tp+fn)*(fn+tn)*(fp+tn))**0.5



print(mcc(tp,fp,tn,fn))
#correctly from 0 to 1

print(mcc(tp+1,fp,tn,fn-1))



#correctly from 1 to 0

print(mcc(tp,fp-1,tn+1,fn))



#incorrectly from 0 to 1

print(mcc(tp,fp+1,tn-1,fn))



#incorrectly from 1 to 0

print(mcc(tp-1,fp,tn,fn+1))
#correctly from 0 to 1

print(mcc(tp+1,fp,tn,fn-1) - mcc(tp,fp,tn,fn)) 



#correctly from 1 to 0

print(mcc(tp,fp-1,tn+1,fn) - mcc(tp,fp,tn,fn))



#incorrectly from 0 to 1

print(mcc(tp,fp+1,tn-1,fn) - mcc(tp,fp,tn,fn))



#incorrectly from 1 to 0

print(mcc(tp-1,fp,tn,fn+1) - mcc(tp,fp,tn,fn))
print(mcc(tp+28,fp,tn,fn-28))

print(mcc(tp,fp-60,tn+60,fn))

print(mcc(tp+14,fp-30,tn+30,fn-14))
print(mcc(tp+6,fp-12,tn+12,fn-6))

print(mcc(tp+12,fp-12,tn+12,fn-12))