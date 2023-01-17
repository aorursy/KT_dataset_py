import numpy as np
from sklearn import metrics 
n = 8 # number of 'training examples'

# create some dummy data 
y_pred = np.zeros(n*10).reshape((n, 10))
y_true = np.zeros(n*10).reshape((n, 10))
y_pred[:4] = [1,0,0,0,0,0,0,0,0,0] # (play with it to see the effects!)
y_true[:] = [1,1,0,0,0,0,0,0,0,0] # (play with it to see the effects!)
print('Micro F1:',metrics.f1_score(y_true, y_pred, average='micro'))
print('Macro F1:',metrics.f1_score(y_true, y_pred, average='macro')) 
# Let's recreate the functions and have a closer look:

def f1_micro(y_true, y_preds, thresh=0.5, eps=1e-20):
    preds_bin = y_preds > thresh # binary representation from probabilities (not relevant)
    truepos = preds_bin * y_true
    
    p = truepos.sum() / (preds_bin.sum() + eps) # take sums and calculate precision on scalars
    r = truepos.sum() / (y_true.sum() + eps) # take sums and calculate recall on scalars
    
    f1 = 2*p*r / (p+r+eps) # we calculate f1 on scalars
    return f1

def f1_macro(y_true, y_preds, thresh=0.5, eps=1e-20):
    preds_bin = y_preds > thresh # binary representation from probabilities (not relevant)
    truepos = preds_bin * y_true

    p = truepos.sum(axis=0) / (preds_bin.sum(axis=0) + eps) # sum along axis=0 (classes)
                                                            # and calculate precision array
    r = truepos.sum(axis=0) / (y_true.sum(axis=0) + eps)    # sum along axis=0 (classes) 
                                                            #  and calculate recall array

    f1 = 2*p*r / (p+r+eps) # we calculate f1 on arrays
    return np.mean(f1) # we take the average of the individual f1 scores at the very end!

print('Micro F1 (sklearn):',metrics.f1_score(y_true, y_pred, average='micro'))
print('Micro F1 (own)    :',f1_micro(y_true, y_pred))
print('Macro F1 (sklearn):',metrics.f1_score(y_true, y_pred, average='macro')) 
print('Macro F1 (own)    :',f1_macro(y_true, y_pred))