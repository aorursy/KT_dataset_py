import pandas as pd 
import numpy as np
import os

import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

def trim_pred(x, alpha):
    upper = 1-alpha
    lower = alpha
    if x > upper:
        return upper
    if x < lower:
        return lower
    else: return x
    
def trimmed_loss(alpha):
    trimmed_preds = [trim_pred(x,alpha) for x in data.Pred]
    return log_loss(data.Result , trimmed_preds)

def annot_min(x,y, ax=None):
    minIxVal = np.argmin(y);
    zeroBasedIx = y[minIxVal];
    xmin = x[minIxVal];
    ymin = y[minIxVal]
    text = "Minimum: Trim Interval = [{}, {}], Log Loss = {}".format(round(xmin,2), round(1-xmin,2), round(ymin, 3))
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="round,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=0.1")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmin, ymin), xytext=(0.94,0.90), **kw)
    

path = "../input/ncaa-2018-preds-and-truth/"

preds = pd.read_csv(os.path.join(path,"2018_predictions_logistic.csv")) 
truth = pd.read_csv(os.path.join(path,"truth.csv")) 

data = truth.merge(preds, left_on='ID', right_on='ID', how='inner')
data.head()
xvals = np.arange(0, .30, 0.001)
yvals = [trimmed_loss(alpha) for alpha in xvals]

plt.figure(figsize=(10,10))
plt.plot(xvals, yvals)
plt.xlabel('Alpha (Trim Amount)', fontsize=14)
plt.ylabel('Log Loss', fontsize=14)
annot_min(list(xvals),yvals, ax=None)
plt.show()
