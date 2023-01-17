# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import metrics
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

def comupute_auc(df,prediction_col,actual_col):
    fpr, tpr, thresholds = metrics.roc_curve(df[actual_col],df[prediction_col], pos_label=1)
    auc_score = metrics.auc(fpr, tpr)
    return auc_score

def plot_roc(df,prediction_col,actual_col):
    fpr, tpr, thresholds = metrics.roc_curve(df[actual_col],df[prediction_col], pos_label=1)
    auc_score = metrics.auc(fpr, tpr)
    sns.set('talk', 'whitegrid', 'dark', font_scale=1, font='DejaVu Sans',
        rc={"lines.linewidth": 2, 'grid.linestyle': '--'})
    lw = 2
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (AUC = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()
    plt.close()
#please specify your target file name for computing roc and auc
target_file='demo_data.csv'
# create dataframe from csvfile
target_path = '../input/' +target_file
df=pd.read_csv(target_path)
#check columns name
print(list(df.columns))
# please prediction column name and actual column name
prediction_col = 'prediction'
actual_col = 'actual'
# compute auc
auc_score = comupute_auc(df,prediction_col,actual_col)
print('AUC score is {}.'.format(auc_score))
# plot ROC
plot_roc(df,prediction_col,actual_col)

