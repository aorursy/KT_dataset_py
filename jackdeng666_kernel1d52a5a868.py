# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import numpy as np

from sklearn.metrics import roc_curve

from sklearn.metrics import auc



#---自己按照公式实现

def auc_calculate(labels,preds,n_bins=100):

    postive_len = sum(labels)

    negative_len = len(labels) - postive_len

    total_case = postive_len * negative_len

    pos_histogram = [0 for _ in range(n_bins)]

    neg_histogram = [0 for _ in range(n_bins)]

    bin_width = 1.0 / n_bins

    for i in range(len(labels)):

        nth_bin = int(preds[i]/bin_width)

        if labels[i]==1:

            pos_histogram[nth_bin] += 1

        else:

            neg_histogram[nth_bin] += 1

    accumulated_neg = 0

    satisfied_pair = 0

    for i in range(n_bins):

        satisfied_pair += (pos_histogram[i]*accumulated_neg + pos_histogram[i]*neg_histogram[i]*0.5)

        accumulated_neg += neg_histogram[i]



    return satisfied_pair / float(total_case)



if __name__ == '__main__':



    y = np.array([1,0,0,0,0,0,1,0,1])

    pred = np.array([0.9, 0.8, 0.3, 0.1,0.4,0.9,0.66,0.7,0.5])





    fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)

    print("-----sklearn:",auc(fpr, tpr))

    print("-----py脚本:",auc_calculate(y,pred))
