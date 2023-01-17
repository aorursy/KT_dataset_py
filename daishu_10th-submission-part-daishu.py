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
import matplotlib.pyplot as plt

import seaborn as sns



id_name = 'TransactionID'

label_name = 'isFraud'



submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')

def Merge_sub(weightDic):

    plt.figure(figsize=(14,7))

    all_sub = np.zeros(submission.shape[0])

    for suffix in weightDic.keys():

        sub = pd.read_csv('/kaggle/input/10th-submissionpart-daishu/submission_%s.csv'%(suffix))

        sns.distplot(np.log10(sub[label_name]),bins=200,label=suffix)

        all_sub += sub[label_name] * weightDic[suffix]

    sub[label_name] = all_sub

    sns.distplot(np.log10(sub[label_name]),bins=200,label='after merge')

    plt.legend()

    sub.to_csv('./submission.csv',index=False)

    return None
Merge_sub({'k_gt_-1_lgb_metric_0.958464_0.957415_0.999982':0.4,

           'k_gt_0_lgb_metric_0.947764_0.948421_0.999979':0.2,

           'k_gt_1_lgb_metric_0.950788_0.951153_1.000000':0.4})