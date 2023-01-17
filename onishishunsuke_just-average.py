# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm_notebook as tqdm

import category_encoders as ce



from sklearn.preprocessing import StandardScaler



import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/train.csv')

df_test = pd.read_csv('/kaggle/input/test.csv')

df_country = pd.read_csv('/kaggle/input/country_info.csv')

sub = pd.read_csv('/kaggle/input/sample_submission.csv')
df_train_m = df_train.query('Gender == "Male"')

df_train_w =df_train.query('Gender != "Male"')



df_test_m = df_test.query('Gender == "Male"')

df_test_w = df_test.query('Gender != "Male"')
df_train_m['ConvertedSalary'].mean(),df_train_w['ConvertedSalary'].mean()
df_test_m['ConvertedSalary'] = 97015.8023733578

df_test_w['ConvertedSalary'] = 91838.7436853704
df_merge = pd.merge(sub, df_test_m, left_on='ConvertedSalary', right_on='ConvertedSalary')

df_merge = pd.merge(sub, df_test_w, left_on='ConvertedSalary', right_on='ConvertedSalary')

submit = df_merge.loc[:,['Respondent','ConvertedSalary']]
X_train = df_train.loc[:,['Employment','AssessJob1','Age']]

y_train = df_train['ConvertedSalary']



X_test = df_test.loc[:,['Employment','AssessJob1','Age']]
y_train.mean()
sub.head()
sub['ConvertedSalary'] = 96303.81045858299
sub.to_csv('test_sub2.csv',index=False)
sub.head()