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
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_submit = pd.read_csv('../input/sample_submission.csv')
df_train.info()
df_test.info()
df_submit.info()
df_train.head()
df_test.head()
df_submit.head()
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(df_test['experiment'])
encoder.classes_
encoder.fit(df_train['event'])
train_event = encoder.classes_
print(train_event)
encoder.fit(df_train['experiment'])
train_experiment = encoder.classes_
print(train_experiment)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

cm = sns.light_palette("orange", as_cmap=True)

cross_data = pd.crosstab(df_train['event'],df_train['experiment'])
s = cross_data.style.background_gradient(cmap='viridis', low=.5, high=0).highlight_null('red')
display(s)
cross_data.plot(kind="bar", stacked = True, figsize = (20,8))
plt.title("Number of Congnitive Biases per event")
len(set(df_train.time).intersection(df_test["time"]) )
df_train.count()
def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 
print(intersection(df_train['time'], df_test['time']))
target = df_train.event
s = pd.Series(target)
df2 = pd.get_dummies(s)
df_train = pd.concat([df_train, df2], axis=1, sort=False)
target_Col = ['A','B','C','D']
df_train.describe()
df_train.corr()
y_train = df2
x_train_col = ['time','seat','eeg_fp1','eeg_f7','eeg_f8','eeg_t4','eeg_t6','eeg_t5','eeg_t3','eeg_fp2','eeg_o1','eeg_p3','eeg_pz','eeg_f3','eeg_fz','eeg_f4','eeg_c4','eeg_p4','eeg_poz','eeg_c3','eeg_cz','eeg_o2','r','ecg','gsr'] 
x_train = df_train['crew']
for x in x_train_col:
   x_train = pd.concat([x_train, df_train[x]], axis=1, sort=False)
x_train.head()
