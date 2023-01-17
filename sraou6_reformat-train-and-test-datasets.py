# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pwd
root_dir = "/kaggle/input"
df = pd.read_csv(os.path.join(root_dir,'data','Data_Entry_2017.csv'))
df.head()
df = df[['Image Index','Finding Labels']] # drop other columns
df.head()
df_train_val = pd.read_csv(os.path.join(root_dir,'data','train_val_list.txt'), header=None)

df_test = pd.read_csv(os.path.join(root_dir,'data','test_list.txt'), header=None)
df_train_val.columns = ['Image Index']

df_test.columns = ['Image Index']

df_train_val.head()
n_train = len(df_train_val)

n_test = len(df_test)

total = n_train + n_test

ratio_train = n_train / total;

ratio_test = n_test / total;

print('#Samples:', total)

print('#Samples in Train and Validation Set:', n_train, "(%.2f)" % ratio_train )

print('#Samples in Test Set:', n_test, "(%.2f)" % ratio_test)
df['Finding Labels'].unique() 
classes = ['Atelectasis','Consolidation','Infiltration','Pneumothorax','Edema',\

           'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',\

           'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'No Finding']
""" Create a dictionary with class labels as keys and the values being colums indicating presence(1) or absence(0)

    for all Image Index. Initialize column values with -1. Later set as 1 if present or 0 otherwise."""

class_train_dict = {}

class_test_dict = {}

for key in classes:

    class_train_dict[key] = np.ones([len(df_train_val),1])*-1

    class_test_dict[key] = np.ones([len(df_test),1])*-1
df_train_val.head()
df = df.set_index(['Image Index'])
df.head()
for i in range(len(df_train_val)):

    for key in classes:

        if key in df.loc[df_train_val['Image Index'][i]]['Finding Labels']:

            class_train_dict[key][i] = 1

        else:

            class_train_dict[key][i] = 0    
for j in range(len(df_test)):

    for key in classes:

        if key in df.loc[df_test['Image Index'][j]]['Finding Labels']:

            class_test_dict[key][j] = 1

        else:

            class_test_dict[key][j] = 0    
for class_name in classes:

    df_train_val[class_name] =  class_train_dict[class_name]

    df_test[class_name] =  class_test_dict[class_name]
df_test.head()
n_positive_train = df_train_val.set_index('Image Index').sum()

print('# Positive samples training, validation set:')

print(n_positive_train)



n_positive_test = df_test.set_index('Image Index').sum()

print('\n# Positive samples test set:')

print(n_positive_test)
df_train_val.to_csv('train_val.csv')

df_test.to_csv('test.csv')