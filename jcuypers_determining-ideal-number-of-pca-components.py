# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder, StandardScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Let's load data and check if we got what we needed

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



print (df_train.shape)

print (df_test.shape)
# convert categorical values 

num_train = len(df_train)

x_all = pd.concat([df_train, df_test])



for c in x_all.columns:

    if x_all[c].dtype == 'object':

        lbl = LabelEncoder()

        lbl.fit(list(x_all[c].values))

        x_all[c] = lbl.transform(list(x_all[c].values))



df_train = x_all[:num_train]

df_test = x_all[num_train:]
#df_train = StandardScaler().fit_transform(df_train)
def calculate_optimal_n_comp(df):

    U, s, V = np.linalg.svd(df, full_matrices=True)

    total = np.sum(s)

    ksum = 0

    for i in range(0, s.shape[0]):

        ksum += s[i]

        print("n_comp {} - {:.10f}% variance explained ".format(i+1,(ksum/total)*100))
calculate_optimal_n_comp(df_train)