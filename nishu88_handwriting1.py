# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../input/A_Z Handwritten Data"))

# Any results you write to the current directory are saved as output.
data_train_file= "../input/A_Z Handwritten Data/A_Z Handwritten Data.csv"
data_test_file= "../input/A_Z Handwritten Data/A_Z Handwritten Data.csv"

df_train=pd.read_csv(data_train_file)
df_test= pd.read_csv(data_test_file)

df_train.head()
df_train.describe()
import matplotlib.pyplot as plt 
%matplotlib inline
for i in range(5000,5005):
    sample =np.reshape(df_test[df_test.columns[1:]].iloc[i].values/255,(28,28))
    plt.figure()
    plt.title("labelled cs {}".format(df_test["0"].iloc[i]))
    plt.imshow(sample,'gray')
