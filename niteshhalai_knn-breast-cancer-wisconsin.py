# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import matplotlib.pyplot as plt #Library for general visualizations

import seaborn as sns #For more beautiful visualizations

import numpy as np #Library that handles mathematical operations

import pandas as pd #Working with .csv files

import time #General Python time library



#Magic command to the jupyter notebook that we want all visualizations to stay within the file

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/data.csv")

df.head(10)
print(df.shape)

print(df.info())
targets = df.diagnosis

ax = sns.countplot(targets, label="Count", palette="Set3")

M,B = targets.value_counts()

print('Malignant:',M,"Percent:",int(M/(M+B)*100),"%")

print('Benign:',B,"   Percent:",int(B/(M+B)*100),"%")

targets = np.where(targets.values == 'M', 0, 1)