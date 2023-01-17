import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



#ignore warnings

import warnings

warnings.filterwarnings('ignore')

print('-'*25)

train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')
train.head()
train.columns
train.info()
train.describe
train.head()
train["cp_type"].unique()
train["cp_time"].unique()
train["cp_dose"].unique()
train.cp_dose = [1 if each == "D1" else 0 for each in train.cp_dose]
train.cp_dose
def barPlot(var):

    var_train  = train[var] # get feature

    var_value = var_train.value_counts() # value_counts() count number of categorical variable 

    

    #visualize

    plt.figure(figsize = (8,4))

    plt.bar(var_value.index,var_value)

    plt.xticks(var_value.index, var_value.index.values)

    plt.ylabel('Frequency')

    plt.title(var)

    plt.show()

    print("{}: \n".format(var,var_value))
c1 = ["cp_type","cp_time","cp_dose"]

for x in c1:

    barPlot(x)