import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

chance50 = pd.read_csv("/kaggle/input/50_percent_chance.csv")

chance25 = pd.read_csv("/kaggle/input/25_percent_chance.csv")

chance10 = pd.read_csv("/kaggle/input/10_percent_chance.csv")
def get_random_list(n, percent):

    """

    generate a list with n items of 0 or 1

    percent between 0 to 100

    """

    random_list = []

    

    for i in range(n):

        random_list.append(int(random.randint(0,99)<percent))

    

    return random_list





def get_result(sample,risk):

    """

    compute the 100 trial with % of risk

    """

    l = []

    

    for i in range(100):

        tmp = pd.Series(get_random_list(1000,risk),name="Validate",index=sample.index)

        result = sample.join(tmp)

        l.append(result.loc[result.Score == result.Validate]['Score'].count())

    

    return l
pd.Series(get_result(chance10,2)).describe()
pd.Series(get_result(chance25,1)).describe()
result50 = get_result(chance50,10)

pd.Series(result50).describe()
pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



plt.figure(figsize=(20,6))

sns.barplot(x=pd.Series(result50).value_counts().sort_index().index,

            y=pd.Series(result50).value_counts().sort_index())