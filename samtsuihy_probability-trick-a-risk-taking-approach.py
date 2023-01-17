import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

train = pd.read_csv("/kaggle/input/titanic/train.csv",index_col=0)



female = train[['Survived','Sex']].loc[train.Sex == "female"]

female.shape        
female.loc[female.Survived==1]['Survived'].count()
import random



def get_random_list(n, percent):

    """

    generate a list with n items of 0 or 1

    percent between 0 to 100

    """

    random_list = []

    for i in range(n):

        random_list.append(int(random.randint(0,99)<percent))

    

    return random_list
l = []



for i in range(100):

    tmp = pd.Series(get_random_list(314,98),

                 name="Validate",

                 index=female.index)

    result = female.join(tmp)

    l.append(result.loc[result.Survived==result.Validate]['Survived'].count())

    

pd.Series(l).describe()

    
pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



plt.figure(figsize=(10,6))

sns.barplot(x=pd.Series(l).value_counts().sort_index().index,

            y=pd.Series(l).value_counts().sort_index())