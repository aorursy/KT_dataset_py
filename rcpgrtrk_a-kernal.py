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
df=pd.read_csv("/kaggle/input/fifa19/data.csv")

df.head()
df.corr()# Burada neyin neyle bağlıntılı olduğuna baktım 
df.columns
#scatter plot

import matplotlib.pyplot as plt

df.plot(kind="scatter", x="Overall", y="Potential", grid=True, color="blue",alpha=0.5)

plt.xlabel("Overall")

plt.ylabel("Potential")

plt.legend()

plt.title("Potential vs Overall")

#potansiyel ve overall arasaındaki farka baktım.
df[np.logical_and(df['Age']<22, df['Potential']>88 )] # burada ise yası 22 den kücük potansiyeli 88 dan büyük oyuncululara baktım.