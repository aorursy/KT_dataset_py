# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.tools.plotting import scatter_matrix

import matplotlib.pyplot as plt



# Make the graphs a bit prettier, and bigger

pd.set_option('display.mpl_style', 'default')



# Always display all the columns

pd.set_option('display.line_width', 5000) 

pd.set_option('display.max_columns', 60)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

df = pd.read_csv("../input/diabetes.csv", names=names)

df.describe()
df.head()
df['Outcome'].value_counts()
colors=['red','green']

scatter_matrix(df,figsize=[20,20],marker='x',c=df.Outcome.apply(lambda x:colors[x]))

plt.show()