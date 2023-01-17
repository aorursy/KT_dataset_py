# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import seaborn as sns

sns.set(style="whitegrid", color_codes=True)



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import matplotlib 

import matplotlib.pyplot as plt

import sklearn

%matplotlib inline

import matplotlib.pyplot as plt 

plt.rcParams["figure.figsize"] = [16, 12]

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

filenames = check_output(["ls", "../input"]).decode("utf8").strip()
df = pd.read_csv("../input/Iris.csv") 

print(df.dtypes)

df.head()
varnames = df.columns.values



for varname in varnames:

    if varname not in ['Name', 'Ticket', 'Cabin'] and df[varname].dtype == 'object':

        lst = df[varname].unique()

        print(varname + " : " + str(len(lst)) + " values such as " + str(lst))
sns.pairplot(x_vars = ['SepalLengthCm'],y_vars=['SepalWidthCm'], hue = 'Species', data = df, size = 10);
sns.pairplot(x_vars = ['PetalLengthCm'],y_vars=['PetalWidthCm'], hue = 'Species', data = df, size = 10);
sns.violinplot(x="Species", y="PetalLengthCm", data=df, inner=None)
sns.swarmplot(x="Species", y="SepalLengthCm", data=df,  alpha=.5)
sns.barplot(x="Species", y="SepalLengthCm", data=df)