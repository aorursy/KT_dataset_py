# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



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

filename = check_output(["ls", "../input"]).decode("utf8").strip()

df = pd.read_csv("../input/" + filename, thousands=",")

print(df.dtypes)

df.head()


df.describe()




varnames = df.columns.values



for varname in varnames:

    if df[varname].dtype == 'object':

        lst = df[varname].unique()

        print(varname + " : " + str(len(lst)) + " values such as " + str(lst))
df['date_local'] = pd.to_datetime(df['date_local'])
df.describe()
table = df.groupby(['parameter_name', 'state_name',     'date_local'])[  'arithmetic_mean'].mean()
table.head(5)
dM = table.loc['Mercury PM10 STP']
dM.plot()
dM.head()
dM.loc['Arizona'].plot()
def plotChemState(chem,state):

    table.loc[chem,state].plot()

    plt.title(chem + " in " + state)
chems = df['parameter_name'].unique()
states = df['state_name'].unique()
plotChemState(chems[4],states[10])
plotChemState(chems[14],states[42])