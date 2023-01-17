# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Test.csv', low_memory=False)
data.head()
# data = data.astype(object).where(pd.notnull(data),"None")

# data.head()
# data = data.astype(int)

data = data.astype({"SalesID": int, "MachineID": int, "datasource":int,"YearMade":int})

# data.info()
data.head()
# check if a column has any nan values

data.fiSecondaryDesc.isna().any()
# replacing with mean values

data.fillna(0, inplace=True)

data['MachineHoursCurrentMeter'] = data['MachineHoursCurrentMeter'].fillna((data['MachineHoursCurrentMeter'].mean()))

# data.MachineHoursCurrentMeter.isna().any()

data['UsageBand'] = data['UsageBand'].fillna('NA')

data.UsageBand.unique()
# data['saledate'] = data['saledate'].astype('datetime64[ns]')
# data.boxplot(['UsageBand'])

# data.boxplot('SalesID','UsageBand')
# sns.heatmap(data, cmap='RdYlGn_r', linewidths=0.5, annot=True)

cols_lis = data.columns

# cols_lis = data.columns

# cols_lis

for col in cols_lis:

    data[col].plot.hist()

    plt.title(col)

    plt.show()
data.corr(method='pearson')
def plot_corr(df,size=10):

    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.



    Input:

        df: pandas DataFrame

        size: vertical and horizontal size of the plot'''



    corr = df.corr()

    fig, ax = plt.subplots(figsize=(size, size))

    ax.matshow(corr)

    plt.xticks(range(len(corr.columns)), corr.columns);

    plt.yticks(range(len(corr.columns)), corr.columns);
plot_corr(data)
rs = np.random.RandomState(0)

df = pd.DataFrame(rs.rand(10, 10))

corr = data.corr()

corr.style.background_gradient(cmap='coolwarm')