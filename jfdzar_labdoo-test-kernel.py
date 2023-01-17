# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/dootronics.csv',sep='\t')
df.head(3)
def show_barplot_column(column, threshold, other_bar=True):

    plt.figure(figsize=(15,5))

    values = df[column].value_counts()

    

    other_values = values.loc[~(values > threshold)].sum()

    values = values.loc[(values > threshold)]

    if other_bar: values['Other'] = other_values

    

    values.plot(kind='bar')

    

    sns.barplot(values.index, y=values)

    """

    plt.xticks(rotation=90)

    r_values = np.array(values)

    for i in range(len(r_values)):

        plt.text(x = i-0.25 , y = r_values[i]+10, s = r_values[i], size = 10)

        

    plt.show()

    """
show_barplot_column('Hub',200)
show_barplot_column('Country',200, True)