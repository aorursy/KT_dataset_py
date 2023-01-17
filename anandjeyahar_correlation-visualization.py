# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import json

%matplotlib inline



import matplotlib.pyplot as plt

import itertools
def mscatter(p, x, y, typestr="o"):

    p.scatter(x, y, marker=typestr, alpha=0.5)



def correlation_analyze(df, exclude_columns = None):

    #TODO: Plot scatter plot of all combinations of column variables in the df

    import matplotlib.pyplot as plt

    import numpy as np

    columns = filter(lambda x: x not in exclude_columns, df.columns)

    numerical_columns = filter(lambda x: df[x].dtype in [np.float64, np.int64] ,columns)

    combos = list(itertools.combinations(numerical_columns, 2))

    # TODO: based on the len(combos) decide how many figures to plot as there's a max of 9 subplots in mpl

    fig = plt.figure()

    for i, combo in enumerate(combos):

        u,v = combo

        # Damn odd way of matplotlib's putting together how many sub plots and which one.

        ax1 = fig.add_subplot(int("3" + str(int(len(combos)/3)) + str(i + 1)))

        mscatter(ax1, df[u], df[v])

        plt.legend(loc='upper left')

    plt.show()

    print(df.corr())

    
df = pd.read_csv("../input/Iris.csv") # the iris dataset is now a Pandas DataFrame

df.describe()

df.head()

correlation_analyze(df, exclude_columns='Id')