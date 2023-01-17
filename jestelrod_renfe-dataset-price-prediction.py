# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/renfe.csv")

""" WE WILL CHECK VALUES AT THE BEGINNING AND AT THE END OF THE DATASET"""

data.head(10) # USED TO CHECK THE FIRST 10 ROWS OF THE DATASET
data.tail(10) # USED TO CHECK THE LAST 10 ROWS OF THE DATASET
# Lets see the price distribution

data.price.describe()
data.count()
import matplotlib.pyplot as plt



def routes(df, group):

    for i in group:

        seti = np.unique(df[i])

        Q_i = list()

        for orig in seti:

            aux = len(df.index[data[i]==orig])

            Q_i.append(aux)

        if len(seti) >5:

            plt.barh(seti, Q_i)

            plt.show()

        else:

            plt.bar(seti, Q_i)

            plt.show()

        

routes(data, ["origin", "destination", "train_type"])
data_nan = data[data["price"].isnull()]

data_nan.head(10)
data_nan.tail(10)
from matplotlib import pyplot as plt

trainstypes = data.train_type.unique()

traincounter = list()

for trains in trainstypes:

    totaldata = len(data.index[data["train_type"]== trains])

    datanoprice = data_nan[data["train_type"]== trains]

    datanoprice = len(datanoprice.index)

    no_prices_rate = datanoprice / totaldata

    traincounter.append(no_prices_rate)

    #print ("Type of train", trains, "No price rate", no_prices_rate)



plt.barh(trainstypes, traincounter)

plt.show()
data_clean = data.dropna()

routes(data_clean, ["origin", "destination", "train_type"])