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
import pandas as pd

TSLA = pd.read_csv("../input/tesla-stock-data-from-2010-to-2020/TSLA.csv")
TSLA.head()
# # library & dataset

# import seaborn as sns

# plt.figure( figsize = (20,10))

# sns.boxplot( x=TSLA["Close"] )

# plt.show()

from scipy import stats

pearson_coef, p_value = stats.pearsonr(TSLA["Open"], TSLA["Close"])



# Pearson coefficient / correlation coefficient - how much are the two columns correlated?

print(pearson_coef)



# P-value - how sure are we about this correlation?

print(p_value)
# libraries

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

 

# plot

plt.plot( 'Open', 'High', data=TSLA, linestyle='-', marker='*')

plt.show()
TSLA.corr()
import seaborn as sns

import matplotlib.pyplot as plt

 

plt.figure( figsize = (20,10))

# use the function regplot to make a scatterplot

sns.regplot(x=TSLA["High"], y=TSLA["Low"],marker=".")

plt.show()

 

# Without regression fit:

# sns.regplot(x=top50["Energy"], y=top50["Loudness..dB.."], fit_reg=False)

# plt.show()
slope, intercept, r_value, p_value, std_err = stats.linregress(TSLA["Open"],TSLA["Close"])

print("y = " + str(slope) + "x + " + str(intercept))



# Same as (Pearson) correlation coefficient

print(r_value)
TSLA.describe()
import seaborn as sns

import matplotlib.pyplot as plt

#Servery_Data = sns.load_dataset('Servery_Data')

 

# use the function regplot to make a scatterplot

sns.regplot(x=TSLA["Open"], y=TSLA["Close"], fit_reg=False)

plt.show()

 

# Without regression fit:

#sns.regplot(x=df["sepal_length"], y=df["sepal_width"], fit_reg=False)

#sns.plt.show()
# libraries

import matplotlib.pyplot as plt

import numpy as np

 

# create data





plt.figure( figsize = (20,10))



x = TSLA["Open"]

y = TSLA["Close"]

z = .5



plt.ylabel("Close", fontsize = 20)

plt.xlabel("Open", fontsize = 20)

plt.title("Tesla stock data from 2010 - 2020", fontsize = 20)



# use the scatter function

plt.scatter(x, y, s=z*1000, alpha=0.5)

plt.show()