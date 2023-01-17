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


Servery_Data = pd.read_csv("/kaggle/input/servery-data/Data Project - Part II - Sheet1 (1).csv")
Servery_Data.head()
Servery_Data.corr()
Servery_Data.describe()
# library & dataset

import seaborn as sns

import matplotlib.pyplot as plt

#Servery_Data = sns.load_dataset('Servery_Data')

 

# use the function regplot to make a scatterplot

sns.regplot(x=Servery_Data["Grade"], y=Servery_Data["Servery visits/week"], fit_reg=False)

plt.show()

 

# Without regression fit:

#sns.regplot(x=df["sepal_length"], y=df["sepal_width"], fit_reg=False)

#sns.plt.show()

# libraries

import matplotlib.pyplot as plt

import numpy as np

 

# create data





plt.figure( figsize = (20,10))



x = Servery_Data["Grade"]

y = Servery_Data["Servery visits/week"]

z = 1



plt.ylabel("Servery visits/week", fontsize = 20)

plt.xlabel("Grade", fontsize = 20)

plt.title("10th, 11th, 12th, and Faculty Visits to the Servery Per Week", fontsize = 20)



# use the scatter function

plt.scatter(x, y, s=z*1000, alpha=0.5)

plt.show()


