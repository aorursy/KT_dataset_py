# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input/images"]).decode("utf8"))

ch = check_output(["ls", "../input/"])



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
ch = df_train.ix[:,0:6]

ch.describe()
import matplotlib.pyplot as plt

import seaborn as sns

sns.set(font_scale=1)



pd.options.display.mpl_style = 'default'

#df_train.ix[:,0:20].hist(bins=10,figsize=(9,7),grid=False)
corr = df_train.corr()

plt.figure(figsize=(10, 10))



sns.heatmap(corr, vmax=1, square=True,annot=True,cmap='cubehelix')

plt.title('Correlation between features')

plt.show()
corr[:2]
