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
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



dfc = pd.read_csv('../input/cereal.csv')

print (dfc.info())
print (dfc.describe())
print (dfc['mfr'].unique())
print (dfc['type'].unique())
d1 = dfc['mfr']

d2 = dfc['shelf']

fig, axes = plt.subplots(1, 2)

sns.countplot(d1, ax=axes[0]).set_title('Manufacturers')

sns.countplot(d2, ax=axes[1]).set_title('Position in Shelf')
dfc_sample = dfc.drop(['sodium', 'potass', 'vitamins', 'shelf', 'weight', 'cups'], axis=1)

sns.pairplot(dfc_sample)