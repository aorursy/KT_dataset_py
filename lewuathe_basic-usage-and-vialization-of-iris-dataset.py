# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings # current version of seaborn generates a bunch of warnings that we'll ignore

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Exploring the iris data as my learning pandas and matplotlib

data = pd.read_csv('../input/Iris.csv')

data.head()
data.plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm')
# Seaborn visualization

import seaborn

seaborn.pairplot(data.drop("Id", axis=1), hue='Species', size=2)
# Calculate correlation coefficient

cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm']

cm = np.corrcoef(data[cols].values.T)

heatmap = seaborn.heatmap(cm, cbar=True, 

                          annot=True, 

                          square=True, 

                          fmt='.2f',

                          annot_kws={'size': 15},

                          yticklabels=cols,

                          xticklabels=cols)
# Visualize kernel density estimation 

seaborn.pairplot(data.drop("Id", axis=1), hue="Species", size=2.2, diag_kind="kde")
# You can change the style of seaborn

seaborn.set(style='dark')

seaborn.pairplot(data.drop("Id", axis=1), hue="Species", size=2.2, diag_kind="kde")
# You can reset changed style

seaborn.reset_orig()

seaborn.pairplot(data.drop("Id", axis=1), hue="Species", size=2.2, diag_kind="kde")