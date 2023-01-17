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

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd

df = pd.read_csv('/kaggle/input/habermans-survival-data-set/haberman.csv')

df.head()
df.tail()
df.columns
df.shape
#Renaming the attributes for better undertanding

df = df.rename(columns = {"30" : "age", "64" : "operation_year", "1" : "axillary_lymph_node", "1.1" : "survival_status"})

df.head()
df.info()
df.describe()
# The values are mapped as follows. 1 :-> 'True' 2 :-> 'False'

df['survival_status'] = df['survival_status'].map({1:True, 2:False})

df.head()
sns.set_style("whitegrid");

sns.FacetGrid(df, hue="survival_status", size=4).map(plt.scatter, "age", "axillary_lymph_node").add_legend();

plt.show();
plt.close();

sns.set_style("whitegrid");

sns.pairplot(df, hue="survival_status", size=3);

plt.show()
corr = df.corr(method = 'spearman')

sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values)