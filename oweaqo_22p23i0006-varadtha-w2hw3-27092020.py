# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv',)

df.info()

df.head()
data = df[["neighbourhood","price"]].groupby("neighbourhood").mean().reset_index()



print(len(data))

data.head(10)
df = data.set_index('neighbourhood')
from matplotlib import pyplot as plt

from scipy.cluster import hierarchy



plt.figure(figsize=(50, 50))

plt.title("Dendrograms between neighbourhood and mean price", fontsize= 40)



d = hierarchy.linkage(df, method='ward')

hierarchy.dendrogram(d, leaf_rotation=90, leaf_font_size=12, labels=df.index)



plt.show()