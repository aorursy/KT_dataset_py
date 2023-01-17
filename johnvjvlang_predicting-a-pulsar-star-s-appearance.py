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

        

import seaborn as sns

import matplotlib.pyplot as plt



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/predicting-a-pulsar-star/pulsar_stars.csv")  

df.head()
df.info()
colorscheme = ["#34495e", "#2ecc71"]

sns.pairplot(df, hue = "target_class", palette = "husl", diag_kind = "kde", kind = 'scatter')
plt.figure(figsize=(16,12))

sns.heatmap(data=df.corr(),annot=True,cmap="coolwarm",linewidths=1,fmt=".2f")

plt.title("Correlation Map",fontsize=20)

plt.tight_layout()

plt.show()  