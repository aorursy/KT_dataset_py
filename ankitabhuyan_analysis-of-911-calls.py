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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv("/kaggle/input/montcoalert/911.csv")

df.head()
df.info()
df['zip'].value_counts().head(5)
df['twp'].value_counts().head(5)
df["title"].nunique()

df["Reason"] = df["title"].apply(lambda title: title.split(':')[0])

df['Reason']
df['Reason'].value_counts().head(3)
sns.countplot(x="Reason", data=df, orient = 'h', palette = "husl")
type(df['timeStamp'].iloc[0])