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
df = pd.read_csv("../input/scrubbed.csv")
df.describe()
df.dtypes
df
df[['city','latitude','longitude ']].tail() #= df['latitude'].str.replace('33q.200088', '33.200088')

#df.latitude.astype("float64")

#pd.to_numeric(df['latitude'])
df['latitude'].isnull().any()
#df['latitude'] = df['latitude'].str.replace('33q.200088', '33.200088')

#df.notnull()
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')

df['latitude'] = df['latitude'].fillna(33.200088)

df.latitude.astype("float64")
X1 = df['longitude ']

X2 = df['latitude']
import seaborn as sns

import matplotlib.pyplot as plt
sns.distplot(X1, kde = False, label='longitude', axlabel=False)

sns.distplot(X2, kde = False, label='latitude', axlabel=False)

plt.legend()
import matplotlib.pyplot as plt

plt.plot(X1, X2, 'r.')
#df[df['city']=='mescalero indian reservation']
#X2.isnull().any()
df.describe()