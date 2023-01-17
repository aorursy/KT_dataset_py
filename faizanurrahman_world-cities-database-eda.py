# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/worldcitiespop.csv')
df.head()
df.Country.nunique()
df.Country.value_counts()
sns.countplot(x='Country',data=df)
df.Region.nunique()
dfc = df.copy(deep=True)
dfc.isnull().sum()
dfc = dfc.dropna(how='any',axis=0)
by_country = dfc.groupby('Country',as_index=False).agg({"Population":"sum"})
by_country = by_country.sort_values(by='Population', ascending=False)
by_country.head()
plt.figure(figsize=(10,10))
sns.barplot(x='Country',y='Population',data=by_country[:20])
by_city = dfc.sort_values(by='Population',ascending=False)
by_city.head()
plt.figure(figsize=(18,10))
sns.barplot(x='City',y='Population',data=by_city[:20])
