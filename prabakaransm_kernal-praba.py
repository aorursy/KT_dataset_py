# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
path = '../input/'
pd.options.display.max_columns = 999
df=pd.read_csv(path + 'zomato.csv' ,encoding = "ISO-8859-1")
df.head(2)
df.shape
df.dtypes
df_CC = pd.read_excel( path + 'Country-Code.xlsx')
df_CC.head(3)
df.columns
df_grp = df.groupby(['Country Code'],as_index=False).count()[['Country Code','Restaurant ID']]
df_grp.columns=['Country Code','No of Restauant']
grp = df_grp.join(df_CC.set_index('Country Code'),on='Country Code')
grp.head()
sns.set(rc={'figure.figsize':(10,5)})
sns.barplot(grp['Country'],grp['No of Restauant'])
plt.xticks(rotation = 90)
plt.show()
df_grp_ct = df.groupby(['Country Code'],as_index=False)
df_ct_mean = df_grp_ct['Aggregate rating'].agg(np.mean)
df_ct_mean.head(3)
mean_res = df_ct_mean.join(df_CC.set_index('Country Code'),on='Country Code')
mean_res.head(3)
sns.set(rc={'figure.figsize':(10,7)})
sns.barplot(mean_res['Country'],mean_res['Aggregate rating'])
plt.xticks(rotation=70)
plt.show()
