#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcS8tOEWWnDQYce1N4YGPXMICzPtzsudVDSIRujRmsxd6C72Gp88',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTWtxCzazRUMWgQcfxYWQYwjFvko9x8Xw-aZRdAweUHdgqOKSsE',width=400,height=400)
df = pd.read_excel('/kaggle/input/the-2019-ai-index-report/AI INDEX 2019 PUBLIC DATA/1. R_D/NESTA/Women in AI/Women_in_ai_country_data.xlsx')

df.head()
df.dtypes
df["year"].plot.hist()

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQOWYw1m3602py3CEmM45ohQv6rP_-GMJm70SxsBJjTc1slbX97',width=400,height=400)
df["year"].plot.box()

plt.show()
sns.pairplot(df, x_vars=['year'], y_vars='Country', markers="+", size=4)

plt.show()
dfcorr=df.corr()

dfcorr
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQAZO3JoPBbJRgLA8ZoTLLGCJJhRd-DxWiPmGDbGuMgukHdXiOR',width=400,height=400)
sns.heatmap(dfcorr,annot=True,cmap='winter')

plt.show()
fig, axes = plt.subplots(1, 1, figsize=(12, 4))

sns.boxplot(x='year', y='Country', data=df, showfliers=False);
df.plot.area(y=['year','Country'],alpha=0.4,figsize=(12, 6));
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTjLTrCj0o2CDMtWwCoSfMdl7C03y0YDx20u5e4SuByQsHInUgz',width=400,height=400)