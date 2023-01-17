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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df1=pd.read_csv('../input/ks-projects-201801.csv')
df1.head(10) # ID:ID,name:プロジェクト名,category:カテゴリ,main_category:メインカテゴリ,currency:通貨,deadline:締切日,goal:目標金額,launched:開始日

             # pledged:誓約数,state:結果,backers:支援者数,country:国,usd pledged:誓約金額,usd_pledged_real:○○?,usd_goal_real:○○?
df1.describe() # 要約統計量の表示
pd.plotting.scatter_matrix(df1,figsize=(20,20)) # 散布図行列を計算、表示

plt.show()
df1.corr() # 各相関係数を表示
sns.heatmap(df1.corr()) # 各相関係数のヒートマップを表示

plt.show()