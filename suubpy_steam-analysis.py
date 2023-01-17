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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
steam_data = pd.read_csv('/kaggle/input/steam-video-games/steam-200k.csv')

steam_data.head()
data_copy = steam_data.copy()

data_copy.head()
steam_data = steam_data.rename(columns={"The Elder Scrolls V Skyrim": "Name of the games"})

steam_data = steam_data.rename(columns={"151603712": "User ID"})

steam_data = steam_data.rename(columns={"1.0": "Hoursplay"})
steam_data.head()
steam_data.info()
from pandas.api.types import CategoricalDtype

steam_data.purchase = steam_data.purchase.astype(CategoricalDtype(ordered = True))
steam_data.dtypes
steam_data.purchase.head(1)
steam_data.count()
total_user_id = steam_data["User ID"].unique().sum()

total_user_id
steam_data["Name of the games"].value_counts()
total_game = steam_data.groupby('Name of the games')['Name of the games'].agg('count')

total_game
total_purchase = steam_data.groupby('purchase')['purchase'].agg('count')

total_purchase
steam_data.describe().T
steam_data.isnull().sum()
steam_data.columns
steam_data["purchase"].value_counts().plot.barh().set_title("purchase column frequency");
sns.barplot(x = "purchase", y = steam_data.purchase.index, data = steam_data)
steam_data["Hoursplay"].plot()
steam_data['Name of the games'].value_counts().head(10).plot(kind='barh', figsize=(20,10))
steam_data.Hoursplay.plot(kind = 'hist',bins = 100,figsize = (10,10))

plt.show()


sns.kdeplot(steam_data.Hoursplay, shade = True);
sns.pairplot(steam_data);
sns.pairplot(steam_data, hue="purchase");
sns.pairplot(steam_data, kind = "reg" , hue = "purchase");
sns.boxplot(x='purchase',y='Hoursplay',data=steam_data,palette='rainbow')
sns.heatmap(steam_data.corr(), annot=True)