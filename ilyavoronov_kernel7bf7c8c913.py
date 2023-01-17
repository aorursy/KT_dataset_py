# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import seaborn as sns
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
file_path='../input/google-play-store-apps/googleplaystore.csv'
data = pd.read_csv(file_path)

#группировка по Genres
data=data.groupby(['Genres'])['Rating','Genres'].mean()
data.sort_values(by=['Rating'], ascending=False,inplace=True)
#выбираем лучшие
data.reset_index(level=0, inplace=True)
data = datas.iloc[1:10]
plt.figure(figsize=(20,10))

sns.barplot(x=data['Genres'], y=data['Rating'])