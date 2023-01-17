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
import seaborn as sns
import matplotlib.pyplot as plt
import missingno
import plotly.express as px
data = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv',sep=',',encoding='utf-8')
data.head()
data.tail()
data.info()
data.isnull().sum()
missingno.bar(data)
missingno.heatmap(data)
fig,ax = plt.subplots(figsize=(10,5))
sns.countplot(x = data['type'])
fig2 , ax2 = plt.subplots(figsize = (5,5))
ax2.pie(data['type'].value_counts(),shadow=True, startangle=90,explode = (0.1,0),autopct='%1.1f%%', labels = data.type.unique().tolist())
ax2.axis('equal')
plt.tight_layout()
plt.legend(loc = "upper right")
plt.show()
fig,ax = plt.subplots(figsize=(30,30))
sns.countplot(y = data['duration'],hue = data.type);
plt.title("total number of times")
d2016 = data['release_year'] == 2016
data_2016 = data[d2016]

d2017 = data['release_year'] == 2017
data_2017 = data[d2017]

d2018 = data['release_year'] == 2018
data_2018 = data[d2018]

d2019 = data['release_year'] == 2019
data_2019 = data[d2019]
fig,ax = plt.subplots(figsize=(10,10))
sns.set_color_codes("pastel")
sns.barplot(data["rating"].value_counts().index,data["rating"].value_counts().values, color="b")
fig,ax = plt.subplots(figsize=(10,10))
plt.bar(data_2016["rating"].value_counts().index,data_2016["rating"].value_counts().values, color="b",label="2016")
plt.bar(data_2017["rating"].value_counts().index,data_2017["rating"].value_counts().values, color="r",label="2017")
plt.bar(data_2018["rating"].value_counts().index,data_2018["rating"].value_counts().values, color="y",label="2018")
plt.bar(data_2019["rating"].value_counts().index,data_2019["rating"].value_counts().values, color="g",label="2019")
ax.legend();
data_2019.head()
fig1, ax1 = plt.subplots(1,2,figsize=(25,10))
ax1[0].pie(data_2019.type.value_counts(), explode = (0,0.1),shadow = True, autopct='%1.1f%%', labels = data_2019.type.unique().tolist());
ax1[1].pie(data_2019["country"].value_counts(),autopct='%1.1f%%',labels = data_2019["country"].value_counts().index.tolist(),shadow = True);
d2019_turkey = data_2019['country'] == "Turkey"
data_2019_turkey = data_2019[d2019_turkey]
data_2019_turkey
data_2019_turkey.info()
dTurkey = data['country'] == "Turkey"
data_Turkey = data[dTurkey]
data_Turkey.head()
data_Turkey.info()
data_Turkey.director.unique().tolist()
