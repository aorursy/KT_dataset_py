# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/obesity-levels/ObesityDataSet_raw_and_data_sinthetic.csv', encoding='ISO-8859-2')

df.head()
print(f"data shape: {df.shape}")
df.describe()
plt.figure(figsize=(20,4))

plt.subplot(131)

sns.countplot(x= 'Age', data = df, palette="GnBu_d",edgecolor="black")

plt.subplot(132)

sns.countplot(x= 'Height', data = df, palette="flag",edgecolor="black")

plt.subplot(133)

sns.countplot(x= 'Weight', data = df, palette="Greens_r",edgecolor="black")

plt.show()
#Codes from Gabriel Preda



def plot_count(feature, title, df, size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')

    g.set_title("Number and percentage of {}".format(title))

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()
plot_count("Weight", "Weight", df,4)
plt.figure(figsize=(20,4))

plt.subplot(131)

sns.countplot(x= 'family_history_with_overweight', data = df, palette="GnBu_d",edgecolor="black")

plt.subplot(132)

sns.countplot(x= 'MTRANS', data = df, palette="flag",edgecolor="black")

plt.xticks(rotation=45)

plt.subplot(133)

sns.countplot(x= 'CALC', data = df, palette="Greens_r",edgecolor="black")

plt.xticks(rotation=45)

plt.show()
sns.countplot(x="NObeyesdad",data=df,palette="flag",edgecolor="black")

plt.title('NObeyesdad', weight='bold')

plt.xticks(rotation=45)

plt.yticks(rotation=45)

# changing the font size

sns.set(font_scale=1)
plt.figure(figsize=(16, 4))

plt.subplot(121)

sns.boxplot(x = 'Weight', y = 'SCC', data = df)

plt.subplot(122)

sns.boxplot(x = 'Weight', y = 'CAEC', data = df)

plt.show()
# Distribution of different type of amount

fig , ax = plt.subplots(1,3,figsize = (12,5))



Age = df.Age.values

Height= df.Height.values

Weight = df.Weight.values



sns.distplot(Age , ax = ax[0] , color = 'pink').set_title('Obesity Levels & Age' , fontsize = 14)

sns.distplot(Height , ax = ax[1] , color = 'cyan').set_title('Obesity Levels & Height' , fontsize = 14)

sns.distplot(Weight , ax = ax[2] , color = 'purple').set_title('Obesity Levels & Weight' , fontsize = 14)





plt.show()
import plotly.express as px

fig = px.line(df, x="Weight", y="TUE", color_discrete_sequence=['darksalmon'], 

              title="Obesity Levels and the Use of Technology")

fig.show()