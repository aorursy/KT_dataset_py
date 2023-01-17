# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

import seaborn as sns



from collections import Counter

import warnings 

warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('../input/top-women-chess-players/top_women_chess_players_aug_2020.csv')
df['Inactive_flag'].fillna('active',inplace=True)
data=df[df['Inactive_flag']=='active']
data.columns
data.info()
data.head()
data.describe()
## Categorical Variable 

def bar_plot(variable):

    # get feature

    var=data[variable]

    # count number of categorical variable

    var_value=var.value_counts()

    # visualization

    plt.figure(figsize=(15,3))

    plt.bar(var_value.index,var_value)

    plt.xticks(var_value.index,var_value.index.values)

    plt.ylabel('Frequency')

    plt.title(variable)

    plt.show('{}:\n {}:'.format(variable,var_value))

    

    
category1=['Title']

for c in category1 :

    bar_plot(c)
category2=['Title','Federation']

for i in category2:

    print('{} \n '.format(data[i].value_counts()))
def plot_hist(variable):

    plt.figure(figsize=(9,3))

    plt.hist(data[variable],bins=50)

    plt.xlabel(variable)

    plt.ylabel('Frequency')

    plt.title('{} distribution with hist'.format(variable))

    plt.show()
num_variable=['Year_of_birth','Standard_Rating','Rapid_rating','Blitz_rating']

for i in num_variable:

    plot_hist(i)
# free data analysis 

data[(data['Year_of_birth']>2000)&(data['Rapid_rating']>2300)]
data[(data['Year_of_birth']<1970)&(data['Rapid_rating']>2300)]
data.Name[data['Blitz_rating']==data['Blitz_rating'].max()]
data.Name[data['Standard_Rating']==data['Standard_Rating'].max()]
data['Name'][data['Rapid_rating']==data['Rapid_rating'].max()]
data['Year_of_birth'][data['Year_of_birth']==data['Year_of_birth'].min()]

# en yasli kisiyi bul 
data[data['Year_of_birth']==data['Year_of_birth'].max()]
# groupby

data.groupby('Title').Standard_Rating.mean()
data.groupby('Title').Blitz_rating.mean()
data.groupby('Title').Rapid_rating.mean()
# defining column using other columns

data['total_rating']=(data['Standard_Rating']+data['Rapid_rating']+data['Blitz_rating'])/3

data.head()

data.Name[data['total_rating']==data['total_rating'].max()]
#list comprehension

data['yas_siniri']=['20_yas_alti' if 2020-i<20 else '20_yas' if 2020-i==20 else '20_yas_ustu' for i in data['Year_of_birth']]

data.head()

# value counts 

data['yas_siniri'].value_counts()
# datayi yas a gore 3 e ayir

data1=data[data['yas_siniri']=='20_yas_ustu']

data2=data[data['yas_siniri']=='20_yas_alti']

data3=data[data['yas_siniri']=='20_yas']
data1.shape

data1.index=range(0,1822,1)
data2.shape

data2.index=range(0,737,1)
data3.shape

data3.index=range(0,142,1)
# yas gruplarina gore basari degerlendirmesi yapalim
#Plot

data1_basari=data1.loc[:,'Standard_Rating']

data1_basari.plot()

plt.title('20 Yas üstü basari')

plt.show()
# Plot

data2_basari=data2.loc[:,'Standard_Rating']

data2_basari.plot(color='red')

plt.title('20 Yas alti basari')

plt.show()
# Plot

data3_basari=data2.loc[:,'Standard_Rating']

data3_basari.plot(color='green')

plt.title('20 Yas Basari')

plt.show()