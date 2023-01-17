# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import warnings

warnings.filterwarnings('ignore')

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_dc = pd.read_csv('../input/dc-wikia-data.csv')
data_dc.head()
data_dc.tail()
data_dc.sample(5)
data_dc.describe()
data_dc.info()
data_dc.columns
data_dc = data_dc.rename(columns={'page_id':'Page_id',

                                  'name':'Name',

                                  'urlslug':'Urlslug',

                                  'ID':'ID',

                                  'ALIGN':'Align',

                                  'EYE':'Eye',

                                  'HAIR':'Hair',

                                  'SEX':'Gender',

                                  'GSM':'GSM',

                                  'ALIVE':'Alive',

                                  'APPEARANCES':'Appearances',

                                  'FIRST APPEARANCE':'FirstAppearances',

                                  'YEAR':'Year'})
data_dc['Inc'] = 'DC'
data_dc.head()
data_marvel = pd.read_csv('../input/marvel-wikia-data.csv')
data_marvel.head()
data_marvel.tail()
data_marvel.sample(5)
data_marvel.describe()
data_marvel.info()
data_marvel.columns
data_marvel = data_marvel.rename(columns={'page_id':'Page_id',

                                          'name':'Name',

                                          'urlslug':'Urlslug',

                                          'ID':'ID',

                                          'ALIGN':'Align',

                                          'EYE':'Eye',

                                          'HAIR':'Hair',

                                          'SEX':'Gender',

                                          'GSM':'GSM',

                                          'ALIVE':'Alive',

                                          'APPEARANCES':'Appearances',

                                          'FIRST APPEARANCE':'FirstAppearances',

                                          'Year':'Year'})
data_marvel['Inc'] = 'Mervel'
data = pd.concat([data_dc,data_marvel])
data.describe()
data.info()
data.isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sns

def drow_pie(dataset,column):

    f,ax=plt.subplots(1,2,figsize=(18,9))

    explode_list = [0.1] * (dataset[dataset['Inc'] == 'DC'][column].unique().size-1)

    dataset[dataset['Inc'] == 'DC'][column].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[0],shadow=True)

    ax[0].set_title('DC {} Count'.format(column))

    ax[0].set_ylabel('Count')

    explode_list = [0.1] * (dataset[dataset['Inc'] == 'Mervel'][column].unique().size-1)

    dataset[dataset['Inc'] == 'Mervel'][column].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[1],shadow=True)

    ax[1].set_title('Mervel {} Count'.format(column))

    ax[1].set_ylabel('Count')

    plt.show()
drow_pie(data,'ID')
plt.figure(figsize=(20,5))

sns.set_context("paper", 2.0, {"lines.linewidth": 4})

sns.countplot(x='ID',hue='Inc',data=data)
drow_pie(data,'Align')
plt.figure(figsize=(20,5))

sns.set_context("paper", 2.0, {"lines.linewidth": 4})

sns.countplot(x='Align',hue='Inc',data=data)
drow_pie(data,'Eye')
plt.figure(figsize=(10,20))

sns.set_context("paper", 2.0, {"lines.linewidth": 4})

sns.countplot(y='Eye',hue='Inc',data=data)
drow_pie(data,'Hair')
plt.figure(figsize=(10,20))

sns.set_context("paper", 2.0, {"lines.linewidth": 4})

sns.countplot(y='Hair',hue='Inc',data=data)
drow_pie(data,'Gender')
plt.figure(figsize=(25,5))

sns.set_context("paper", 2.0, {"lines.linewidth": 4})

sns.countplot(x='Gender',hue='Inc',data=data)
drow_pie(data,'GSM')
plt.figure(figsize=(25,5))

sns.set_context("paper", 2.0, {"lines.linewidth": 4})

sns.countplot(x='GSM',hue='Inc',data=data)
drow_pie(data,'Alive')
plt.figure(figsize=(20,5))

sns.set_context("paper", 2.0, {"lines.linewidth": 4})

sns.countplot(x='Alive',hue='Inc',data=data)
import numpy as np

plt.figure(figsize=(20,5))

sns.distplot(data['Appearances'],hist=False,bins=1)
sns.FacetGrid(data, hue="Inc", size=8).map(sns.kdeplot, "Appearances").add_legend()

plt.ioff() 

plt.show()
# Select the top 100 appearances

data_top_100_dc = data[data['Inc'] == 'DC'].nlargest(100,'Appearances')  

data_top_100_dc.shape
data_top_100_dc.head()
# Select the top 100 appearances

data_top_100_mervel = data[data['Inc'] == 'Mervel'].nlargest(100,'Appearances')  

data_top_100_mervel.shape
data_top_100_mervel.head()
data_top_100 = pd.concat([data_top_100_dc,data_top_100_mervel])

data_top_100.shape
sns.FacetGrid(data_top_100, hue="Inc", size=8).map(sns.kdeplot, "Appearances").add_legend()

plt.ioff() 

plt.show()
f,ax=plt.subplots(1,2,figsize=(18,9))

explode_list = [0.1] * (data_top_100[data_top_100['Inc'] == 'DC']['ID'].unique().size)

data_top_100[data_top_100['Inc'] == 'DC']['ID'].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('DC {} Count'.format('ID'))

ax[0].set_ylabel('Count')

explode_list = [0.1] * (data_top_100[data_top_100['Inc'] == 'Mervel']['ID'].unique().size)

data_top_100[data_top_100['Inc'] == 'Mervel']['ID'].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[1],shadow=True)

ax[1].set_title('Mervel {} Count'.format('ID'))

ax[1].set_ylabel('Count')

plt.show()
plt.figure(figsize=(20,5))

sns.set_context("paper", 2.0, {"lines.linewidth": 4})

sns.countplot(x='ID',hue='Inc',data=data_top_100)
plt.figure(figsize=(15,10))

sns.swarmplot(x="Inc", y="Appearances",hue='ID', data=data_top_100)
plt.figure(figsize=(15,10))

sns.boxplot(x="ID", y="Appearances",

            hue="Inc", palette=["m", "g"],

            data=data_top_100)
f,ax=plt.subplots(1,2,figsize=(18,9))

explode_list = [0.1] * (data_top_100[data_top_100['Inc'] == 'DC']['Align'].unique().size-1)

data_top_100[data_top_100['Inc'] == 'DC']['Align'].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('DC {} Count'.format('Align'))

ax[0].set_ylabel('Count')

explode_list = [0.1] * (data_top_100[data_top_100['Inc'] == 'Mervel']['Align'].unique().size-1)

data_top_100[data_top_100['Inc'] == 'Mervel']['Align'].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[1],shadow=True)

ax[1].set_title('Mervel {} Count'.format('Align'))

ax[1].set_ylabel('Count')

plt.show()
plt.figure(figsize=(20,5))

sns.set_context("paper", 2.0, {"lines.linewidth": 4})

sns.countplot(x='Align',hue='Inc',data=data_top_100)
plt.figure(figsize=(15,10))

sns.swarmplot(x="Inc", y="Appearances",hue='Align', data=data_top_100)
plt.figure(figsize=(20,10))

sns.boxplot(x="Align", y="Appearances",

            hue="Inc", palette=["m", "g"],

            data=data_top_100)
f,ax=plt.subplots(1,2,figsize=(18,9))

explode_list = [0.1] * (data_top_100[data_top_100['Inc'] == 'DC']['Eye'].unique().size-1)

data_top_100[data_top_100['Inc'] == 'DC']['Eye'].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('DC {} Count'.format('Eye'))

ax[0].set_ylabel('Count')

explode_list = [0.1] * (data_top_100[data_top_100['Inc'] == 'Mervel']['Eye'].unique().size)

data_top_100[data_top_100['Inc'] == 'Mervel']['Eye'].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[1],shadow=True)

ax[1].set_title('Mervel {} Count'.format('Eye'))

ax[1].set_ylabel('Count')

plt.show()
plt.figure(figsize=(10,10))

sns.set_context("paper", 2.0, {"lines.linewidth": 4})

sns.countplot(y='Eye',hue='Inc',data=data_top_100)
plt.figure(figsize=(15,10))

sns.swarmplot(x="Inc", y="Appearances",hue='Eye', data=data_top_100)
plt.figure(figsize=(15,25))

sns.boxplot(y="Eye", x="Appearances",

            hue="Inc", palette=["m", "g"],

            data=data_top_100)
f,ax=plt.subplots(1,2,figsize=(18,9))

explode_list = [0.1] * (data_top_100[data_top_100['Inc'] == 'DC']['Hair'].unique().size-1)

data_top_100[data_top_100['Inc'] == 'DC']['Hair'].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('DC {} Count'.format('Hair'))

ax[0].set_ylabel('Count')

explode_list = [0.1] * (data_top_100[data_top_100['Inc'] == 'Mervel']['Hair'].unique().size)

data_top_100[data_top_100['Inc'] == 'Mervel']['Hair'].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[1],shadow=True)

ax[1].set_title('Mervel {} Count'.format('Hair'))

ax[1].set_ylabel('Count')

plt.show()
plt.figure(figsize=(10,10))

sns.set_context("paper", 2.0, {"lines.linewidth": 4})

sns.countplot(y='Hair',hue='Inc',data=data_top_100)
plt.figure(figsize=(15,10))

sns.swarmplot(x="Inc", y="Appearances",hue='Hair', data=data_top_100)
plt.figure(figsize=(15,25))

sns.boxplot(y="Hair", x="Appearances",

            hue="Inc", palette=["m", "g"],

            data=data_top_100)
f,ax=plt.subplots(1,2,figsize=(18,9))

explode_list = [0.1] * (data_top_100[data_top_100['Inc'] == 'DC']['Gender'].unique().size)

data_top_100[data_top_100['Inc'] == 'DC']['Gender'].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('DC {} Count'.format('Gender'))

ax[0].set_ylabel('Count')

explode_list = [0.1] * (data_top_100[data_top_100['Inc'] == 'Mervel']['Gender'].unique().size)

data_top_100[data_top_100['Inc'] == 'Mervel']['Gender'].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[1],shadow=True)

ax[1].set_title('Mervel {} Count'.format('Gender'))

ax[1].set_ylabel('Count')

plt.show()
plt.figure(figsize=(20,5))

sns.set_context("paper", 2.0, {"lines.linewidth": 4})

sns.countplot(x='Gender',hue='Inc',data=data_top_100)
data_top_100[data_top_100['Gender']=='Genderfluid Characters']
plt.figure(figsize=(15,10))

sns.swarmplot(x="Inc", y="Appearances",hue='Gender', data=data_top_100)
plt.figure(figsize=(20,10))

sns.boxplot(x="Gender", y="Appearances",

            hue="Inc", palette=["m", "g"],

            data=data_top_100)
f,ax=plt.subplots(1,2,figsize=(18,9))

explode_list = [0.1] * (data_top_100[data_top_100['Inc'] == 'DC']['GSM'].unique().size-1)

data_top_100[data_top_100['Inc'] == 'DC']['GSM'].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('DC {} Count'.format('GSM'))

ax[0].set_ylabel('Count')

explode_list = [0.1] * (data_top_100[data_top_100['Inc'] == 'Mervel']['GSM'].unique().size-1)

data_top_100[data_top_100['Inc'] == 'Mervel']['GSM'].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[1],shadow=True)

ax[1].set_title('Mervel {} Count'.format('GSM'))

ax[1].set_ylabel('Count')

plt.show()
plt.figure(figsize=(20,5))

sns.set_context("paper", 2.0, {"lines.linewidth": 4})

sns.countplot(x='GSM',hue='Inc',data=data_top_100)
plt.figure(figsize=(15,10))

sns.swarmplot(x="Inc", y="Appearances",hue='GSM', data=data_top_100)
plt.figure(figsize=(20,10))

sns.boxplot(x="GSM", y="Appearances",

            hue="Inc", palette=["m", "g"],

            data=data_top_100)
f,ax=plt.subplots(1,2,figsize=(18,9))

explode_list = [0.1] * (data_top_100[data_top_100['Inc'] == 'DC']['Alive'].unique().size)

data_top_100[data_top_100['Inc'] == 'DC']['Alive'].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('DC {} Count'.format('Alive'))

ax[0].set_ylabel('Count')

explode_list = [0.1] * (data_top_100[data_top_100['Inc'] == 'Mervel']['Alive'].unique().size)

data_top_100[data_top_100['Inc'] == 'Mervel']['Alive'].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[1],shadow=True)

ax[1].set_title('Mervel {} Count'.format('Alive'))

ax[1].set_ylabel('Count')

plt.show()
plt.figure(figsize=(20,5))

sns.set_context("paper", 2.0, {"lines.linewidth": 4})

sns.countplot(x='Alive',hue='Inc',data=data_top_100)
plt.figure(figsize=(15,10))

sns.swarmplot(x="Inc", y="Appearances",hue='Alive', data=data_top_100)
plt.figure(figsize=(20,10))

sns.boxplot(x="Alive", y="Appearances",

            hue="Inc", palette=["m", "g"],

            data=data_top_100)