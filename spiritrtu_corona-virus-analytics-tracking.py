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
import pandas as pd

import folium

df=pd.read_csv("/kaggle/input/covid19-corona-virus-india-dataset/covid19_data_map.csv")

df.head()
df.describe()
df.describe(include='all')
df.info()
df.dtypes

df.isnull().values.any()
df.isnull().sum()




#show columns

for i,col in enumerate(df.columns):

    print(i+1,". column is ",col)



df['Deaths**'].value_counts()
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(7,7))

ax=sns.barplot(x=df['Deaths**'].value_counts().index,

              y=df['Deaths**'].value_counts().values,

              palette=sns.cubehelix_palette(120))

plt.xlabel('Deaths')

plt.ylabel('Frequency')

plt.title('Show of Deaths frequencyBar Plot')

plt.show()
missing_data=df.isnull()

for column in missing_data.columns.values.tolist():

    print(column)

    print(missing_data[column].value_counts())

    print("")
f, ax = plt.subplots(figsize=(12, 8))



sns.set_color_codes("pastel")

sns.barplot(x="Total Confirmed cases*", y="Name of State / UT", data=df,

            label="Confirmed", color="b")

sns.set_color_codes("muted")

sns.barplot(x="Cured/Discharged/Migrated*", y="Name of State / UT", data=df,

            label="Recovered", color="g")

# Add a legend and informative axis label

ax.legend(ncol=2, loc="lower right", frameon=True)

sns.despine(left=True, bottom=True)
x=df['Total Confirmed cases*'].sum()

y=df['Deaths**'].sum()

z=df['Cured/Discharged/Migrated*'].sum()

plt.figure(figsize=(10,5))

labels=['Confirmed','Deaths','Recovered']

colors=['pink','red','silver']

explode=[0.5,0,0]

values=[x,y,z]



plt.pie(values,labels=labels,colors=colors,explode=explode,autopct='%1.1f%%',shadow=True)

plt.legend(['Confirmed','Deaths','Recovered'] , loc=0)

plt.axis('equal')

plt.tight_layout()

plt.show()
labels = ('Confirmed','Deaths','Recovered')

x_index = [0,1,2]

# indexes is the first parameter 

plt.bar(x_index, [x,y,z], width = 0.8, align='center', alpha=1.0)

plt.xticks(x_index, labels)

plt.xlabel('cases')

plt.ylabel('Count')

plt.title('India covid 19')

plt.show()