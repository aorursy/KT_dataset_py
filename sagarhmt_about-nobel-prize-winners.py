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
import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud

%matplotlib inline
df = pd.read_csv('/kaggle/input/a-visual-history-of-nobel-prize-winners-dataset/nobel.csv')

df.head()
df.tail()
df.info()
def percentage_teller(DF):

    for column in DF.columns:

        null_values = DF[column].isnull().sum()

        total_values = df.shape[0]

        percent = (null_values/ total_values)*100

        print("In {0} column {1:.2f} % values are null".format(column, percent))

percentage_teller(df) 
df.drop(['death_date', 'death_city', 'death_country', 'laureate_id', 'organization_name', 'organization_city']

        , axis = 1, inplace = True)
def auto_label(graph):

    for bar in graph.patches:

        height = bar.get_height()

        plt.annotate(height,

        xy = (bar.get_x() + bar.get_width()/2, height),

        xytext = (0, 3),

        textcoords="offset points", va = 'bottom', ha = 'center')
plt.figure(figsize = (7, 6))

Graph1 = sns.countplot(df['category'])

auto_label(Graph1)
df['birth_country'].unique()
countries = df['birth_country'].copy()

countries.replace({'United States of America':'USA', 'United Kingdom': 'UK'}, inplace = True)

countries.dropna(inplace = True)

countries = countries.str.strip()



COUNTRIES = list()



def remove_bracket(Countries):

    for i in countries:

        if i.endswith(')'):

            i = i.split(' ')[-1]

            n = len(i)

            COUNTRIES.append(i[1:n-1])

        else:

            COUNTRIES.append(i)

            

            

remove_bracket(countries)

text = " ".join(country for country in COUNTRIES)



wc = WordCloud()

wc.generate(text)



plt.imshow(wc, interpolation='bilinear')

plt.axis('off')

plt.margins(x = 0, y = 0)

plt.show()
country = pd.Series(COUNTRIES)

top_10_country = country.value_counts().nlargest(10).index

nobel = country.value_counts().nlargest(10).values

with plt.style.context('fivethirtyeight'):

    graph2 = plt.barh(top_10_country[::-1], nobel[::-1])

    for i in graph2:

        width = i.get_width()

        plt.annotate(width, xy = (width, i.get_y() + i.get_height()/2),

                     xytext = (0, 6),

                     textcoords = 'offset points', va = 'top')
Female = (df['sex'].value_counts()['Female'] / df['sex'].value_counts().sum()) * 100

Male = 100 - Female



plt.pie([Male, Female], labels = ['Male', 'Female'], shadow = True, wedgeprops={'edgecolor': 'black'},

       autopct='%1.1f%%')
year = df['year'] - pd.DatetimeIndex(df['birth_date']).year

age_young = year.nsmallest(5).astype('int64')

name_young = df.iloc[[885, 85, 166, 171, 189], 6]

youngest = pd.concat([name_young, age_young], axis = 1).reset_index().drop(columns = 'index')

youngest.rename(columns = {0:'age', 'full_name':'Name'})