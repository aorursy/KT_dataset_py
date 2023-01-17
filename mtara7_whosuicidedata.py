# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
suicide_data=pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')

n_rows=suicide_data.shape[0]

n_columns=suicide_data.shape[1]

print("The No Of Sample in Data is {}, The No Of Features in Data is {}".format(n_rows,n_columns))


suicide_data.columns
import seaborn as sns

suicide_rate_year=suicide_data.groupby('year')['suicides_no'].sum()
ax=sns.lineplot(data=suicide_rate_year)

ax.set_xlabel('year')

ax.set_ylabel('suicides_no')

ax.set_title('Suicides Per Year')

plt.show()


sns.relplot(x='year',y='suicides/100k pop',hue='sex',kind='line',data=suicide_data)

plt.show()

plt.title('Population Distribution according to age')

sns.barplot(y='age',x='population',data=suicide_data)

plt.show()
plt.title('Suicide Distribution according to age')

sns.barplot(y='age',x='suicides_no',data=suicide_data)

plt.show()
# pull out top 10 country where suicide rate is higher

top_10_suicidal_country=suicide_data.groupby('country')['suicides/100k pop'].sum().sort_values(ascending=False).head(10)



# Convert their index to a list which will be used to filter out data from original dataset

top_10_suicidal_country=top_10_suicidal_country.index.to_list()



#filter out data and group bt country-year feature

top_10_suicidal_country_data=suicide_data.loc[suicide_data['country'].isin (top_10_suicidal_country),['country-year','gdp_per_capita ($)','suicides/100k pop']]

top_10_suicidal_country_data=top_10_suicidal_country_data.groupby('country-year').agg({'suicides/100k pop':'sum','gdp_per_capita ($)':'mean'})



#visualize

plt.figure(figsize=(8,8))

sns.regplot(x='gdp_per_capita ($)',y='suicides/100k pop',data=top_10_suicidal_country_data)

plt.show()