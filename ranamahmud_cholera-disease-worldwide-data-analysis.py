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
import missingno as msno

import matplotlib.pyplot as plt

import seaborn as sns

# library for seasonal decomposition

from statsmodels.tsa.seasonal import seasonal_decompose

# for map

import plotly.express as px

# set seaborn style

sns.set_style("white")

plt.style.use('fivethirtyeight')

df = pd.read_csv("../input/cholera-dataset/data.csv")
df.head()
# check data types

df.dtypes
df[df['Number of reported cases of cholera'] == '3 5']
df.isnull().sum()
# visualize missing values

msno.matrix(df);
# replace missing values in numeric columns with 0



df.replace(np.nan, '0', regex = True, inplace = True)

# check missing value count

df.isnull().sum()
# there are Unknown in cells which are creating problem

df.replace('Unknown', '0', regex = True, inplace = True)
df['Number of reported cases of cholera'] = df['Number of reported cases of cholera'].str.replace('3 5','0')

df['Number of reported deaths from cholera'] = df['Number of reported deaths from cholera'].str.replace('0 0','0')

df['Cholera case fatality rate'] = df['Cholera case fatality rate'].str.replace('0.0 0.0','0')
##### correct data types

df.Country = df.Country.astype("string")

df['Year'] = df['Year'].astype("int")

df['Number of reported cases of cholera'] = df['Number of reported cases of cholera'].astype("int64")

df['Number of reported deaths from cholera'] = df['Number of reported deaths from cholera'].astype("int64")

df['Cholera case fatality rate'] = df['Cholera case fatality rate'].astype("float")

df['WHO Region'] = df['WHO Region'].astype("string")

df.describe()
df.Country.value_counts()
df.Country.nunique()
df.head()
sns.boxplot('Number of reported cases of cholera',data = df)

plt.title("Boxplot of Number of reported cases of cholera")

plt.xlabel("Cholera Cases");
sns.boxplot('Number of reported cases of cholera',data = df)

plt.title("Boxplot of Number of Number of reported deaths from cholera")

plt.xlabel("Reported Deaths");
sns.boxplot('Number of reported cases of cholera',data = df)

plt.title("Boxplot of Number of Cholera case fatality rat")

plt.xlabel("Fatality rat");
# subset data for 2016

df_16 = df[df.Year == 2016]
# we'll exclude the countries with fatality rate 0

fig = px.sunburst(df_16[df_16['Cholera case fatality rate']!=0], path=['WHO Region', 'Country'], \

                  color = 'Cholera case fatality rate',

                  values='Number of reported deaths from cholera',hover_data=['Cholera case fatality rate'])

fig.show()
# we'll exclude the countries with fatality rate 0

fig = px.sunburst(df_16[df_16['Cholera case fatality rate']!=0], path=['WHO Region', 'Country'], \

                  color = 'Cholera case fatality rate',

                  values='Number of reported cases of cholera',hover_data=['Number of reported cases of cholera'])

fig.show()
# we'll exclude the countries with fatality rate 0

fig = px.sunburst(df_16[df_16['Cholera case fatality rate']!=0], path=['WHO Region', 'Country'], \

                  color = 'Number of reported deaths from cholera',

                  values='Number of reported cases of cholera',hover_data=['Number of reported cases of cholera'])

fig.show()
# calculate correlaiton

corr = df.drop(["Country","WHO Region"], axis = 1).corr()
corr
sns.set(rc={'figure.figsize':(11.7,8.27)})

mask = np.triu(np.ones_like(corr, dtype=np.bool))

sns.heatmap(corr,center=0,mask=mask,

            square=True, linewidths=.5, cbar_kws={"shrink": .5});
df.Year.describe()
# We'll subset data for last 10 years

df_last_ten = df[(df.Year <= 2016) & (df.Year >= 2007)]
df_last_ten.Year.nunique()
df_last_ten.describe()
# subset countries that don't have cholera in last 10 years
# count number of cases in last 10 years for each contry

total_ten = df_last_ten.groupby('Country')['Number of reported cases of cholera'].sum()
total_ten.sort_values()[0:10].plot(kind= 'bar')

plt.title("Bottom 10 Countries with Least\n amount of Cholera Cases in 2016-2007");
total_ten.sort_values(ascending=False)[0:10].plot(kind= 'bar')

plt.title("Top 10 Countries with Most\n amount of Cholera Cases in 2016-2007");
total_ten[total_ten == 0]
df_last_ten[df_last_ten.Country == "Slovenia"]
df[df.Country == "Slovenia"]
# countries with least amount of cholera cases

total_ten.sort_values()[0:10]
# subset data for 2016

df_16 = df[df.Year == 2016]
fig = px.choropleth(df_16, locations="Country", color='Number of reported cases of cholera',\

                    locationmode = 'country names',

                    hover_name="Country", animation_frame="Year")

fig.show()
fig = px.choropleth(df_16, locations="Country", color='Number of reported deaths from cholera',\

                    locationmode = 'country names',

                    hover_name="Country", animation_frame="Year")

fig.show()
fig = px.choropleth(df_16, locations="Country", color='Cholera case fatality rate',\

                    locationmode = 'country names',

                    hover_name="Country", animation_frame="Year")

fig.show()
df.groupby('Year')['Number of reported cases of cholera'].mean().plot()

plt.title("Average Number of reported cases of cholera");
df.groupby('Year')['Number of reported deaths from cholera'].mean().plot()

plt.title("Average Number of reported deaths from cholera");
df.groupby('Year')['Cholera case fatality rate'].mean().plot()

plt.title("Average Cholera case fatality rate");
df_bangladesh = df[df.Country == "Bangladesh"]

df_bangladesh.head()
fig, ax1 = plt.subplots()



color = 'tab:red'

ax1.set_xlabel('Year')

ax1.set_ylabel('Number of Cases', color=color)

ax1.plot(df_bangladesh.Year,df_bangladesh[ 'Number of reported cases of cholera'] , color="#fa26a0",\

        label = 'Cases')

ax1.plot(df_bangladesh.Year,df_bangladesh[ 'Number of reported deaths from cholera'] , color="tab:red",\

        label="Deaths")

ax1.tick_params(axis='y', labelcolor=color)



ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis



color = 'tab:blue'

ax2.set_ylabel('Fatality Rate', color=color)  # we already handled the x-label with ax1

ax2.plot(df_bangladesh.Year, df_bangladesh['Cholera case fatality rate'], color=color,\

        label = 'Fatality rate')

ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped

ax1.legend(loc="upper right")

ax2.legend(loc="center right")

plt.title("Bangladesh Cholera Status 1973-2000")

plt.show()