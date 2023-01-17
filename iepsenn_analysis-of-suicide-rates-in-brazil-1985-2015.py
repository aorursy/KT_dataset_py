# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/master.csv")

print(df.columns)

df.sample(1)
data = df[df.country == 'Brazil']

print(data.shape)

data.sample(1)
countries = df.country.unique().tolist()

countries[:5]
suicide_no = []

for c in countries:

    s = 0 

    for i in range(0, len(df.country)):

        if(c==df.country[i]):

            s += df.iloc[i]['suicides_no']

    suicide_no.append(s)
print(countries[0])

print(suicide_no[0])
d = {'country' : countries, 'total_suicides' : suicide_no}

suicide_world = pd.DataFrame(data=d)

indexes = suicide_world['total_suicides'].sort_values(ascending=False)[:10].index.tolist()

top_suicide_no = []

top_countries = []

for i in indexes:

    top_countries.append(suicide_world.iloc[i].country)

    top_suicide_no.append(suicide_world.iloc[i]['total_suicides'])

    

print(top_countries)

print(top_suicide_no)
labels = top_countries

sizes = [38.4, 40.6, 20.7, 10.3]

colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'blue', 'brown', 'tomato', 'magenta', 'darkseagreen', 'purple']

patches, texts = plt.pie(top_suicide_no, colors=colors, shadow=True, startangle=90)

plt.legend(patches, labels, loc="best")

plt.axis('equal')

plt.tight_layout()

plt.show()
index = np.arange(len(top_countries))

plt.bar(top_countries, top_suicide_no)

#plt.xlabel('Countries', fontsize=5)

plt.ylabel('Number of Suicides', fontsize=5)

plt.xticks(index, top_countries, fontsize=5, rotation=30)

plt.title('Top 10 countries with cases of suicide')

plt.show()
data.describe()
data.columns
female = data.groupby(by=['sex', 'age'])['suicides_no'].sum()[:6]

male = data.groupby(by=['sex', 'age'])['suicides_no'].sum()[6:]

f = []

m = []

fn = []

mn = []

for i in range(6):

    f.append(female.index[i][1].split(' ')[0])

    m.append(male.index[i][1].split(' ')[0])

    fn.append(female[i])

    mn.append(male[i])

print(f)

print(m)
index = np.arange(len(f))

plt.bar(f, fn)

#plt.xlabel('Countries', fontsize=5)

plt.ylabel('Number of Suicides', fontsize=5)

plt.xticks(index, f, fontsize=8, rotation=30)

plt.title('age x suicide rate  - women')

plt.show()
index = np.arange(len(m))

plt.bar(m, mn)

#plt.xlabel('Countries', fontsize=5)

plt.ylabel('Number of Suicides', fontsize=5)

plt.xticks(index, f, fontsize=8, rotation=30)

plt.title('age x suicide rate - men')

plt.show()
print('total female: ', sum(fn))

print('total male: ', sum(mn))
d = {'women' : [sum(fn)], 'men' : [sum(mn)]}

perSex = pd.DataFrame(data=d)

perSex.plot.bar()
year = data.groupby(by=['country-year'])['suicides_no'].sum().index.tolist()

suicide_no = data.groupby(by=['country-year'])['suicides_no'].sum().values.tolist()
year_ = [d[6:] for d in year]



print(year_[:5])

print(suicide_no[:5])
corr_year = data.pivot_table(index='year',

                                  values='suicides_no', aggfunc=np.median)

corr_year
d = {'year' : year_, 'suicides' : suicide_no}

perYear = pd.DataFrame(data=d)

perYear.plot.bar(x='year',y='suicides')

plt.title("suicides per year")

plt.show()
data.plot.scatter(x='year',y='HDI for year')

plt.title("increase of the HDI over the years")

plt.show()
data.plot.line(x='year',y='gdp_per_capita ($)')

plt.title("year x gdp (PIB)")

data.plot.hist(x='suicides_no',y='gdp_per_capita ($)')

plt.title("suicide rate x gdp (PIB)")

plt.show()
data.plot.hexbin(x='suicides_no',y='population', gridsize=8, color='lightgreen')

plt.title("suicide rate x population")

plt.show()