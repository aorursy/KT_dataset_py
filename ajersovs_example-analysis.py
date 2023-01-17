

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from scipy.stats import ttest_ind, ttest_rel

print(os.listdir("../input"))
df = pd.read_csv("../input/master.csv")

df.head()
df.info()
c=df.columns

for col in c:

    cos = df[col].isnull().sum()

    print(('column {} has ' + str(cos) + ' missing values').format(col))
df = df.drop(['HDI for year','country-year'], axis = 1)

df.head()
df_num = df.drop(['country','sex','generation'] ,axis = 1)

df_num.describe()

df_cats = df.drop(['year','suicides_no','population','suicides/100k pop',\

                   'gdp_per_capita ($)',' gdp_for_year ($) '], axis=1)

df_cats.describe()
df_num = df_num[df_num['suicides_no'] < 2000]

fig, axarr = plt.subplots(2, 2, figsize=(10, 10))

df_num['year'].hist(ax=axarr[0][0])

axarr[0][0].set_title("year", fontsize=18)

df_num['suicides_no'].hist(ax=axarr[0][1], bins = 25)

axarr[0][1].set_title("suicides_no", fontsize=18)

df_num['population'].hist(ax=axarr[1][0])

axarr[1][0].set_title("population", fontsize=18)

df_num['suicides/100k pop'].hist(ax=axarr[1][1], bins = 20)

axarr[1][1].set_title("suicides/100k pop", fontsize=18)
fig, axarr = plt.subplots(2, 2, figsize=(10, 10))

x = np.log10(df['suicides_no'].replace(0, np.nan).dropna()).hist(ax=axarr[0][0], bins =20)

axarr[0][0].set_title("suicides_no", fontsize=18)

x = np.log10(df['population'].replace(0, np.nan).dropna()).hist(ax=axarr[0][1], bins =20)

axarr[0][1].set_title("population", fontsize=18)

x = np.log10(df['suicides/100k pop'].replace(0, np.nan).dropna()).hist(ax=axarr[1][0], bins =20)

axarr[1][0].set_title("suicides/100k pop", fontsize=18)



def ttest_decade(year1,year2):

    male = df[df['year']==year1]["suicides_no"]

    female = df[df['year']==year2]["suicides_no"]



    ttest,pval = ttest_ind(male,female)



    if pval <0.05:

        return print("ttest",ttest.round(3),"\n"\

                     "p-value",pval.round(3),"\nThere -IS- significant difference "\

                     "between sample means of {} and {} years\n".format(year1,year2))

    elif pval == 1:

        return None

    else:

        return print("ttest",ttest.round(3),"\n"\

                     "p-value",pval.round(3),"\nThere is -NO- significant difference "\

                     "between sample means of {} and {} years\n".format(year1,year2))

    

Y1 = [1985,1995]

Y2 = [1995,2005,2015]

import itertools

for year_1, year_2 in itertools.product(Y1, Y2):

    ttest_decade(year_1,year_2)
first = df[df['year']==1985]['suicides_no'].mean()

second = df[df['year']==2015]['suicides_no'].mean()

print('Suicides overall rate increased for {:.2f}% during last 30 years'.format(((second / first)-1)*100))



first = df[df['year']==1985]['population'].mean()

second = df[df['year']==2015]['population'].mean()

print('Suicides overall rate increased for {:.2f}% during last 30 years'.format(((second / first)-1)*100))
# Drop 2016 due to lack of records on this year

df=df[df['year']!=2016]



plt.figure(figsize = (12,5))

df.most_pop_countries= df.groupby('country')['suicides_no'].sum().sort_values(ascending = False)[:10]

df.most_pop_countries.plot(kind='bar', fontsize=12)

plt.ylabel('suicides number')

plt.title('Top-10 countries by suicides 1985-2015', fontsize = 20)
cats = ['5-14 years', '15-24 years', '25-34 years', '35-54 years', '55-74 years']

cat_dtype = pd.api.types.CategoricalDtype(categories=cats, ordered=True)

df_cats['age'] = df['age'].astype(cat_dtype)



suicides_rate_per_years = df.groupby(['year','age'])['suicides_no'].sum()

plt.figure(figsize=(12,5))

suicides_rate_per_years.unstack().plot(kind='line',linewidth=5.0, figsize=(15,7), grid=5, fontsize =12)

plt.ylabel('suicides number')

plt.title('Suicide number by age group 1985-2015',fontsize = 20)
d = { 'G.I. Generation':['1910-1924'],'Silent':['1925-1945'],'Boomers':['1946-1964'],\

     'Generation X':['1965-1979'],'Millenials':['1980-1994'],'Generation Z':['1995-2012']}

col=['period']

gens = pd.DataFrame(data=d, index=col).T

gens
plt.figure(figsize=(12,5))

total = float(len(df_cats) )



ax = sns.countplot(x="generation", data=df_cats)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.1f}%'.format((height/total)*100),

            ha="center") 

plt.title('Relative weight of each age group 1985-2015',fontsize = 20)

plt.show()


suicides_rate_per_gen = df.groupby(['year','generation'])['suicides_no'].sum()

suicides_rate_per_gen.unstack().plot(kind='line',linewidth=6.0, figsize=(12,5), fontsize=12)

plt.title("Suicide numbers by generations 1985-2015", fontsize = 20)

plt.ylabel('suicides number')
pal = ['pink','powderblue']

suicides_rate = df.groupby(['year','sex'])['suicides_no'].sum()

suicides_rate.unstack().plot(kind='area',linewidth=3.0, figsize=(12,5), colors = pal, fontsize=12)

plt.ylabel('suicides number')

plt.title('Suicide rates by gender 1985-2015', fontsize = 20)
adf = df.copy()

adf=df.loc[df['country'].isin(['Russian Federation','Latvia','Estonia','Ukraine','Belarus',\

                               'Lithuania','United States'])]

x=pd.crosstab(adf['country'], df['year'], values = df['suicides/100k pop'], aggfunc = 'sum').round(0)

x = x.apply(lambda row: row.fillna(row.mean()), axis=1)

plt.figure(figsize=(15,8))

sns.heatmap(x)

plt.title('Heatmap of suicide rate per countries per year', fontsize = 25)

plt.ylabel('')

# Delete 3 biggest suicide countries (Russia, USA, Japan) to ease up the correlation graph interpretation

df_without_top3 = df[(df['country']!='Russian Federation')&(df['country']!='United States')\

        &(df['country']!='Japan')]

#Make a pivot table with summarized suicides numbers each year per each country

df_sui = pd.pivot_table(df_without_top3, values='suicides_no', index='country', columns='year', \

                        aggfunc = 'sum')



#Fill in all the gaps with mean values to receive the same sample sizes for every country

df_sui = df_sui.apply(lambda row: row.fillna(row.mean()), axis=1)

#Create new summarizing column with average values

df_sui['Avg_suicide'] = df_sui.apply(lambda row: row.mean(), axis=1).round(0)

country_mean_suicide = df_sui[['Avg_suicide']]



#Repeat all above with second feature

df_gdp = pd.pivot_table(df_without_top3, values='gdp_per_capita ($)', index='country', columns='year')

df_gdp = df_gdp.apply(lambda row: row.fillna(row.mean()), axis=1)

df_gdp['Avg_gdp'] = df_gdp.apply(lambda row: row.mean(), axis=1).round(0)

country_mean_gdp = df_gdp[['Avg_gdp']]



plt.figure(figsize=(12,5))

plt.scatter(x=country_mean_gdp, y=country_mean_suicide, s = 100)

plt.title('Suicide rate and GDP per capita correlation 1985-2016', fontsize = 20)

plt.xlabel('Average GDP per capita ($)')

plt.ylabel('Average suicides rate per year ')