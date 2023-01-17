import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

%pylab inline
data = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')

data = data.drop(['country-year','HDI for year'],axis=1)

print(data.shape)

data.head()
data.info()
data[['suicides_no','population','suicides/100k pop','gdp_per_capita ($)']].describe()
from IPython.display import Image

Image('../input/country/missed countries.png')
suic_sum = pd.DataFrame(data['suicides_no'].groupby(data['country']).sum())

suic_sum = suic_sum.reset_index().sort_index(by='suicides_no',ascending=False)

most_cont = suic_sum.head(8)

fig = plt.figure(figsize=(20,10))

plt.title('Count of suicides for 31 years.')

sns.set(font_scale=2)

sns.barplot(y='suicides_no',x='country',data=most_cont,palette="Blues_d")

plt.xticks(rotation=45)

plt.ylabel('');

plt.xlabel('')

plt.tight_layout()
from mpl_toolkits.basemap import Basemap

concap = pd.read_csv('../input/world-capitals-gps/concap.csv')

concap.head()
def reg(x):

    if x=='Russia':

        res = 'Russian Federation'

    else:

        res=x

    return res

concap['CountryName'] = concap['CountryName'].apply(reg)



data_full = pd.merge(concap[['CountryName', 'CapitalName', 'CapitalLatitude', 'CapitalLongitude']],\

         suic_sum,left_on='CountryName',right_on='country')
def mapWorld(col1,size2,title3,label4,metr=100,colmap='hot'):

    m = Basemap(projection='mill',llcrnrlat=-60,urcrnrlat=70,\

            llcrnrlon=-110,urcrnrlon=180,resolution='c')

    m.drawcoastlines()

    m.drawcountries()

    m.drawparallels(np.arange(-90,91.,30.))

    m.drawmeridians(np.arange(-90,90.,60.))

    lat = data_full['CapitalLatitude'].values

    lon = data_full['CapitalLongitude'].values

    a_1 = data_full[col1].values

    if size2:

        a_2 = data_full[size2].values

    else: a_2 = 1

    m.scatter(lon, lat, latlon=True,c=a_1,s=metr*a_2,linewidth=1,edgecolors='black',cmap=colmap, alpha=1)

    

    cbar = m.colorbar()

    cbar.set_label(label4,fontsize=30)

    plt.title(title3, fontsize=30)

    plt.show()

sns.set(font_scale=1.5)

plt.figure(figsize=(15,15))

mapWorld(col1='suicides_no', size2=False,title3='Suicide count',label4='',metr=300,colmap='viridis')
data_full = pd.merge(concap[['CountryName', 'CapitalName', 'CapitalLatitude', 'CapitalLongitude']],\

         data,left_on='CountryName',right_on='country')

plt.figure(figsize=(15,15))

mapWorld(col1='gdp_per_capita ($)', size2=False,title3='GDP ($)',label4='',metr=200,colmap='viridis')
suic_sum_m = data['suicides_no'].groupby([data['country'],data['sex']]).sum()

suic_sum_m = suic_sum_m.reset_index().sort_index(by='suicides_no',ascending=False)

most_cont_m = suic_sum_m.head(10)

most_cont_m.head(10)

fig = plt.figure(figsize=(20,5))

plt.title('Count of suicides for 31 years.')

sns.set(font_scale=1.5)

sns.barplot(y='suicides_no',x='country',hue='sex',data=most_cont_m,palette='Set2');

plt.ylabel('Count of suicides')

plt.tight_layout()
suic_mean = pd.DataFrame(data['suicides/100k pop'].groupby(data['country']).mean())

suic_mean = suic_mean.reset_index()

suic_mean_most = suic_mean.sort_index(by='suicides/100k pop',ascending=False).head(8)



fig = plt.figure(figsize=(15,5))

plt.title('suicides/100k pop.')

#sns.set(font_scale=1.5)

sns.barplot(y='suicides/100k pop',x='country',data=suic_mean_most,palette="GnBu_d");

plt.ylabel('suicides/100k pop')

plt.tight_layout()
data_past = data[data['year']<2000]

suic_mean = pd.DataFrame(data_past['suicides/100k pop'].groupby(data_past['country']).mean())

suic_mean = suic_mean.reset_index()

data_full = pd.merge(concap[['CountryName', 'CapitalName', 'CapitalLatitude', 'CapitalLongitude']],\

         suic_mean,left_on='CountryName',right_on='country')

plt.figure(figsize=(15,15))

mapWorld(col1='suicides/100k pop', size2=False,title3='Suicides/100k pop before 2000 year',label4='',metr=300,colmap='viridis')
data_last = data[data['year'] > 2000]

suic_mean = pd.DataFrame(data_last['suicides/100k pop'].groupby(data_last['country']).mean())

suic_mean = suic_mean.reset_index()

data_full = pd.merge(concap[['CountryName', 'CapitalName', 'CapitalLatitude', 'CapitalLongitude']],\

         suic_mean,left_on='CountryName',right_on='country')

plt.figure(figsize=(15,15))

mapWorld(col1='suicides/100k pop', size2=False,title3='Suicides/100k pop after 2000 year',label4='',metr=300,colmap='viridis')
suic_sum_yr = pd.DataFrame(data['suicides_no'].groupby(data['year']).sum())

suic_sum_yr = suic_sum_yr.reset_index().sort_index(by='suicides_no',ascending=False)

most_cont_yr = suic_sum_yr

fig = plt.figure(figsize=(30,10))

plt.title('Count of suicides for years.')

sns.set(font_scale=2.5)

sns.barplot(y='suicides_no',x='year',data=most_cont_yr,palette="OrRd");

plt.ylabel('Count of suicides')

plt.xlabel('')

plt.xticks(rotation=45)

plt.tight_layout()
suic_sum_yr = pd.DataFrame(data['suicides_no'].groupby([data['generation'],data['year']]).sum())

suic_sum_yr = suic_sum_yr.reset_index().sort_index(by='suicides_no',ascending=False)

most_cont_yr = suic_sum_yr

fig = plt.figure(figsize=(30,10))

plt.title('The distribution of suicides by age groups')



sns.set(font_scale=2)

sns.barplot(y='suicides_no',x='year',hue='generation',data=most_cont_yr,palette='deep');

plt.ylabel('Count of suicides')

plt.xticks(rotation=45)

plt.tight_layout()
suic_sum_yr = pd.DataFrame(data['suicides_no'].groupby([data['generation'],data['country']]).sum())

suic_sum_yr = suic_sum_yr.reset_index().sort_index(by='suicides_no',ascending=False)

most_cont = suic_sum_yr



data_full = pd.merge(concap[['CountryName', 'CapitalName', 'CapitalLatitude', 'CapitalLongitude']],\

         most_cont,left_on='CountryName',right_on='country')
data_new = data[data['year']<2000]

title_map = 'Generation suicides before 2000 year'

data_gener = pd.DataFrame(data_new['suicides_no'].groupby([data_new['generation'],data_new['country']]).sum()).reset_index()

age_max = pd.DataFrame(data_gener['suicides_no'].groupby(data_gener['country']).max()).reset_index()

gen_full = pd.merge(age_max,data_gener,left_on=['suicides_no','country'],right_on=['suicides_no','country'])



data_full = pd.merge(concap[['CountryName', 'CapitalName', 'CapitalLatitude', 'CapitalLongitude']],\

         gen_full,left_on='CountryName',right_on='country')

data_full.dropna(inplace=True)



def gener(x):

    dic_t = {'Generation X':100,'Silent':200,'G.I. Generation':300,'Boomers':400,'Millenials':500,'Generation Z':600}

    return dic_t[x]

data_full.generation = data_full.generation.apply(gener)

print(" Generation X:100 \n Silent:200 \n G.I. Generation:300 \n Boomers:400 \n Millenials:500 \n Generation Z:600")

plt.figure(figsize=(15,15))

mapWorld(col1='generation', size2='suicides_no', title3=title_map,label4='',metr=0.01,colmap='viridis')
data_new = data[data['year']>=2000]

title_map = 'Generation suicides after 2000 year'

data_gener = pd.DataFrame(data_new['suicides_no'].groupby([data_new['generation'],data_new['country']]).sum()).reset_index()

age_max = pd.DataFrame(data_gener['suicides_no'].groupby(data_gener['country']).max()).reset_index()

gen_full = pd.merge(age_max,data_gener,left_on=['suicides_no','country'],right_on=['suicides_no','country'])



data_full = pd.merge(concap[['CountryName', 'CapitalName', 'CapitalLatitude', 'CapitalLongitude']],\

         gen_full,left_on='CountryName',right_on='country')

data_full.dropna(inplace=True)



def gener(x):

    dic_t = {'Generation X':100,'Silent':200,'G.I. Generation':300,'Boomers':400,'Millenials':500,'Generation Z':600}

    return dic_t[x]

data_full.generation = data_full.generation.apply(gener)

print(" Generation X:100 \n Silent:200 \n G.I. Generation:300 \n Boomers:400 \n Millenials:500 \n Generation Z:600")

plt.figure(figsize=(15,15))

mapWorld(col1='generation', size2='suicides_no', title3=title_map,label4='',metr=0.01,colmap='viridis')