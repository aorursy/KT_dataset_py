# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
choco=pd.read_csv('/kaggle/input/chocolate-bar-ratings/flavors_of_cacao.csv')
choco.tail()
choco['Specific Bean Origin\nor Bar Name'].unique()
choco['Bean\nType'].unique()
choco['Broad Bean\nOrigin'].unique()
choco.columns
new_columns=['company','bean_region','review_#','review_date','cocoa_%','company_loc',
             'rating','bean_type','bean_country']
old_columns=choco.columns
choco=choco.rename(columns=dict(zip(old_columns,new_columns)))
#renaming columns into more "logical"
choco=choco[['bean_region','bean_country','bean_type','cocoa_%',
       'review_#','review_date','company','company_loc','rating']]
choco.info()
choco.head(50)
choco.groupby('bean_region')[['rating']].mean().sort_values(by='rating',ascending=False).head(30)
choco[choco['bean_type'].isnull()]
choco[choco['bean_type']=='Trinitario']
choco['bean_type'].describe()
choco.bean_type.values[1]
choco['bean_type'].unique()
sns.set_style('whitegrid')
plt.figure(figsize=(17,6))
sns.countplot(x=choco['bean_type'])
plt.xticks(rotation='vertical')
plt.tight_layout
choco[choco['bean_type']=='\xa0']
choco=choco.replace('\xa0','none')
type(choco.bean_type.values[1])
choco.head(10)
len(choco[choco['bean_type']=='none'])
choco['bean_type'].isnull().value_counts()
choco['bean_type'].fillna(value='none',inplace=True)
choco['bean_type'].isnull().value_counts()
choco[choco['bean_type']=='none'].count()
def find_word(text):
    if 'Forastero'in text:
        return 1
    else:
        return 0
choco['bean_type'].apply(find_word).sum()
choco['Forastero']=choco['bean_type'].apply(find_word)
choco['Forastero']
sns.countplot(x='Forastero',data=choco)
def find_word(text):
    if 'Trinitario'in text:
        return True
    else:
        return False
choco['bean_type'].apply(find_word).value_counts()
choco['Trinitario']=choco['bean_type'].apply(find_word)
sns.countplot(x='Trinitario',data=choco)
def find_word(text):
    if 'Criollo'in text:
        return 1
    else:
        return 0
choco['Criollo']=choco['bean_type'].apply(find_word)
choco['Criollo'].sum()
sns.countplot(x='Criollo',data=choco)
def is_blend(text):
    if (',' in text) or('Blend' in text) or ('mix'in text) :
        return 1
    else:
        return 0

is_blend('Amazon Blend')
choco['blend']=choco['bean_type'].apply(is_blend)
choco['blend'].value_counts()
choco1=pd.concat((choco['Forastero'],choco['Trinitario'],choco['Criollo'],choco['blend']),axis=1)
choco1
fig, ax =plt.subplots(1,4)
sns.countplot(choco1['Forastero'], ax=ax[0])
sns.countplot(choco1['Trinitario'], ax=ax[1])
sns.countplot(choco1['Criollo'], ax=ax[2])
sns.countplot(choco1['blend'], ax=ax[3])
plt.tight_layout()
choco.groupby('blend')['rating'].mean()
choco.groupby('Forastero')['rating'].mean()
choco.groupby('Criollo')['rating'].mean()
choco.groupby('Trinitario')['rating'].mean()
def not_blend(text):
    if ('Forastero'in text) and (',' not in text) and ('Blend' not in text) and  ('mix'not in text) :
        return 1
    else:
        return 
choco['bean_type'].apply(not_blend).sum()
def not_blend(name, text):
    if (name in text) and (',' not in text) and ('Blend' not in text) and  ('mix'not in text) :
        return 1
    else:
        return 0
choco['Forastero']=choco['bean_type'].apply(lambda text: not_blend("Forastero",text))
choco['Criollo']=choco['bean_type'].apply(lambda text: not_blend("Criollo",text))
choco['Trinitario']=choco['bean_type'].apply(lambda text: not_blend("Trinitario",text))
choco['Forastero'].sum()
choco['Criollo'].sum()
choco['Trinitario'].sum()
choco.groupby("Forastero")['rating'].mean()
choco.groupby("Trinitario")['rating'].mean()
choco.groupby("Criollo")['rating'].mean()
ratings=pd.DataFrame(data=[choco.groupby("Forastero")['rating'].mean(),
                          choco.groupby("Trinitario")['rating'].mean(),
                          choco.groupby("Criollo")['rating'].mean(),
                          choco.groupby("blend")['rating'].mean()], index=["Forastero",'Trinitario',
                                                                             'Criollo','blend'], 
                                                                     columns=None)
ratings
choco.head()
choco['bean_country'].nunique()
choco['bean_country'].unique()
choco[choco['bean_country'].isnull()]
choco['bean_country'].fillna(value='Madagascar',inplace=True)
choco[choco['bean_country'].isnull()]
choco[choco['bean_country']=='none'][['bean_region','bean_type']].head(40)
len(choco[choco['bean_country']=='none'])
choco2=choco.drop(choco[choco['bean_country']=='none'].index)
len(choco2[choco2['bean_country']=='none'])
choco2.apply(len)
country_count=pd.DataFrame(choco2['bean_country'].value_counts().head(24),
                           index=None,columns=None)
country_count
country_count.columns
country_count.index
country_count.sum()/17.22
country_count.head(5).sum()/17.22
country_count.plot(kind='bar',color='green')
l=[]
for country in country_count.index:
    l.append(round(choco[choco['bean_country']==country]['rating'].mean(),2))
    
print(f'{l}')
l
import pycountry

input_countries = country_count.index

countries = {}
for country in pycountry.countries:
    countries[country.name] = country.alpha_3

codes = [countries.get(country, 'Unknown code') for country in input_countries]

print(codes)
#replacing missing values
Country_code=['VEN', 'ECU', 'PER', 'MDG', 'DOM', 'NIC', 'BRA', 'BOL', 'BLZ', 'PNG', 'COL', 'CRI',
              'VNM', 'TZA', 'TTO', 'GHA', 'MEX', 'US-HI', 'GTM', 'DOM', 'JAM', 'GRD', 'IDN', 'HND']
top_24_rating=pd.DataFrame(list(zip(Country_code,l)),country_count.index,columns=['Country Code','Rating'])
top_24_rating
top_24_countries=pd.concat([country_count,top_24_rating],axis=1)
top_24_countries=top_24_countries[['Country Code','bean_country','Rating']]
top_24_countries.rename(columns={'bean_country':'# of data'},inplace=True)
top_24_countries

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
data=dict(type='choropleth',
         locations=Country_code, 
         z=top_24_countries['Rating'],
         text=top_24_countries['Country Code'],
         colorbar={'title':'Chocolate rating'})
layout=dict(title='Chocolate rating for top 24 beans origin countries',
           geo=dict(showframe=False,projection={'type':'mercator'}))

choromap4=go.Figure(data=[data],layout=layout)

iplot(choromap4)
#countplot per countries with associated mean score
def display_figures(ax):
    i=0
    for p in ax.patches:
        h=p.get_height()
        if (h>0):
            value=l[i]
            ax.text(p.get_x()+p.get_width()/2,h+6, value,ha='center')
            i=i+1
            
            
plt.figure(figsize=(15,10))
ax=sns.barplot(x='Country Code',y='# of data',data=top_24_countries,orient='v')
display_figures(ax)
#barplot for mean score vs.countries
def display_figures(ax):
    i=0
    for p in ax.patches:
        h=p.get_height()
        if (h>0):
            value=top_24_countries['# of data'][i]
            ax.text(p.get_x()+p.get_width()/2,h+0.08, value, ha='center')
            i=i+1
            
            
plt.figure(figsize=(15,10))
ax=sns.barplot(x='Country Code',y='Rating',data=top_24_countries,orient='v')
display_figures(ax)
top_24_countries['Rating'].max()
top_24_countries['Rating'].min()

choco.groupby('bean_country')[['rating']].mean().sort_values(by='rating',ascending=False).head(20)
#making a new dataframe for countries with the highest reviews
df1=pd.DataFrame(round(choco.groupby('bean_country')[['rating']].mean().sort_values(by='rating',ascending=False).head(20),2),
                           index=None,columns=None)
df1
l1=[]
for country in df1.index:
    l1.append(len(choco[choco['bean_country']==country]))
    
len(l1)
#adding the number of reviews to top 20 ratings
top_20_ratings=pd.DataFrame(list(zip(l1,df1['rating'])),index=df1.index,columns=['# of reviews','Avearge rating'])
top_20_ratings
top_20_ratings['# of reviews'].sum()/17.22
df2=pd.DataFrame(round(choco.groupby('bean_country')[['rating']].mean().sort_values(by='rating',ascending=True).head(20),2),
                           index=None,columns=None)
df2
l2=[]
for country in df2.index:
    l2.append(len(choco[choco['bean_country']==country]))
bottom_20_ratings=pd.DataFrame(list(zip(l2,df2['rating'])),index=df2.index,columns=['# of reviews','Avearge rating'])
bottom_20_ratings.tail()
choco[choco['bean_country']=='Carribean']
choco[choco['bean_country']=='Uganda']
choco[choco['bean_country']=='Ivory Coast']
choco[choco['bean_country']=='West Africa']
(bottom_20_ratings['# of reviews'].sum()-73)/17.22
sns.distplot(choco['rating'],kde=False,bins=8)
#checking on company location
choco['company_loc'].nunique()
choco['company_loc'].isnull().value_counts()
choco['company_loc'].unique()
choco.groupby('company_loc')[['rating']].mean()
choco=choco.replace('Amsterdam','Netherlands')
choco.groupby('company_loc')[['rating']].mean()
choco['company_loc'].value_counts()
df3=pd.DataFrame(choco['company_loc'].value_counts().head(15),index=None,columns=None)
df3
l3=[]
for country in df3.index:
    l3.append(round(choco[choco['company_loc']==country]['rating'].mean(),2))


l3
top_companies=pd.DataFrame(list(zip(df3['company_loc'],l3)),index=df3.index,columns=['# of reviews','Rating'])
top_companies.sort_values(by='Rating',ascending=False)

top_24_countries
s=0
for country in  top_companies.index:
    if country in top_24_countries.index:
        s+=1
    else:
        pass
print (s)

choco['cocoa_%']
def per_to_num(data):
    return int(data[0:2])
choco['cocoa_%']=choco['cocoa_%'].apply(per_to_num)
choco['cocoa_%'].unique()
choco['cocoa_%'].value_counts().head(10).sum()/17.95
# %top 10 cocoa percentages
df4=pd.DataFrame(choco['cocoa_%'].value_counts().head(10),index=None,columns=None)
df4
df5=choco.groupby('cocoa_%')[['rating']].mean()
df5
plt.scatter(x=df5.index,y='rating',data=df5)
#let'sgo backand look at the top 10 cocoa percentages
l4=[]
for per in df4.index:
    l4.append(round(choco[choco['cocoa_%']==per]['rating'].mean(),2))


l4
top_10_cocoa_per=pd.DataFrame(list(zip(df4['cocoa_%'],l4)),index=df4.index,columns=['# of reviews','Rating'])
top_10_cocoa_per
plt.scatter(x=top_10_cocoa_per.index,y='Rating',s=top_10_cocoa_per['# of reviews'], data=top_10_cocoa_per,marker='o',c='pink')
choco[choco['bean_country']=='Carribean']['cocoa_%'].mean()
choco[choco['bean_country']=='Uganda']['cocoa_%'].mean()
choco[choco['bean_country']=='Ivory Coast']['cocoa_%'].mean()
choco[choco['bean_country']=='West Africa']['cocoa_%'].mean()
df5
plt.figure(figsize=(15,6))
sns.boxplot(x='cocoa_%', y='rating',data=choco)
