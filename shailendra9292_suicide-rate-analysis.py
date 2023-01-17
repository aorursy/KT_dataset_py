import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/master.csv')
df.head()
df.columns = ['country','year','sex','age','suicide_count','population','suicides_per_100k','country_year','HDI_for_year',

             'gdp_for_year','gdp_per_capita','generation']
df.info()
df['country'] = df['country'].astype('category')

df['sex'] = df['sex'].astype('category')

df['age'] = df['age'].astype('category')

df['gdp_for_year'] = df['gdp_for_year'].astype('str')

df['generation'] = df['generation'].astype('category')
df['gdp_for_year']=df.gdp_for_year.str.replace(',','').astype('int')
df.isnull().sum()
df.dropna(inplace=True)
df.drop(labels=['HDI_for_year','country_year'],inplace=True,axis=1)
df.describe()
plt.figure(dpi=100)

sns.heatmap(df.corr(),annot=True,cmap='YlGnBu')

plt.show()
plt.figure(figsize=(18,4),dpi=150)

ax=sns.barplot(df.age,df.suicide_count)

for p in ax.patches:

    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()), ha='left', va='center',

                xytext=(5, 10), textcoords='offset points')

plt.show()
plt.figure(figsize=(21,5),dpi=150)

df.groupby([df.country,df.age]).suicide_count.sum().nlargest(20).plot(kind='bar')

plt.ylabel('Total number of suicides ')

plt.show()
plt.figure(figsize=(8,4),dpi=90)

ax=sns.countplot(df.generation)

for p in ax.patches:

    ax.annotate(p.get_height(), (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='top',

                xytext=(0, 10), textcoords='offset points')

plt.show()
sns.pairplot(data=df,hue='generation')

plt.show()
sns.pairplot(data=df.loc[:,['suicide_count','population','gdp_for_year','gdp_per_capita']],kind='reg',

             plot_kws={'color':'c','line_kws':{'color':'tomato','linewidth':3}})

plt.show()
plt.figure(figsize=(21,6))

ax=sns.barplot(y='suicide_count',x='year',hue='generation',data=df,palette='Spectral')

plt.show()
sns.lineplot(x=df.year,y=df.suicide_count,hue=df.sex,data=df,markers=True,style='sex')

plt.title('Number of Suicides per year with respect to Sex ')

plt.show()
plt.subplots_adjust(right=1, wspace=1, hspace=None)

plt.figure(figsize=(18,5),dpi=110)

plt.subplot(1,2,1)

plt.title('Number of Suicides per year with respect to Generation ')

sns.lineplot(x=df.year,y=df.suicide_count,hue=df.age,markers=True,style=df.age)

plt.legend(loc=1)

plt.subplot(1,2,2)

plt.title('Number of Suicides per year with respect to Sex ')

ax=sns.barplot(y='suicide_count',x='year',data=df,palette="GnBu",hue=df.sex)

for p in ax.patches:

    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center',

                xytext=(0, 10), textcoords='offset points')

plt.subplots_adjust(wspace=0.27)

plt.show()
df.groupby(df.country).suicide_count.sum().nlargest(10).plot(kind='barh')
max_suicide_countries=pd.DataFrame(df.groupby(df.country).suicide_count.sum().nlargest(10))
max_suicide_countries.head()
min_suicide_countries = pd.DataFrame(df.groupby(df.country).suicide_count.sum()).nsmallest(10,columns='suicide_count')
min_suicide_countries.head()
# How to display row styles dataframe

#https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side



from IPython.display import display_html

def display_side_by_side(*args):

    html_str=''

    for df in args:

        html_str+=df.to_html()

    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
display_side_by_side(min_suicide_countries,max_suicide_countries)