import pandas as pd

import matplotlib as mat

import matplotlib.pyplot as plt

import seaborn as sns

import re

import numpy as np
mat.rcParams.update({'figure.figsize':(20,15),'font.size':14})
sales = pd.read_csv('../input/videogamesales/vgsales.csv')

extra_info = pd.read_csv('../input/ign-dataset/ign.csv')
sales.drop('Rank',1,inplace=True)
sales.info()
sales.head()
sales.loc[sales['NA_Sales'].isna()]
sales.loc[sales['EU_Sales'].isna()]
sales.loc[sales['NA_Sales'].isna(),'NA_Sales'] = sales.loc[sales['NA_Sales'].isna()][['EU_Sales','JP_Sales','Other_Sales','Global_Sales']].apply(lambda x: x[3] - sum(x[:3]),axis=1) 

sales.loc[sales['EU_Sales'].isna(),'EU_Sales'] = sales.loc[sales['EU_Sales'].isna()][['NA_Sales','JP_Sales','Other_Sales','Global_Sales']].apply(lambda x: x[3] - sum(x[:3]),axis=1) 
sales['Name']= sales['Name'].str.lower().str.strip()

extra_info['title']= extra_info['title'].str.lower().str.strip()
extra_info["Platform-md"] = extra_info.platform.apply(lambda x: 

                              ''.join(

                                     [word[0] if word.isalpha()

                                       else word

                                        for word in re.sub(r"(\w)([A-Z])", r"\1 \2", x).split()] 

                                  # regular expression sub method takes 3 arguments the pattern, 

                                  # replacment and the original string

                                  # pattern gives says match any word char and the second match any capital char 

                                  # in the case of PlayStation it will match y and S

                                  # replacement uses backreference \1 says choose the first group (y) and \2 the 2nd (S)

                                  # says put space between the matched chars

                                    )

                            )
extra_info.loc[extra_info.platform.isin(sales["Platform"].unique()),'Platform-md'] = extra_info.loc[extra_info.platform.isin(sales["Platform"].unique())].platform
sales.loc[~sales.Platform.isin(extra_info["Platform-md"].unique())].Platform.unique()
fixing_platform={

  'NDS': 'DS',

    'N3D':'3DS',

    'WU':'WiiU',

    'X':'XB',

   'XO':'XOne',

    'A2600':'2600',

    'SNE':'SNES',

    'G':'GEN',

    'D':'DC',

    'S':'SAT',

    'TGrafx-16':'TG16'

}
for k in fixing_platform.keys():

     extra_info.loc[extra_info['Platform-md'] == k,'Platform-md']=fixing_platform[k]
sales_year_recovered = pd.merge(sales,

                 extra_info.rename(columns={'title':'Name','Platform-md':'Platform'}), 

                 on = ['Name','Platform'],

                 how="left")
sales_year_recovered.loc[sales_year_recovered["Year"].isna(),'Year'] = sales_year_recovered.loc[sales_year_recovered["Year"].isna(),'release_year']
sales.columns
sales_year_recovered = sales_year_recovered.dropna(subset=['Year', 'Publisher'],axis=0)
sales_year_recovered = sales_year_recovered[sales_year_recovered.columns[:15]].drop_duplicates(keep='first')
sales_year_recovered.head()
sales_year_recovered.info()
sales_dummies = pd.concat([sales.drop(['Genre','Publisher'],1),pd.get_dummies(sales[['Genre','Publisher']],dummy_na = False,drop_first=True)],sort=False)
for col in sales.columns[5:9]:

    sales[col + '%'] = (sales[col].astype(float)/sales[sales.columns[9]].astype(float)) * 100 
sales.head()
sales['Rank'] = sales.index + 1
sales['Global_Sales%'] = round((sales['Global_Sales']/sales['Global_Sales'].sum())*100,4)
sales['Global_Sales - CumSum'] = sales['Global_Sales%'].cumsum()
sales[['NA_Sales%','EU_Sales%','JP_Sales%','Other_Sales%','Global_Sales']].describe()
sales.groupby('Genre')[['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales']].sum().sort_values('Global_Sales',ascending=False)
sales.sort_values('Global_Sales').tail(1)
sales.sort_values('NA_Sales').tail(1)
sales.sort_values('EU_Sales').tail(1)
sales.sort_values('JP_Sales').tail(1)
sales.sort_values('Other_Sales').tail(1)
sales_year = sales_year_recovered.groupby('Year')['Global_Sales'].sum().reset_index()
# top 5 years 

sales_year.sort_values(['Global_Sales']).tail()
sales_year['Year'] = pd.to_datetime(sales_year['Year'],format='%Y')
plt.figure(figsize=(15,10))

sales_year.set_index('Year').plot(grid=True)

plt.show()
games_year = sales_year_recovered.groupby('Year')['Name'].count().reset_index().rename(columns={'Name':'Total Games'})
# top 5 years 

games_year.sort_values(['Total Games'],ascending=False).head()
games_year['Year'] = pd.to_datetime(games_year['Year'],format='%Y')
plt.figure(figsize=(15,10))

games_year.set_index('Year').plot(grid=True)

plt.show()
sales_games_year = games_year.set_index('Year').join(sales_year.set_index('Year'))
sales_games_year.plot.scatter(x='Total Games', y='Global_Sales',s=40)

plt.grid()
sales_games_year.corr()
sales_games_year.plot(grid=True)
sales.loc[sales['Year'] == 2008].groupby('Genre')['Name'].count().sort_values()
sales_genre_region = sales.groupby('Genre')[['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales']].mean().reset_index()
sales_genre_region.sort_values('NA_Sales',ascending=False).head(1)
sales.replace({0:np.nan}).groupby('Genre')[['NA_Sales']].count().sort_values('NA_Sales').tail(1)
sales.replace({0:np.nan}).groupby('Genre')[['NA_Sales']].sum().sort_values('NA_Sales').tail(1)
sales.replace({0:np.nan}).groupby('Genre')[['NA_Sales']].count().sort_values('NA_Sales').loc['Platform']
sales.replace({0:np.nan}).groupby('Genre')[['NA_Sales']].sum().sort_values('NA_Sales').loc['Platform']
sales_genre_region.sort_values('EU_Sales',ascending=False).head(1)
sales.replace({0:np.nan}).groupby('Genre')[['EU_Sales']].count().sort_values('EU_Sales').tail(1)
sales.replace({0:np.nan}).groupby('Genre')[['EU_Sales']].sum().sort_values('EU_Sales').tail(1)
sales.replace({0:np.nan}).groupby('Genre')[['EU_Sales']].count().sort_values('EU_Sales').loc['Shooter']
sales.replace({0:np.nan}).groupby('Genre')[['EU_Sales']].sum().sort_values('EU_Sales').loc['Shooter']
sales_genre_region.sort_values('JP_Sales',ascending=False).head(1)
sales.replace({0:np.nan}).groupby('Genre')[['JP_Sales']].count().sort_values('JP_Sales').tail(1)
sales.replace({0:np.nan}).groupby('Genre')[['EU_Sales']].sum().sort_values('EU_Sales').loc['Action']
sales.replace({0:np.nan}).groupby('Genre')[['JP_Sales']].count().sort_values('JP_Sales').loc['Role-Playing']
sales.replace({0:np.nan}).groupby('Genre')[['JP_Sales']].sum().sort_values('JP_Sales').tail(1)
sales_genre_region.sort_values('Other_Sales',ascending=False).head(1)
sales_genre_region = sales.groupby('Genre')[['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales']].sum().reset_index()
sales_genre_region = sales_genre_region.set_index('Genre').rename(columns={'NA_Sales':'North America Sales',

                                                     'EU_Sales':'Europe Sales',

                                                     'JP_Sales':'Japan Sales',

                                                     'Other_Sales':'Other Sales'})
sns.heatmap(sales_genre_region,cmap="OrRd",annot=True,fmt=".2f")

plt.xticks(rotation=45)

plt.show()
sales_genre_region.sort_values('Global_Sales',ascending=False).head(1)
sales_platform_region = sales.groupby('Platform')[['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales']].sum().reset_index()
sales_platform_region.sort_values('NA_Sales',ascending=False).head(1)
sales_platform_region.sort_values('EU_Sales',ascending=False).head(1)
sales_platform_region.sort_values('JP_Sales',ascending=False).head(1)
sales_platform_region.sort_values('Other_Sales',ascending=False).head(1)
sales_platform_region.sort_values('Global_Sales',ascending=False).head(1)
sales_platform_region = sales_platform_region.set_index('Platform').rename(columns={'NA_Sales':'North America Sales',

                                                     'EU_Sales':'Europe Sales',

                                                     'JP_Sales':'Japan Sales',

                                                     'Other_Sales':'Other Sales'})
sns.heatmap(sales_platform_region,cmap="OrRd",annot=True,fmt=".2f")

plt.xticks(rotation=45)

plt.show()
sales_melted = pd.melt(sales.rename(columns={'NA_Sales':'North America Sales',

                                                     'EU_Sales':'Europe Sales',

                                                     'JP_Sales':'Japan Sales',

                                                     'Other_Sales':'Other Sales'}), id_vars = ['Name','Genre','Platform'], value_vars = ['North America Sales','Europe Sales','Japan Sales','Other Sales'])
sales_melted = sales_melted.rename(columns={'variable':'Region','value':'Sales'})
sns.boxplot(x='Region',y='Sales', hue = 'Genre', data=sales_melted)

plt.yscale('log')

plt.grid()
sales['NA_Sales'].hist(bins=int(np.sqrt(len(sales))))

plt.yscale('log')
sales['EU_Sales'].hist(bins=int(np.sqrt(len(sales))))

plt.yscale('log')
sales['Global_Sales'].hist(bins=int(np.sqrt(len(sales))))

plt.yscale('log')
sales_pareto_80 = sales.loc[(sales['Global_Sales - CumSum'] <= 80)].copy()

sales_pareto_20 = sales.loc[(sales['Global_Sales - CumSum'] > 80)].copy()
sales['Global_Sales'].sum()*0.8
sales_pareto_80['Global_Sales'].sum()
round((len(sales_pareto_80)/len(sales))*100)
plt.bar([1,2],[sales_pareto_80['Global_Sales'].sum(),sales_pareto_20['Global_Sales'].sum()])

plt.xticks([1,2],['Caused by 25% of Customers','Caused by 75% of Customers'])

plt.grid()

plt.gca().set_frame_on(False)

plt.gca().text(1-0.1,sales_pareto_80['Global_Sales'].sum()+50,'80% Sales')

plt.gca().text(2-0.1,sales_pareto_20['Global_Sales'].sum()+50,'20% Sales')

    

plt.show()
len(sales)
sales_pareto_80.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False).head()
sales_pareto_80.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False).head()
sales_pareto_80.groupby('Year')['Global_Sales'].sum().sort_values(ascending=False).head()