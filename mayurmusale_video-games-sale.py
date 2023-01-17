import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('darkgrid')
VGSales = pd.read_csv("../input/video-games-sales/vgsales.csv")
YearFind = VGSales[VGSales['Year'].isna()].loc[:,['Name','Year']]
YearFind['Year']=[float(i[-1]) if (i[-1].isnumeric()) & (len(i[-1])==4) else np.nan for i in YearFind['Name'].str.split()]
VGSales.update(YearFind['Year'],overwrite=False)
VGSales.dropna(how='any', inplace=True)
VGSales['Year']=VGSales['Year'].astype('int')
VGSales.info()
VGSales.describe()
Yearly_Sales = VGSales.groupby(by=['Year'])['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales'].sum()
Yearly_Sales.head(10)
Yearly_Sales.describe()
Yearly_Sales['NA_Sales'].sum()/Yearly_Sales['Global_Sales'].sum()
fig, ax =plt.subplots(1,4,figsize=(18,5))

sns.distplot(Yearly_Sales.Global_Sales,ax=ax[0],kde=False,color='orange')

sns.distplot(Yearly_Sales.NA_Sales,ax=ax[0],kde=False,color='lightseagreen')



sns.distplot(Yearly_Sales.Global_Sales,ax=ax[1],kde=False,color='orange')

sns.distplot(Yearly_Sales.EU_Sales,ax=ax[1],kde=False,color='lightseagreen')



sns.distplot(Yearly_Sales.Global_Sales,ax=ax[2],kde=False,color='orange')

sns.distplot(Yearly_Sales.JP_Sales,ax=ax[2],kde=False,color='lightseagreen')



sns.distplot(Yearly_Sales.Global_Sales,ax=ax[3],kde=False,color='orange')

sns.distplot(Yearly_Sales.Other_Sales,ax=ax[3],kde=False,color='lightseagreen')

plt.show()
#Higher Sales Year span

#Higher Sales Region
fig, ax =plt.subplots(1,1,figsize=(15,5))

plt.plot(Yearly_Sales.index,Yearly_Sales.NA_Sales,lw=1.5,ls='--')

plt.plot(Yearly_Sales.index,Yearly_Sales.EU_Sales,lw=1.5,ls='--')

plt.plot(Yearly_Sales.index,Yearly_Sales.JP_Sales,lw=1.5,ls='--')

plt.plot(Yearly_Sales.index,Yearly_Sales.Other_Sales,lw=1.5,ls='--')

plt.bar(Yearly_Sales.index,Yearly_Sales.Global_Sales,color='powderblue')

plt.xlabel('Year',fontsize=12, color="black")

plt.ylabel('Sale_in_MN',fontsize=12, color="black")

plt.title('Yearly Sales Distribution',fontsize=15, color="navy")

plt.show()
fig, ax =plt.subplots(1,1,figsize=(15,10))

Yearly_Sales.plot(ax=ax, kind='barh',stacked=True)

plt.show()
#Gener Popularity
Popular_Gener=VGSales.groupby(by=['Genre'])['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales'].sum()

Popular_Gener = Popular_Gener.sort_values(by=['Global_Sales'],ascending=False)
plt.figure(figsize=(15,5))

sns.barplot(Popular_Gener.index,Popular_Gener.Global_Sales, palette='RdBu')

plt.xticks(rotation= 40)

plt.xlabel('Genre',fontsize=12, color="black")

plt.ylabel('Global_Sales_in(MN)',fontsize=12, color="black")

plt.title('Popular_Genres',fontsize=15, color="navy")

plt.show()
Platform_wise_sale = VGSales.groupby(by=['Platform'])['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales'].sum()
Platform_wise_sale = Platform_wise_sale.sort_values(by=['Global_Sales'],ascending=False)
plt.figure(figsize=(15,5))

sns.barplot(x=Platform_wise_sale.index,y='Global_Sales',data=Platform_wise_sale,palette=None,)

plt.xticks(rotation=40)

plt.xlabel('Platform',fontsize=12, color="black")

plt.ylabel('Global_Sales_in(MN)',fontsize=12, color="black")

plt.title('Popular_Platforms',fontsize=15, color="navy")

plt.show()
Popular_Publisher = VGSales.groupby(by=['Publisher'])['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales'].sum()

Popular_Publisher = Popular_Publisher.sort_values(by=['Global_Sales'],ascending=False)
Popular_Publisher = Popular_Publisher[Popular_Publisher['Global_Sales']>100]
plt.figure(figsize=(15,5))



sns.barplot(x='Global_Sales',y=Popular_Publisher.index,data=Popular_Publisher,palette=None,)

plt.xticks(rotation=40)

plt.xlabel('Global_Sales_in(MN)',fontsize=12, color="black")

plt.ylabel('Publisher',fontsize=12, color="black")

plt.title('Publisher > 100(MN)',fontsize=15, color="navy")

plt.show()
VGSales
#Find out which games are popular in golden years of gaming (1995 to 2015)
Year95_to_2015 = VGSales[(VGSales['Year']>1995) & (VGSales['Year']<=2015)]
Popular_Games = pd.pivot_table(Year95_to_2015,values='Global_Sales',index=['Name'],aggfunc=np.sum)
Popular_Games.sort_values(by=['Global_Sales'],ascending=False, inplace=True)
Popular_Games = Popular_Games[Popular_Games.Global_Sales>=20]
Publisher_golden_year = pd.pivot_table(Year95_to_2015,values='Global_Sales',index=['Publisher'],aggfunc=np.sum)

Publisher_golden_year.sort_values(by=['Global_Sales'],ascending=False, inplace=True)

Publisher_golden_year = Publisher_golden_year[Publisher_golden_year.Global_Sales>=20]
Publisher_golden_year
fig, ax =plt.subplots(1,2,figsize=(10,15))

fig.tight_layout(pad=10.0)

sns.barplot(Popular_Games.Global_Sales,Popular_Games.index,ax=ax[0])

sns.barplot(Publisher_golden_year.Global_Sales,Publisher_golden_year.index,ax=ax[1])



plt.show()