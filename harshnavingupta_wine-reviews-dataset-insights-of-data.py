import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/winemag-data-130k-v2.csv')
dataset.drop(dataset.columns[0],axis = 1,inplace=True)

dataset.head()
dataset.info()
countries_list = pd.DataFrame(dataset.country.value_counts(dropna = True)).iloc[0:10]

countries_list['Index'] = list(range(0,10))

countries_list['Country_Name'] = countries_list.index.values

countries_list.set_index('Index',inplace = True)

countries_list.columns = ['Number Of Wines Produced','Country']

#Top 10 Wine Producing Countries

countries_list
fig, ax = plt.subplots(figsize=(14,7))

sns.set(style="whitegrid")

ax = sns.barplot(y="Country", x="Number Of Wines Produced", data=countries_list)

plt.title("Top 10 Countries Producing Wine",fontsize = 25)

plt.ylabel("Country",fontsize = 20)

plt.xlabel("Number Of Wines Produced",fontsize = 20)
winery_list = pd.DataFrame(dataset.winery.value_counts(dropna = True)).iloc[0:10]

winery_list['Index'] = list(range(0,10))

winery_list['Winery_Name'] = winery_list.index.values

winery_list.set_index('Index',inplace = True)

winery_list.columns = ['Number Of Wines Produced','Winery']

#Top 10 Wine Producing Wineries

winery_list
fig, ax = plt.subplots(figsize=(14,7))

sns.set(style="whitegrid")

sns.barplot(y="Winery", x="Number Of Wines Produced", data=winery_list)

plt.title("Top 10 Wineries Producing Wine",fontsize = 25)

plt.ylabel("Winery",fontsize = 20)

plt.xlabel("Number Of Wines Produced",fontsize = 20)
taster_list = pd.DataFrame(dataset.taster_name.value_counts(dropna = True)).iloc[0:10]

taster_list['Index'] = list(range(0,10))

taster_list['Taster_Name'] = taster_list.index.values

taster_list.set_index('Index',inplace = True)

taster_list.columns = ['Number Of Wines Tasted','Taster Name']

#Top 10 Wine Tasters

taster_list
fig, ax = plt.subplots(figsize=(14,7))

sns.set(style="whitegrid")

ax = sns.barplot(y="Taster Name", x="Number Of Wines Tasted", data=taster_list)

plt.title("Top 10 Wine Tasters",fontsize = 25)

plt.ylabel("Taster Name",fontsize = 20)

plt.xlabel("Number Of Wines Tasted",fontsize = 20)
#Visualising The Points Given To Wines From Top 10 Wineries

wineries_points = dataset[['points','winery']]

wineries_points = wineries_points[wineries_points.winery.isin(winery_list.Winery)]

fig, ax = plt.subplots(figsize=(25,15))

sns.set_style("whitegrid")

sns.violinplot(x="winery", y="points", data=wineries_points)

plt.ylabel("Points",fontsize = 25)

plt.xlabel("Wineries",fontsize = 25)

plt.title("Points Given To Wines Produced By Top 10 Wineries",fontsize = 30)
top_10_variety = pd.DataFrame(dataset['variety'].value_counts()[0:10])

top_10_variety['Index'] = list(range(0,10))

top_10_variety['Variety_Name'] = top_10_variety.index.values

top_10_variety.set_index('Index',inplace = True)

top_10_variety
#Visualising The Points Given To Wines From Top 10 Varieties

varieties_points = dataset[['points','variety']]

varieties_points = varieties_points[varieties_points.variety.isin(top_10_variety.Variety_Name)]

fig, ax = plt.subplots(figsize=(25,15))

sns.set_style("whitegrid")

sns.violinplot(x="variety", y="points", data=varieties_points)

plt.ylabel("Points",fontsize = 25)

plt.xlabel("Varieties",fontsize = 25)

plt.title("Points Given To Wines Belonging To Top 10 Varieties",fontsize = 30)
res = pd.DataFrame(dataset[dataset['variety'] == 'Pinot Noir'].groupby('points').count()).iloc[:,0]

variety_points_dist = res

res = pd.DataFrame(dataset[dataset['variety'] == 'Chardonnay'].groupby('points').count()).iloc[:,0]

variety_points_dist = pd.concat([variety_points_dist,res],axis = 1)

res = pd.DataFrame(dataset[dataset['variety'] == 'Cabernet Sauvignon'].groupby('points').count()).iloc[:,0]

variety_points_dist = pd.concat([variety_points_dist,res],axis = 1)

res = pd.DataFrame(dataset[dataset['variety'] == 'Red Blend'].groupby('points').count()).iloc[:,0]

variety_points_dist = pd.concat([variety_points_dist,res],axis = 1)

res = pd.DataFrame(dataset[dataset['variety'] == 'Bordeaux-style Red Blend'].groupby('points').count()).iloc[:,0]

variety_points_dist = pd.concat([variety_points_dist,res],axis = 1)

variety_points_dist.columns = ['Pinot Noir','Chardonnay','Cabernet Sauvignon','Red Blend','Bordeaux-style Red Blend']

variety_points_dist
variety_points_dist.plot.bar(stacked = True,figsize = (15,8),grid = False)

plt.title("Visualising Points For Top 5 Wine Varieties",fontsize = 25)

plt.xlabel("Points",fontsize = 20)
sns.jointplot(x='price', y='points', data=dataset[dataset['price']<100], kind='hex', gridsize=20)
def price_group(pr):

    if(pr>0 and pr<30):

        return 1

    elif(pr >= 30 and pr < 80):

        return 2

    elif(pr >= 80 and pr < 150):

        return 3

    elif(pr >= 150 and pr < 500):

        return 4

    else:

        return 5
dataset['Price_Group'] = dataset['price'].apply(price_group)

counts = list(dataset['Price_Group'].value_counts())
labels = ['Price <30','Price Between 30 & 80','Price Between 80 & 150','Price Between 150 & 500','Price > 500']

colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','red']

plt.figure(figsize = (15,6)) 

plt.pie(counts,labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)

plt.title("Distribution Of Prices Of Wines",fontsize = 25)

plt.axis('equal')

plt.show()
top_5_countries_pw = dataset.groupby('country').mean().sort_values(by = ['points'],ascending = False)

top_5_countries_pw = pd.DataFrame(top_5_countries_pw.iloc[0:5,0])

top_5_countries_pw['Index'] = list(range(5))

top_5_countries_pw['Country'] = top_5_countries_pw.index.values

top_5_countries_pw.set_index(['Index'],inplace = True)

top_5_countries_pw
top_5_countries_cnt = dataset.groupby('country').count()

top_5_countries_cnt['Index'] = list(range(len(top_5_countries_cnt)))

top_5_countries_cnt['Country'] = top_5_countries_cnt.index.values

top_5_countries_cnt.set_index(['Index'],inplace = True)

top_5_countries_cnt = top_5_countries_cnt[top_5_countries_cnt.Country.isin(top_5_countries_pw.Country)]

disp = top_5_countries_cnt[['Country','winery']]

disp.columns = ['Country','Number Of Wines Produced']

disp
fig, ax = plt.subplots(figsize=(14,7))

sns.set(style="whitegrid")

ax = sns.barplot(y="Country", x="variety", data=top_5_countries_cnt)

plt.title("Top 5 Wine Producing Countries",fontsize = 25)

plt.ylabel("Country Name",fontsize = 20)

plt.xlabel("Number Of Wines Produced",fontsize = 20)
top_5_countries_prw = dataset.groupby('country').mean().sort_values(by = ['price'],ascending = False)

top_5_countries_prw = pd.DataFrame(top_5_countries_prw.iloc[0:5,1])

top_5_countries_prw['Index'] = list(range(5))

top_5_countries_prw['Country'] = top_5_countries_prw.index.values

top_5_countries_prw.set_index(['Index'],inplace = True)

top_5_countries_prw
top_5_countries_cnt = dataset.groupby('country').count()

top_5_countries_cnt['Index'] = list(range(len(top_5_countries_cnt)))

top_5_countries_cnt['Country'] = top_5_countries_cnt.index.values

top_5_countries_cnt.set_index(['Index'],inplace = True)

top_5_countries_cnt = top_5_countries_cnt[top_5_countries_cnt.Country.isin(top_5_countries_prw.Country)]

disp = top_5_countries_cnt[['Country','winery']]

disp.columns = ['Country','Number Of Wines Produced']

disp
fig, ax = plt.subplots(figsize=(14,7))

sns.set(style="whitegrid")

ax = sns.barplot(y="Country", x="variety", data=top_5_countries_cnt)

plt.title("Top 5 Wine Producing Countries - Expensive Wines",fontsize = 25)

plt.ylabel("Country Name",fontsize = 20)

plt.xlabel("Number Of Wines Produced",fontsize = 20)
top_5_countries_prwl = dataset.groupby('country').mean().sort_values(by = ['price'],ascending = True)

top_5_countries_prwl = pd.DataFrame(top_5_countries_prwl.iloc[0:5,1])

top_5_countries_prwl['Index'] = list(range(5))

top_5_countries_prwl['Country'] = top_5_countries_prwl.index.values

top_5_countries_prwl.set_index(['Index'],inplace = True)

top_5_countries_prwl
top_5_countries_cnt = dataset.groupby('country').count()

top_5_countries_cnt['Index'] = list(range(len(top_5_countries_cnt)))

top_5_countries_cnt['Country'] = top_5_countries_cnt.index.values

top_5_countries_cnt.set_index(['Index'],inplace = True)

top_5_countries_cnt = top_5_countries_cnt[top_5_countries_cnt.Country.isin(top_5_countries_prwl.Country)]

disp = top_5_countries_cnt[['Country','winery']]

disp.columns = ['Country','Number Of Wines Produced']

disp
fig, ax = plt.subplots(figsize=(14,7))

sns.set(style="whitegrid")

ax = sns.barplot(y="Country", x="variety", data=top_5_countries_cnt)

plt.title("Top 5 Wine Producing Countries - Cheap Wines",fontsize = 25)

plt.ylabel("Country Name",fontsize = 20)

plt.xlabel("Number Of Wines Produced",fontsize = 20)