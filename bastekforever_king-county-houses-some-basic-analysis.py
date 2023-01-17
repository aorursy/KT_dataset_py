import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np 

import os

import pandas as pd 

import seaborn as sns

import warnings



from mpl_toolkits.mplot3d import axes3d



from mpl_toolkits.basemap import Basemap



plt.style.use('seaborn-whitegrid')



def ignore_warn(*args, **kwargs):

    pass



warnings.warn = ignore_warn 



mpl.rcParams['xtick.labelsize'] = 15

mpl.rcParams['ytick.labelsize'] = 15



print(os.listdir("../input"))
data = pd.read_csv('../input/kc_house_data.csv')

data.columns
data.info()
#as we have zipcode I also in to create feature "city"

#copy first

data['city'] = data['zipcode']

data.head()
#https://m.usps.com/m/ZipLookupAction?search=zip

data['city'] = data['city'].replace(98178, 'SEATTLE, WA')

data['city'] = data['city'].replace(98125, 'SEATTLE, WA')

data['city'] = data['city'].replace(98028, 'KENMORE, WA')

data['city'] = data['city'].replace(98136, 'SEATTLE, WA')

data['city'] = data['city'].replace(98074, 'SAMMAMISH, WA')

data['city'] = data['city'].replace(98053, 'REDMOND, WA')

data['city'] = data['city'].replace(98003, 'FEDERAL WAY, WA')

data['city'] = data['city'].replace(98198, 'SEATTLE, WA')

data['city'] = data['city'].replace(98146, 'SEATTLE, WA')

data['city'] = data['city'].replace(98038, 'MAPLE VALLEY, WA')

data['city'] = data['city'].replace(98007, 'BELLEVUE, WA')

data['city'] = data['city'].replace(98107, 'SEATTLE, WA')

data['city'] = data['city'].replace(98126, 'SEATTLE, WA')

data['city'] = data['city'].replace(98019, 'DUVALL, WA')

data['city'] = data['city'].replace(98103, 'SEATTLE, WA')

data['city'] = data['city'].replace(98002, 'AUBURN, WA')

data['city'] = data['city'].replace(98133, 'SEATTLE, WA')

data['city'] = data['city'].replace(98040, 'MERCER ISLAND, WA')

data['city'] = data['city'].replace(98092, 'AUBURN, WA')

data['city'] = data['city'].replace(98030, 'KENT, WA')

data['city'] = data['city'].replace(98119, 'SEATTLE, WA')

data['city'] = data['city'].replace(98112, 'SEATTLE, WA')

data['city'] = data['city'].replace(98052, 'REDMOND, WA')

data['city'] = data['city'].replace(98027, 'ISSAQUAH, WA')

data['city'] = data['city'].replace(98117, 'SEATTLE, WA')

data['city'] = data['city'].replace(98058, 'RENTON, WA')

data['city'] = data['city'].replace(98001, 'AUBURN, WA')

data['city'] = data['city'].replace(98056, 'RENTON, WA')

data['city'] = data['city'].replace(98166, 'SEATTLE, WA')

data['city'] = data['city'].replace(98023, 'FEDERAL WAY, WA')

data['city'] = data['city'].replace(98070, 'VASHON, WA')

data['city'] = data['city'].replace(98148, 'SEATTLE, WA')

data['city'] = data['city'].replace(98105, 'SEATTLE, WA')

data['city'] = data['city'].replace(98042, 'KENT, WA')

data['city'] = data['city'].replace(98008, 'BELLEVUE, WA')

data['city'] = data['city'].replace(98059, 'RENTON, WA')

data['city'] = data['city'].replace(98122, 'SEATTLE, WA')

data['city'] = data['city'].replace(98144, 'SEATTLE, WA')

data['city'] = data['city'].replace(98004, 'BELLEVUE, WA')

data['city'] = data['city'].replace(98005, 'BELLEVUE, WA')

data['city'] = data['city'].replace(98034, 'KIRKLAND, WA')

data['city'] = data['city'].replace(98075, 'SAMMAMISH, WA')

data['city'] = data['city'].replace(98116, 'SEATTLE, WA')

data['city'] = data['city'].replace(98010, 'BLACK DIAMOND, WA')

data['city'] = data['city'].replace(98118, 'SEATTLE, WA')

data['city'] = data['city'].replace(98199, 'SEATTLE, WA')

data['city'] = data['city'].replace(98032, 'KENT, WA')

data['city'] = data['city'].replace(98045, 'NORTH BEND, WA')

data['city'] = data['city'].replace(98102, 'SEATTLE, WA')

data['city'] = data['city'].replace(98077, 'WOODINVILLE, WA')

data['city'] = data['city'].replace(98108, 'SEATTLE, WA')

data['city'] = data['city'].replace(98168, 'SEATTLE, WA')

data['city'] = data['city'].replace(98177, 'SEATTLE, WA')

data['city'] = data['city'].replace(98065, 'SNOQUALMIE, WA')

data['city'] = data['city'].replace(98029, 'ISSAQUAH, WA')

data['city'] = data['city'].replace(98006, 'BELLEVUE, WA')

data['city'] = data['city'].replace(98109, 'SEATTLE, WA')

data['city'] = data['city'].replace(98022, 'ENUMCLAW, WA')

data['city'] = data['city'].replace(98033, 'KIRKLAND, WA')

data['city'] = data['city'].replace(98155, 'SEATTLE, WA')

data['city'] = data['city'].replace(98024, 'FALL CITY, WA')

data['city'] = data['city'].replace(98011, 'BOTHELL, WA')

data['city'] = data['city'].replace(98031, 'KENT, WA')

data['city'] = data['city'].replace(98106, 'SEATTLE, WA')

data['city'] = data['city'].replace(98072, 'WOODINVILLE, WA')

data['city'] = data['city'].replace(98188, 'SEATTLE, WA')

data['city'] = data['city'].replace(98014, 'CARNATION, WA')

data['city'] = data['city'].replace(98055, 'RENTON, WA')

data['city'] = data['city'].replace(98039, 'MEDINA, WA')

data['city'] = data['city'].replace(98115, 'SEATTLE, WA')

data.city.describe()
#convert "date" do TimeFrame

data['date'] = pd.to_datetime(data['date'])

#and add new feature 'day_of_week'

data['day_of_week'] = data['date'].dt.day_name()
#I also wanna know if the apt was renovated in the past time

data['renovated'] = 0

for x in range(len(data)):

    if data.loc[x, 'yr_renovated'] == 0:

        data.loc[x, 'renovated'] = False

    else:

        data.loc[x, 'renovated'] = True



data.head()
#and logaritmic value for price and sqft_living

data['log_price'] = np.log(data['price'])

data['log_sqft_living'] = np.log(data['sqft_living'])
features = ['price','bedrooms','bathrooms','sqft_living','sqft_lot','floors',

            'waterfront','view','condition','grade','sqft_above','sqft_basement',

            'yr_built','yr_renovated','zipcode','sqft_living15','sqft_lot15']



mask = np.zeros_like(data[features].corr(), dtype=np.bool) 

mask[np.triu_indices_from(mask)] = True 



f, ax = plt.subplots(figsize=(16, 12))

plt.title('Macierz korelacji',fontsize=15)



sns.heatmap(data[features].corr(),linewidths=0.25,vmax=1.0,square=True,cmap="YlGnBu", 

            linecolor='w',annot=True,mask=mask,cbar_kws={"shrink": .75})
data['date'].describe()
data['date'].min()
data['date'].max()
fig, ax = plt.subplots(figsize=(25, 10))

pd.to_datetime(data['date']).value_counts().sort_index().plot.line()

ax.set_title('Number of houses sold between May of 2014 r. and May of 2015 r.', fontsize=20)

ax.set_xlabel("Date",fontsize=15)

ax.set_ylabel("Value",fontsize=15)



ax.xaxis.set_major_locator(mpl.dates.MonthLocator())

ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=15))

ax.xaxis.set_major_formatter(plt.NullFormatter())

ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%h'))
data['date'].value_counts().mean()
data[data['day_of_week'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]['day_of_week'].value_counts().sum()
data[data['day_of_week'].isin(['Sutarday', 'Sunday'])]['day_of_week'].value_counts().sum()
week_data = data[data['day_of_week'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]

fig, ax = plt.subplots(figsize=(25, 10))

pd.to_datetime(week_data['date']).value_counts().sort_index().plot.line()

ax.set_title('Number of houses sold (except weekends)', fontsize=20)

ax.set_xlabel("Date",fontsize=15)

ax.set_ylabel("Value",fontsize=15)



ax.annotate('Independence Day', xy=('2014-07-04', 2), xycoords='data', fontsize=15,

            bbox=dict(boxstyle='round', fc='none', ec='#c62540'),

            xytext=(80, 40), textcoords='offset points', ha='center', arrowprops=dict(arrowstyle="simple",

                            fc="0.6", ec="#c62540"))

ax.annotate('Memorial Day', xy=('2014-05-26', 7), xycoords='data', fontsize=15,

            bbox=dict(boxstyle='round', fc='none', ec='#b132c1'),

            xytext=(-80, 40), textcoords='offset points', ha='center', arrowprops=dict(arrowstyle="simple",

                            fc="0.6", ec="#b132c1"))

ax.annotate('Labor Day', xy=('2014-09-01', 5), xycoords='data', fontsize=15,

            bbox=dict(boxstyle='round', fc='none', ec='#32c19a'),

            xytext=(80, 20), textcoords='offset points', ha='center', arrowprops=dict(arrowstyle="simple",

                            fc="0.6", ec="#32c19a"))

ax.annotate('Thanksgiving Day', xy=('2014-11-27', 2), xycoords='data', fontsize=15,

            bbox=dict(boxstyle='round', fc='none', ec='#a4c132'),

            xytext=(-120, 0), textcoords='offset points', ha='center', arrowprops=dict(arrowstyle="simple",

                            fc="0.6", ec="#a4c132"))

ax.annotate('Martin Luther King’s Jr Day', xy=('2015-01-19', 7), xycoords='data', fontsize=15,

            bbox=dict(boxstyle='round', fc='none', ec='#c18a32'),

            xytext=(0, -30), textcoords='offset points', ha='center', arrowprops=dict(arrowstyle="simple",

                            fc="0.6", ec="#c18a32"))

ax.annotate('Presidents’ Day', xy=('2015-02-16', 7), xycoords='data', fontsize=15,

            bbox=dict(boxstyle='round', fc='none', ec='#3272c1'),

            xytext=(80, 30), textcoords='offset points', ha='center', arrowprops=dict(arrowstyle="simple",

                            fc="0.6", ec="#3272c1"))

ax.annotate('Halloween', xy=('2014-10-31', 40), xycoords='data', fontsize=15,

            bbox=dict(boxstyle='round', fc='none', ec='#c15832'),

            xytext=(-80, 0), textcoords='offset points', ha='center', arrowprops=dict(arrowstyle="simple",

                            fc="0.6", ec="#c15832"))

ax.xaxis.set_major_locator(mpl.dates.MonthLocator())

ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=15))

ax.xaxis.set_major_formatter(plt.NullFormatter())

ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%h'))

#data.price.sort_values()
data.iloc[[7252, 3914, 9254]]
data.iloc[[1149, 15293, 465]]
#data.groupby(['date'])['price'].mean().sort_values(ascending=False)
data[data.date == '2014-10-11']
#data.date.value_counts().sort_values()
data[data.date.isin(['2015-05-15', '2015-05-24', '2015-05-27'])]
print(data.price.sum() / 21613)

print(data.price.median())
data_to_count = data.drop([17296, 19148, 16594])

data_to_count.price.sum()  / 21610
data_to_count = data.drop([7252, 3914, 9254])

data_to_count.price.sum()  / 21610
fig, ax = plt.subplots(1, 1, figsize=(25, 10))



sns.lineplot(x="date", y="price", data=data.groupby(['date'])['price'].mean().reset_index())

ax.set_title('Price Mean', fontsize=15)

ax.set_xlabel("Date",fontsize=15)

ax.set_ylabel("Price Mean",fontsize=15)



ax.xaxis.set_major_locator(mpl.dates.MonthLocator())

ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=15))

ax.xaxis.set_major_formatter(plt.NullFormatter())

ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%h'))



ax.annotate('Houses nr. 17296 i 19148', xy=('2014-10-11', 2020000), xycoords='data', fontsize=15,

            bbox=dict(boxstyle='round', fc='none', ec='#c62540'),

            xytext=(190, 0), textcoords='offset points', ha='center', arrowprops=dict(arrowstyle="simple",

                            fc="0.6", ec="#c62540"))

ax.annotate('House nr. 16594', xy=('2015-05-27', 1310000.0), xycoords='data', fontsize=15,

            bbox=dict(boxstyle='round', fc='none', ec='#c62540'),

            xytext=(-120, 0), textcoords='offset points', ha='center', arrowprops=dict(arrowstyle="simple",

                            fc="0.6", ec="#c62540"))
fig, ax = plt.subplots(1,3,figsize=(25, 6))

sns.barplot(x='index', y='day_of_week', data=data['day_of_week'].value_counts().reindex(index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).reset_index(), ax=ax[0]) #reindex jest potrzebne do zachowania odpowiedniej kolejności dni

ax[0].set_title('Number of houses sold in a day week', fontsize=15)

ax[0].set_xlabel("Day",fontsize=15)

ax[0].set_ylabel("Value",fontsize=15)

ax[0].set_xticklabels(data['day_of_week'].value_counts().reindex(index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).index, rotation=90, fontsize=15)



sns.barplot(x='day_of_week', y='price', 

            data=data.groupby(['day_of_week'])['price'].mean().reindex(index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).reset_index(), ax=ax[1]) #reindex jest potrzebne do zachowania odpowiedniej kolejności dni

ax[1].set_title('Price mean of houses sold in a day week', fontsize=15)

ax[1].set_xlabel("Day",fontsize=15)

ax[1].set_ylabel("Price mean",fontsize=15)

ax[1].set_xticklabels(data['day_of_week'].value_counts().reindex(index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).index, rotation=90, fontsize=15)



sns.barplot(x='day_of_week', y='price', 

            data=data.groupby(['day_of_week'])['price'].sum().reindex(index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).reset_index(), ax=ax[2]) #reindex jest potrzebne do zachowania odpowiedniej kolejności dni

ax[2].set_title('Total value of houses sold in a day week', fontsize=15)

ax[2].set_xlabel("Day",fontsize=15)

ax[2].set_ylabel("Total value",fontsize=15)

ax[2].set_xticklabels(data['day_of_week'].value_counts().reindex(index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).index, rotation=90, fontsize=15)

data.price.describe()
data[data.price == data.price.min()]
data[data.price == data.price.max()]
print(data.price.kurt())

print(data.price.skew())
fig, ax = plt.subplots(1,2, figsize=(25, 6))

sns.distplot(data.price, ax=ax[0])

ax[0].set_title('Price', fontsize=15)

ax[0].set_xlabel("Price",fontsize=15)

sns.distplot(data[data.price <= 1500000].price, ax=ax[1])

ax[1].set_title('Price', fontsize=15)

ax[1].set_xlabel("Price",fontsize=15)
(data[data.price < 1000000]['price'].sum() * 100) / (data.price.sum())
data.bedrooms.describe()
data[data.bedrooms == data.bedrooms.min()]['bedrooms'].value_counts()
data[data.bedrooms == data.bedrooms.min()]['sqft_living'].mean()
data[data.bedrooms == data.bedrooms.min()]['price'].mean()
data[data.bedrooms == data.bedrooms.max()]
#zależność pomiedzy ceną a ilością łazienek

plt.style.use('seaborn-notebook')

fig, ax = plt.subplots(1,3,figsize=(25, 6))

data.bedrooms.value_counts().sort_index().plot.bar(ax=ax[0])

ax[0].set_title('Number of houses/Number of rooms', fontsize=15)

ax[0].set_xlabel("Number of rooms",fontsize=15)

ax[0].set_ylabel("Number of houses",fontsize=15)

ax[0].set_xticklabels(data.bedrooms.value_counts().sort_index().index, rotation=0, fontsize=15)



sns.boxplot(x="bedrooms", y="price", data=data, ax=ax[1])

ax[1].set_title('Price/Number of rooms', fontsize=15)

ax[1].set_xlabel("Number of rooms",fontsize=15)

ax[1].set_ylabel("Price",fontsize=15)

#ax[1].set_xticklabels(data.bedrooms.value_counts().sort_index().index, rotation=90, fontsize=15)



data[(data.price < 2000000) & (data.bedrooms < 10)].plot.hexbin(x='bedrooms', y='price', gridsize=10, ax=ax[2])

ax[2].set_title('Price/Number of rooms', fontsize=15)

ax[2].set_xlabel("Number of rooms",fontsize=15)

ax[2].set_ylabel("Price",fontsize=15)

#ax[2].set_xticklabels(data.bedrooms.value_counts().sort_index().index, rotation=90, fontsize=15)
(data[data.bedrooms.isin([2,3,4,5])]['bedrooms'].value_counts().sum() * 100) / 21613 
data[data.bedrooms == 11]
data.bathrooms.describe()
print('Number of houses without bathrooms: ' + data[data.bathrooms == data.bathrooms.min()]['bathrooms'].value_counts().sum().astype(str))

print('Average sqft_living of houses without bathrooms: ' + str(data[data.bathrooms == data.bathrooms.min()]['sqft_living'].mean()))

print('Average price of houses without bathrooms: ' + str(data[data.bathrooms == data.bathrooms.min()]['price'].mean()))
print('Number of houses with 8 bathrooms: ' + data[data.bathrooms == data.bathrooms.max()]['bathrooms'].value_counts().sum().astype(str))

print('Average sqft_living of houses with 8 bathrooms: ' + str(data[data.bathrooms == data.bathrooms.max()]['sqft_living'].mean()))

print('Average price of houses with 8 bathrooms: ' + str(data[data.bathrooms == data.bathrooms.max()]['price'].mean()))
data[data.bathrooms == 8]
fig, ax = plt.subplots(figsize=(25, 6))

sns.countplot(data.bathrooms, ax=ax)

ax.set_title('Number of houses/Number of bathrooms', fontsize=15)

ax.set_xlabel("Number of bathrooms",fontsize=15)

ax.set_ylabel("Number of houses",fontsize=15)

ax.set_xticklabels(data.bathrooms.value_counts().sort_index().index, rotation=0, fontsize=15)
plt.style.use('default')

sns.jointplot(x='bathrooms', y='price', data=data[(data.price <= 1500000) & (data.bathrooms <= 5.0)], kind='hex', gridsize=10)
plt.style.use('seaborn-notebook')

fig, ax = plt.subplots(figsize=(25, 6))

sns.boxplot(x='bathrooms', y='price', data = data, ax=ax)

ax.set_title('Price/Number of bathrooms', fontsize=15)

ax.set_xlabel("Number of bathrooms",fontsize=15)

ax.set_ylabel("Price",fontsize=15)
data.sqft_living.describe()
print(data.sqft_living.kurt())

print(data.sqft_living.skew())
sns.kdeplot(data.sqft_living)
sns.distplot(data[data.sqft_living < 7000]['sqft_living'], bins=30, kde=False)
sns.jointplot(x='sqft_living', y='price', data=data[(data.price < 2000000) & (data.sqft_living <= 5000)], kind='hex')
plt.style.use('seaborn-notebook')

fig, ax = plt.subplots(2,2,figsize=(25, 20))



data_to_plot = data[(data.price < 6000000) & (data.sqft_living <= 11000) & (data.bedrooms <= 7) & (data.bathrooms <= 5.0)]

data_to_plot.plot.scatter(x='sqft_living', y='price', ax=ax[0][0])

ax[0][0].set_title('Price/sqft_living', fontsize=15)

ax[0][0].set_xlabel("sqft_living",fontsize=15)

ax[0][0].set_ylabel("Price",fontsize=15)

data_to_plot.plot.scatter(x='sqft_living', y='price', ax=ax[0][1], c='condition', colormap='viridis')

ax[0][1].set_title('Price/sqft_living', fontsize=15)

ax[0][1].set_xlabel("sqft_living",fontsize=15)

ax[0][1].set_ylabel("Price",fontsize=15)

data_to_plot.plot.scatter(x='sqft_living', y='price', ax=ax[1][0], c='bedrooms', colormap='viridis')

ax[1][0].set_title('Price/sqft_living', fontsize=15)

ax[1][0].set_xlabel("sqft_living",fontsize=15)

ax[1][0].set_ylabel("Price",fontsize=15)

data_to_plot.plot.scatter(x='sqft_living', y='price', ax=ax[1][1], c='bathrooms', colormap='viridis')

ax[1][1].set_title('Price/sqft_living', fontsize=15)

ax[1][1].set_xlabel("sqft_living",fontsize=15)

ax[1][1].set_ylabel("Price",fontsize=15)
data.condition.describe()
data.condition.value_counts()
sns.lmplot(x='sqft_living', y='price', col='condition', hue='condition', fit_reg=False, data=data_to_plot, col_wrap=5, height=3)
sns.lmplot(x='log_sqft_living', y='log_price', hue='condition', data=data, fit_reg=False)
data.sqft_lot.describe()
print(data.sqft_lot.kurt())

print(data.sqft_lot.skew())
sns.kdeplot(data.sqft_lot)
sns.distplot(data[data.sqft_lot < 250000]['sqft_lot'], bins=90)
data_sqft_lot = data[(data.price < 6000000) & (data.sqft_lot <= 175000) & (data.bedrooms <= 7) & (data.bathrooms <= 5.0)]

sns.lmplot(x='sqft_lot', y='price', col='condition', hue='condition', fit_reg=False, data=data_sqft_lot, col_wrap=5, height=4)
data_sqft_lot = data[(data.price < 6000000) & (data.sqft_lot <= 175000) & (data.bedrooms <= 7) & (data.bathrooms <= 5.0)]

sns.lmplot(x='sqft_lot', y='price', col='bathrooms', hue='bathrooms', fit_reg=False, data=data_sqft_lot, col_wrap=5, height=4)
data.floors.describe()
data.floors.value_counts()
fig, ax = plt.subplots(1,2, figsize=(25, 6))

sns.countplot(data.floors, ax=ax[0])

sns.boxplot(x='floors', y='price', data=data[data.price <= 6000000], ax=ax[1])
data.waterfront.describe()
data.waterfront.value_counts()
sns.lmplot(x='sqft_living', y='price', hue='waterfront', data=data, fit_reg=False)
fig, ax = plt.subplots(3,2, figsize=(25, 20))

sns.boxplot(x='condition', y='price', hue='waterfront', data=data_sqft_lot, ax=ax[0][0])

sns.boxplot(x='bedrooms', y='price', hue='waterfront', data=data_sqft_lot, ax=ax[0][1])

sns.boxplot(x='bathrooms', y='price', hue='waterfront', data=data_sqft_lot, ax=ax[1][0])

sns.boxplot(x='floors', y='price', hue='waterfront', data=data_sqft_lot, ax=ax[1][1])

sns.boxplot(x='grade', y='price', hue='waterfront', data=data_sqft_lot, ax=ax[2][0])

sns.boxplot(x='view', y='price', hue='waterfront', data=data_sqft_lot, ax=ax[2][1])
data.view.describe()
data.view.value_counts()
print((data[data.view == 0]['view'].value_counts() * 100) / 21613)

print((data[data.view == 4]['view'].value_counts() * 100) / 21613)
sns.violinplot(x='view', y='price', data=data_sqft_lot)
data.grade.describe()
fig, ax = plt.subplots(1,2, figsize=(25, 6))

sns.countplot(data.grade, ax=ax[0])

sns.violinplot(x='grade', y='price', data=data_sqft_lot, ax=ax[1])
f, ax = plt.subplots(figsize=(8,6))



matcorr = data.corr()

n = 10

cols = matcorr.nlargest(n, 'grade').index

cm = np.corrcoef(data[cols].values.T)

sns.heatmap(cm, annot=True, xticklabels=cols.values, yticklabels=cols.values)
var = 'yr_built'

data_to_plot = pd.concat([data['price'], data[var]], axis=1)

f, ax = plt.subplots(figsize=(20,8))

fig = sns.boxplot(x=var, y='price', data=data_to_plot[data_to_plot.price < 1500000])

plt.xticks(rotation=90)
f, ax = plt.subplots(figsize=(20,8))

sns.countplot(data.yr_built)

ax.set_title('Number of houses built in 1900-2015', fontsize=15)

ax.set_xlabel("Year",fontsize=15)

ax.set_ylabel("Value",fontsize=15)

ax.set_xticklabels(data['yr_built'].sort_values().unique(), rotation=90,fontsize=10)
#data.yr_renovated.value_counts()
data.yr_renovated.unique()
print((data[data.yr_renovated > 0]['yr_renovated'].value_counts().sum() * 100) / 21613)
sns.lmplot(x='sqft_living', y='price', hue='renovated', data=data, fit_reg=False)
sns.lmplot(x='log_sqft_living', y='log_price', hue='renovated', data=data, fit_reg=False)
f, ax = plt.subplots(figsize=(20,8))

sns.countplot(data[data.yr_renovated > 0]['yr_renovated'])

ax.set_title('Number of houses renovated in 1934-2015', fontsize=15)

ax.set_xlabel("Year",fontsize=15)

ax.set_ylabel("Value",fontsize=15)

ax.set_xticklabels(data[data.yr_renovated > 0]['yr_renovated'].sort_values().unique(), rotation=90,fontsize=10)
data['lat'].describe() 
data['long'].describe() 
sns.jointplot(x='long', y='lat', data=data, kind='hex')
fig, ax = plt.subplots(figsize=(10, 10))

sns.kdeplot(data['long'], data['lat'], shade=True, ax=ax)

ax.annotate('Seattle (District 4, 5, 6, 7)', xy=(-122.35, 47.68), xycoords='data', fontsize=15,

            bbox=dict(boxstyle='round', fc='none', ec='#0844a5'),

            xytext=(20, 80), textcoords='offset points', ha='center', arrowprops=dict(arrowstyle="simple",

                            fc="0.6", ec="#0844a5"))

ax.annotate('Seattle (District 3)', xy=(-122.305, 47.61), xycoords='data', fontsize=15,

            bbox=dict(boxstyle='round', fc='none', ec='#0844a5'),

            xytext=(140, -20), textcoords='offset points', ha='center', arrowprops=dict(arrowstyle="simple",

                            fc="0.6", ec="#0844a5"))

ax.annotate('Seattle (District 1, 2)', xy=(-122.38, 47.54), xycoords='data', fontsize=15,

            bbox=dict(boxstyle='round', fc='none', ec='#0844a5'),

            xytext=(15,-100), textcoords='offset points', ha='center', arrowprops=dict(arrowstyle="simple",

                            fc="0.6", ec="#0844a5"))

ax.annotate('Lake Washington', xy=(-122.24, 47.63), xycoords='data', fontsize=15,

            bbox=dict(boxstyle='round', fc='none', ec='#0844a5'),

            xytext=(120, 0), textcoords='offset points', ha='center', arrowprops=dict(arrowstyle="simple",

                            fc="0.6", ec="#0844a5"))

ax.annotate('Kirkland', xy=(-122.2, 47.71), xycoords='data', fontsize=15,

            bbox=dict(boxstyle='round', fc='none', ec='#0844a5'),

            xytext=(120, 0), textcoords='offset points', ha='center', arrowprops=dict(arrowstyle="simple",

                            fc="0.6", ec="#0844a5"))

ax.annotate('Renton', xy=(-122.18, 47.49), xycoords='data', fontsize=15,

            bbox=dict(boxstyle='round', fc='none', ec='#0844a5'),

            xytext=(120, 0), textcoords='offset points', ha='center', arrowprops=dict(arrowstyle="simple",

                            fc="0.6", ec="#0844a5"))

ax.annotate('Maple Valley', xy=(-122.04, 47.36), xycoords='data', fontsize=15,

            bbox=dict(boxstyle='round', fc='none', ec='#0844a5'),

            xytext=(120, 0), textcoords='offset points', ha='center', arrowprops=dict(arrowstyle="simple",

                            fc="0.6", ec="#0844a5"))
data.zipcode.describe()
data.zipcode.unique()
fig, ax = plt.subplots(figsize=(25,10))

sns.countplot(data.zipcode, ax=ax)

ax.set_title('Nmber of houses sold according zipcode', fontsize=20)

ax.set_xlabel("Zipcode",fontsize=15)

ax.set_ylabel("Value",fontsize=15)

ax.set_xticklabels(data.zipcode.value_counts().sort_index().index, rotation=90, fontsize=15)
data[data.zipcode.isin([98039, 98148])]['city'].unique()
data[data.zipcode.isin([98103, 98038, 98115])]['city'].unique()
#data.city.value_counts()
fig, ax = plt.subplots(figsize=(25,10))

sns.countplot(data.city, ax=ax)

ax.set_title('Number of huses sold in cities', fontsize=20)

ax.set_xlabel("City",fontsize=15)

ax.set_ylabel("Value",fontsize=15)

ax.set_xticklabels(data.city.unique(), rotation=90, fontsize=15)
palette = {"MEDINA, WA":"#f7a431","BELLEVUE, WA":"#159e09","MERCER ISLAND, WA":"#43ff32", "SEATTLE, WA":"#006a91", "KIRKLAND, WA":'#af0000',

          "SAMMAMISH, WA":"#ff5959", "WOODINVILLE, WA":"#f8fc0a", "REDMOND, WA":'#49ceff', "ISSAQUAH, WA": "#ff5b5b", 

           "FALL CITY, WA": "#38bc6d"}



fig, ax = plt.subplots(figsize=(25,10))



mydata = data.groupby(['city', 'zipcode'])['price'].mean().sort_values(ascending=False).reset_index().head(30)

sns.barplot(x='zipcode', y='price', data=mydata, hue='city', dodge=False, palette=palette, order=mydata['zipcode'], ax=ax)

ax.set_title('Price mean', fontsize=20)

ax.set_xlabel("Zipcode",fontsize=15)

ax.set_ylabel("Price mean",fontsize=15)

ax.set_xticklabels(mydata['zipcode'], rotation=90, fontsize=15)

ax.legend(prop={'size': 15})

fig, ax = plt.subplots(figsize=(25,10))



mydata = data.groupby(['city'])['price'].mean().sort_values(ascending=False).reset_index()

sns.barplot(x='city', y='price', data=mydata, order=mydata['city'], ax=ax)

ax.set_title('Price mean', fontsize=20)

ax.set_xlabel("City",fontsize=15)

ax.set_ylabel("Price mean",fontsize=15)

ax.set_xticklabels(mydata['city'], rotation=90, fontsize=15)
sns.boxplot(x='view', y='price', hue='waterfront', data=data)
fig=plt.figure(figsize=(19,12.5))

ax=fig.add_subplot(2,2,1, projection="3d")

ax.scatter(data['floors'],data['bedrooms'],data['bathrooms'],c="darkred",alpha=.5)

ax.set(xlabel='\nFloors',ylabel='\nRooms',zlabel='\nBathrooms')

ax.set(ylim=[0,12])



ax=fig.add_subplot(2,2,2, projection="3d")

ax.scatter(data['floors'],data['bedrooms'],data['sqft_living'],c="darkred",alpha=.5)

ax.set(xlabel='\nFloors',ylabel='\nRooms',zlabel='\nsqft_living')

ax.set(ylim=[0,12])



ax=fig.add_subplot(2,2,3, projection="3d")

ax.scatter(data['sqft_living'],data['sqft_lot'],data['bathrooms'],c="darkblue",alpha=.5)

ax.set(xlabel='\nsqft_living',ylabel='\nsqft_lot',zlabel='\nBathrooms')

ax.set(ylim=[0,250000])



ax=fig.add_subplot(2,2,4, projection="3d")

ax.scatter(data['sqft_living'],data['sqft_lot'],data['bedrooms'],c="darkblue",alpha=.5)

ax.set(xlabel='\nsqft_living',ylabel='\nsqft_lot',zlabel='\nRooms')

ax.set(ylim=[0,250000])
fig=plt.figure(figsize=(9.5,6.25))

ax=fig.add_subplot(1,1,1, projection="3d")

ax.scatter(data['view'],data['grade'],data['yr_built'],c="purple",alpha=.5)

ax.set(xlabel='\nView',ylabel='\nGrade',zlabel='\nBuilt')
sns.jointplot(x='sqft_living', y='sqft_above', data=data[(data.sqft_living <= 5000)], kind='hex')
#data.sqft_basement.value_counts()
print((data[data.sqft_basement > 0]['sqft_basement'].value_counts().sum() * 100) / 21613)
sns.jointplot(x='sqft_living', y='sqft_basement', data=data[(data.sqft_basement > 0) & (data.sqft_living <= 5000)], kind='hex')
sns.jointplot(x='sqft_living', y='sqft_living15', data=data[(data.sqft_living15 <= 5000) & (data.sqft_living <= 5000)], kind='hex')
sns.jointplot(x='sqft_lot', y='sqft_lot15', data=data[(data.sqft_lot15 <= 15000) & (data.sqft_lot <= 15000)], kind='hex')
data.to_csv('king_county.csv', index=False)


























































