# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing dataset 



df = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')
pd.set_option('display.max_columns', None)
# viewing column and the data inside them and how they correlate



df.head()
#now viewing which column is empty and of which data type



df.info()
#countries with most booking, month wise



#grouping by countries and making dummy data of month

gc = pd.get_dummies(df, columns=['arrival_date_month']).groupby('country').sum()



#selecting countries which have more than 27000 arrival_data_day_month

sc = gc.loc[gc['arrival_date_day_of_month']>=27000]



#removing prefix from every month

rp = sc.rename(columns = lambda x: x.replace('arrival_date_month_', ''))



#selecting particular columns

sp = rp.loc[:, 'April':'September']



plt.figure(figsize = (18,8))

ax = sns.lineplot(data= sp, dashes=False)

ax.set_xlabel('Countries', fontsize = 20)

ax.set_ylabel('Visit Count', fontsize = 20)

ax.set_title('Month Wise Visiting Countries', fontsize = 30)

plt.show()
#in this we can see that which country spends the most amount of money when the are visiting



#countries who visits the most

some_countries = ('PRT', 'GBR', 'FRA', 'ESP', 'DEU', 'ITA', 'IRL', 'BEL', 'BRA', 'NLD','USA','CHE')



#counting how much time those top countries appear

dft = df.loc[df['country'].isin(some_countries)]['country']



plt.rcParams['figure.figsize'] = (18, 8)

sns.set_style('whitegrid')

ax = sns.lineplot(x = dft, y = df['adr'])

ax.set_xlabel('Country name', fontsize =20)

ax.set_ylabel('Expenditure',fontsize =20)

ax.set_title('Expenditure of Visiting Countries', fontsize = 30)

plt.show()
#combining the stays in weekends and stays in week days to get the total number nights spent

stay = df['stays_in_weekend_nights'] + df['stays_in_week_nights']



plt.rcParams['figure.figsize'] = (18, 8)

sns.set_style('whitegrid')

ax = sns.lineplot(x = dft, y = stay)

ax.set_xlabel('Country name', fontsize =20)

ax.set_ylabel('Night Spent',fontsize =20)

ax.set_title('Country wise spending nights', fontsize = 30)

plt.show()
#summing up the adults, children and babies column to see how much people visit



people = df['adults'] + df['children'] + df['babies']



plt.rcParams['figure.figsize'] = (15, 7)

sns.set_style('whitegrid')

ax = sns.lineplot(x = dft, y = people)

ax.set_xlabel('Country name', fontsize =20)

ax.set_ylabel('People visiting',fontsize =20)

ax.set_title('Country Wise People visiting', fontsize = 30)

plt.show()
#in this we will see which hotel books the most



plt.figure(figsize = (18,8))

sns.set_style('darkgrid')

ax = sns.countplot(x = 'hotel', data = df, hue = 'is_canceled', palette = 'pink')

ax.set_xlabel('Hotel', fontsize=20)

ax.set_ylabel('Hotel Count', fontsize=20)

ax.set_title('Type of Hotel', fontsize=30)

plt.show()
#how much hotel gets canceled



canc_count = df['is_canceled'].value_counts()

canc_count
#which hotel gets canceled the most



canceled_hotel = df[(df['is_canceled']==1) & (df['hotel'])]['hotel'].value_counts()



canceled_hotel
#now see percent wise cancelation of hotel



labels = canceled_hotel.index

data  = canceled_hotel.values



plt.rcParams['figure.figsize'] = (15,9)



plt.pie(data, labels = labels,autopct='%1.1f%%',shadow=True, startangle=90)

plt.axis('equal')

plt.title("Canceled Hotel Percent by Type", fontsize=20)

plt.show()
#which hotel got most booking in which month



plt.figure(figsize = (18,8))

ax = sns.countplot( x = df['arrival_date_month'],data = df, hue = 'hotel', palette = 'husl')

ax.set_xlabel('Month', fontsize = 20)

ax.set_ylabel('Hotels', fontsize = 20)

ax.set_title('Month with Type of Hotel', fontsize = 30)

plt.show()
#which type of hotel has the highest lead_time



res_lead = df[(df['hotel']=='Resort Hotel')]['lead_time'].sum()

cit_lead = df[(df['hotel']=='City Hotel')]['lead_time'].sum()
plt.figure(figsize = (18,8))

ax = sns.barplot(x = ['Resort','City'],y = [res_lead,cit_lead],palette = 'magma')

ax.set_xlabel('Hotel', fontsize=20)

ax.set_ylabel('Lead Time Total', fontsize=20)

ax.set_title('Type of Hotel with Lead Time', fontsize=30)

plt.show()
some_countries = ('PRT', 'GBR', 'FRA', 'ESP', 'DEU', 'ITA', 'IRL', 'BEL', 'BRA', 'NLD','USA','CHE')

dft = df.loc[df['country'].isin(some_countries)]['country']



plt.rcParams['figure.figsize'] = (18, 8)

sns.set_style('whitegrid')

ax = sns.countplot(dft, hue = df['hotel'], palette = 'bone')

ax.set_xlabel('Country name', fontsize =20)

ax.set_ylabel('Tourist Trip',fontsize =20)

ax.set_title('Countries like for Particular type of hotel', fontsize = 30)

plt.show()
#dropping this one hotel, it is the only hotel which is getting 5400 amount



df.index[df['adr']==5400]

df.drop([48515],inplace = True)
#which type of hotel earns the most





sns.violinplot(x = 'hotel', y='adr', data =df)

ax.set_xlabel('Hotel Type', fontsize =20)

ax.set_ylabel('Earning Amount',fontsize =20)

ax.set_title('Earning of Hotel by Type', fontsize = 30)

plt.show()
#month with most number of bookings



plt.figure(figsize = (18,8))

sns.set_style("darkgrid")

ax = sns.countplot(x = df['arrival_date_month'], data = df)

ax.set_xlabel('Month', fontsize = 20)

ax.set_ylabel('Hotels', fontsize = 20)

ax.set_title('Month With most booking', fontsize = 30)



plt.show()



#month with most number of lead_time



plt.figure(figsize = (18,8))

sns.set_style("whitegrid")

ax = sns.violinplot(x = 'arrival_date_month', y = 'lead_time' ,data=df)

ax.set_xlabel('Month', fontsize = 20)

ax.set_ylabel('Lead Time', fontsize = 20)

ax.set_title('Most Number of Lead Time', fontsize = 30)

plt.show()
#in which month hotel get canceled the most



canceled_month = df[(df['is_canceled']==1) & (df['arrival_date_month'])]['arrival_date_month'].value_counts()
plt.figure(figsize = (18,8))

sns.set_style("darkgrid")

ax = sns.lineplot(x = canceled_month.index, y = canceled_month.values, markers=True, dashes=False)

ax.set_xlabel('Month', fontsize = 20)

ax.set_ylabel('Canceled Hotels', fontsize = 20)

ax.set_title('Month With most canceled hotel', fontsize = 30)



plt.show()
labels = canceled_month.index

sizes = canceled_month.values

colors = plt.cm.rainbow(np.linspace(0,3))



explode = (0.2,0.1, 0, 0,0,0,0,0,0,0,0,0)

plt.rcParams['figure.figsize'] = (15,9)



plt.pie(sizes, labels=labels,explode = explode ,  autopct='%1.1f%%',shadow=True, startangle=90, colors = colors)



plt.axis('equal')  

plt.title("Month with most canceled hotel by Percent", fontsize =20)

plt.show()
#month where hotels gained most

sns.stripplot(x = 'arrival_date_month' , y = 'adr', data = df)

ax.set_xlabel('Month', fontsize =20)

ax.set_ylabel('Earning',fontsize =20)

ax.set_title('Month Wise Earning of Hotel', fontsize = 30)

plt.show()