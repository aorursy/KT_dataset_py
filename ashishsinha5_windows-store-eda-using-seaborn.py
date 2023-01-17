import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('darkgrid')
df = pd.read_csv('../input/windows-store/msft.csv')
df.head()
# Check for nan

df[df.isna().any(axis=1)]
df.dropna(inplace=True)
# Preprocessing, converting string to float

def rupee_to_num(x):

    if x == 'Free':

        return 0

    x = x[2:]

    x = x.replace(",", "")

    return float(x)



def paid_idc(x):

    if x:

        return "Paid"

    return 'Free'



df['Price'] = df['Price'].apply(rupee_to_num)

df['Paid/Free'] = df['Price'].apply(paid_idc)
# Extracting Date attributes

df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].dt.year

df['Month'] = df['Date'].dt.month

df['Day'] = df['Date'].dt.day

df['Day_of_Week'] = df['Date'].dt.dayofweek

df.drop('Date', axis = 1, inplace = True)

df.head()
# Number of apps

num_apps = df['Name'].nunique()

num_cat = df['Category'].nunique()

num_free = len(df[df['Price'] == 0])

num_paid = num_apps - num_free

app_per_cat = dict(df['Category'].value_counts())

print("Number of apps in the dataset - {}".format(num_apps))

print("Number of unique categories - {}".format(num_cat))

print("Number of free apps - {}".format(num_free))

print("Numner of paid apps - {}".format(num_paid))
sns.barplot(x = ['free_apps', 'paid_apps'], y = [num_free, num_paid])

plt.title("Paid vs free apps")

plt.ylabel('Number of apps')

plt.show()
df.groupby('Year').nunique()['Name'].plot()

plt.title("App launches per year")

plt.ylabel('Number of apps')

plt.xticks(df['Year'].unique())

plt.show()
sns.distplot(df['Rating'], bins = 10, kde = False)

plt.ylabel("Frequency")

plt.xticks(np.histogram(df['Rating'])[1])

plt.show()
sns.barplot(list(app_per_cat.values()), list(app_per_cat.keys()),orient= 'h')

plt.title("Number of apps per category")

plt.show()
# Distribution of Ratings per category

fig, ax = plt.subplots(nrows=4, ncols=4, figsize = (20,15))

k = 0

for i in range(3):

    for j in range(4):

        sns.distplot(df[df['Category'] == list(app_per_cat.keys())[k]]['Rating'], bins = 10, kde = False, ax=ax[i,j])

        ax[i,j].set_xticks(np.histogram(df[df['Category'] == list(app_per_cat.keys())[k]]['Rating'])[1])

        ax[i,j].set_ylabel(list(app_per_cat.keys())[k])

        k+=1

sns.distplot(df[df['Category'] == list(app_per_cat.keys())[k]]['Rating'], bins = 10, kde = False, ax=ax[3,0])

ax[3,0].set_ylabel(list(app_per_cat.keys())[k])

ax[3,0].set_xticks(np.histogram(df[df['Category'] == list(app_per_cat.keys())[k]]['Rating'])[1])

plt.show()
df['Rating'].groupby(df['Category']).mean().plot.barh()

plt.title("Average rating per category")
df['No of people Rated'].groupby(df['Category']).mean().plot.barh()

plt.title("Average Number of people rated per app per category ")

plt.show()
sns.scatterplot(df[df['Price'] > 0]['No of people Rated'], df[df['Price'] > 0]['Price'], hue = df['Rating'])

plt.title("Price vs Popularity")

plt.show()
df['Rating'].groupby(df['Paid/Free']).mean().plot.bar()

plt.title("Average Rating for free vs Paid apps")

plt.show()
df['No of people Rated'].groupby(df['Day_of_Week']).sum().plot.bar()

plt.title("Trafic vs Day_of_week")

plt.show()