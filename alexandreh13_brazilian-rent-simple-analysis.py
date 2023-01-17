import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("../input/brasilian-houses-to-rent/houses_to_rent_v2.csv")
data.tail()
data.shape
data.dtypes
data.columns = [x.strip().replace(" (R$)",'').replace(' ','_') for x in data.columns]

data.columns
data.isnull().sum()
counts = data.city.value_counts()

counts
plt.figure(figsize = (16, 6))

data.city.value_counts().plot(kind='pie', autopct='%.2f%%')
pet_accept_count = data[data.animal == 'acept'].city.value_counts()

pet_accept_count
pet_dict = {}

for i, v in pet_accept_count.items():

    pet_dict[i] = (v/counts[i])*100



pet_series = pd.Series(pet_dict)

print(pet_series.sort_values(ascending=False))

pet_series.sort_values(ascending=False).plot(kind='bar')
data['rooms'].value_counts().plot(kind='bar')
data['bathroom'].value_counts().plot(kind='bar')
data['parking_spaces'].value_counts().plot(kind='bar')
common_places = data.loc[(data.rooms==3) & (data.bathroom==1) & (data.parking_spaces==1)]

common_places
plt.title("Avarage rent of common places for each city")

plt.ylabel("Rent")

print(common_places.groupby('city').rent_amount.mean())

common_places.groupby('city').rent_amount.mean().plot(kind='bar')
sb.set_style('darkgrid')

def mean_prices_plot(feature):

    plt.figure(figsize = (18, 6))

    sb.barplot(x=data.city, y=data[feature])
feature_prices = data[['hoa', 'rent_amount', 'property_tax', 'fire_insurance', 'total']]



for feature in feature_prices:

    mean_prices_plot(feature)

bh_hoa = data[data['city'] == 'Belo Horizonte']

mean_bh_hoa = bh_hoa.hoa.mean()

std_bh_hoa = bh_hoa.hoa.std()



print("Belo Horizonte hoa mean and std -> {} | {}".format(mean_bh_hoa, std_bh_hoa))
out_hoa = bh_hoa.hoa.mean() + bh_hoa.hoa.std()

bh_hoa[bh_hoa.hoa>out_hoa]
outliers_index = bh_hoa[bh_hoa.hoa>out_hoa].index



print("Deleted samples: ", outliers_index)

for index in outliers_index:

    data.drop(index)
sp_prop_tax = data[data.city=='São Paulo']

sp_prop_tax.describe()
mean_sp_tax = sp_prop_tax.property_tax.mean()

std_sp_tax = sp_prop_tax.property_tax.std()



print("São Paulo property tax mean and std -> {} | {}".format(mean_sp_tax, std_sp_tax))
out_tax = mean_sp_tax+std_sp_tax

sp_prop_tax[sp_prop_tax.property_tax>out_tax]
outliers_index = sp_prop_tax[sp_prop_tax.property_tax>out_tax].index

print("Deleted samples: ", outliers_index)

for index in outliers_index:

    data.drop(index)
correlations = data.corr(method='pearson')

plt.figure(figsize = (16, 8))

sb.heatmap(correlations, vmin=0, vmax=1, annot=True, cmap = plt.cm.RdYlBu_r, linewidths=.7)
plt.figure(figsize = (10, 6))

sb.scatterplot(x=data['hoa'], y=data['total'], hue=data.city, style=data.city)
plt.figure(figsize = (10, 6))

sb.scatterplot(x=data['rent_amount'], y=data['total'], hue=data.city, style=data.city)
plt.figure(figsize = (10, 6))

sb.scatterplot(x=data['property_tax'], y=data['total'], hue=data.city, style=data.city)
plt.figure(figsize = (10, 6))

sb.scatterplot(x=data['fire_insurance'], y=data['total'], hue=data.city, style=data.city)