import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

sns.set()
dataset = pd.read_csv('../input/wfp_market_food_prices.csv', encoding = 'ISO-8859-1')

dataset.head()
# remove *id columns

for i in list(dataset):

    if i.find('id') >= 0:

        del dataset[i]

        

dataset.head()
# so what is unique and freq countries?



country_unique, country_freq = np.unique(dataset['adm0_name'], return_counts = True)

for i in range(country_unique.shape[0]):

    

    # print unsorted

    print(country_unique[i], ': ', country_freq[i])
# now we want to visualize bar graph for top 10 countries

# copy actual arrays



country_unique_cp = country_unique.copy()

country_freq_cp = country_freq.copy()



countries_name, countries_freq = [], []



for i in range(10):

    index = np.argmax(country_freq_cp)

    countries_freq.append(country_freq_cp[index])

    countries_name.append(country_unique_cp[index])

    country_freq_cp = np.delete(country_freq_cp, index, axis = 0)

    country_unique_cp = np.delete(country_unique_cp, index, axis = 0)

    

plt.figure(figsize = (20, 10))

y = np.arange(len(countries_name))

plt.bar(y, countries_freq)

plt.xticks(y, countries_name)

plt.ylabel('freq')

plt.show()
# so what is unique and freq commoduties?



commo_unique, commo_freq = np.unique(dataset['cm_name'], return_counts = True)

for i in range(country_unique.shape[0]):

    

    # print unsorted

    print(commo_unique[i], ': ', commo_freq[i])
# now we want to visualize bar graph for top 10 countries

# copy actual arrays



commo_unique_cp = commo_unique.copy()

commo_freq_cp = commo_freq.copy()



commo_name, commo_freq = [], []



for i in range(10):

    index = np.argmax(commo_freq_cp)

    commo_freq.append(commo_freq_cp[index])

    commo_name.append(commo_unique_cp[index])

    commo_freq_cp = np.delete(commo_freq_cp, index, axis = 0)

    commo_unique_cp = np.delete(commo_unique_cp, index, axis = 0)

    

plt.figure(figsize = (20, 10))

y = np.arange(len(commo_name))

plt.bar(y, commo_freq)

plt.xticks(y, commo_name)

plt.ylabel('freq')

plt.show()
# how about some countries vs commoduties visualization 2014 data, shall we?

# i got some trust issues with original unique id, so i will use LabelEncoder



# copy first

dataset_matrix = dataset[['adm0_name', 'cm_name']].loc[dataset['mp_year'] == 2014].values.copy()



country_name = np.unique(dataset_matrix[:, 0])

food_name = np.unique(dataset_matrix[:, 1])



from sklearn.preprocessing import LabelEncoder



# change into int

for i in range(dataset_matrix.shape[1]):

    dataset_matrix[:, i] = LabelEncoder().fit_transform(dataset_matrix[:, i])

    

country_id = np.unique(dataset_matrix[:, 0])

food_id = np.unique(dataset_matrix[:, 1])



heatmap_2014 = np.zeros([country_id.shape[0], food_id.shape[0]])

for i in range(dataset_matrix.shape[0]):

    x = dataset_matrix[i, 0]

    y = dataset_matrix[i, 1]

    heatmap_2014[x, y] += 1





plt.figure(figsize = (60, 30))

sns.heatmap(heatmap_2014, annot = False, fmt = "d", linewidths = .5)

plt.yticks([i for i in range(country_id.shape[0])], country_name, rotation ='horizontal')

plt.xticks([i for i in range(food_id.shape[0])], food_name, rotation ='vertical')

plt.show()
# How about we check, how many times each country do some exchange every month?



# i got some trust issues with original unique id, so i will use LabelEncoder



# copy first

dataset_matrix = dataset[['adm0_name', 'mp_month']].values.copy()



country_name = np.unique(dataset_matrix[:, 0])

month_name = np.unique(dataset_matrix[:, 1])



from sklearn.preprocessing import LabelEncoder



# change into int

dataset_matrix[:, 0] = LabelEncoder().fit_transform(dataset_matrix[:, 0])

    

country_id = np.unique(dataset_matrix[:, 0])



heatmap_2014 = np.zeros([country_id.shape[0], 12])

for i in range(dataset_matrix.shape[0]):

    x = dataset_matrix[i, 0]

    y = dataset_matrix[i, 1] - 1

    heatmap_2014[x, y] += 1





plt.figure(figsize = (30, 10))

sns.heatmap(heatmap_2014.T, annot = False, fmt = "d", linewidths = .5)

plt.xticks([i for i in range(country_id.shape[0])], country_name, rotation ='vertical')

plt.yticks([i for i in range(month_name.shape[0])], month_name, rotation ='horizontal')

plt.show()
# how about we check, is it the price of commoduties increased every year?



years = np.unique(dataset['mp_year'])

prices, freq = [], []

for i in range(years.shape[0]):

    price_in_year = dataset['mp_price'].loc[dataset['mp_year'] == years[i]].values.copy()

    freq.append(price_in_year.shape[0])

    prices.append(np.sum(price_in_year))

    

# separate the graph because the gap of the values is too huge

plt.figure(figsize = (40, 10))

plt.subplot(1, 2, 1)

xtick = [i for i in range(years.shape[0])]

plt.plot(xtick, prices, c = 'r', label = 'price')

plt.legend()

plt.title('total payment')

plt.xticks(xtick, years)

plt.subplot(1, 2, 2)

plt.plot(xtick, freq, c = 'g', label = 'freq-transaction')

plt.xticks(xtick, years)

plt.legend()

plt.title('total freq-transaction')

plt.show()