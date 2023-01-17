import csv

import heapq





class DataStructureClass:

    def __init__(self, data=None):

        """ Create a blank list data"""

        if data is None:

            data = []

        self.data = data



    def load_data(self):

        """ Load file AB_NYC_2019.csv"""

        with open('../input/nycairbnb/AB_NYC_2019.csv', encoding='utf8') as csv_file:

            csv_reader = csv.reader(csv_file)

            next(csv_reader)

            for row in csv_reader:



                self.data.append(row)

            csv_file.close()

        return self.data



    def specific_row(self, number=0, value1='', data_input=None):

        """ Get specific rows with specific value in a column(number)"""

        data = []

        for each in data_input:

            if each[number] == value1:

                data.append(each)

        return data





    def convert_to_heap(self, column=0, data_input=None):

        """ Convert to heap to find minimum value"""

        data = []

        for each in data_input:

            data.append(int(each[column]))

        heapq.heapify(data)



        return data



    def get_aColumn(self, column=0, data_input=None):

        """ Get an integer column"""

        data_output = []

        for each in data_input:

            data_output.append(int(each[column]))



        return data_output

    

    def quick_select(self, S, k):

        """ Return the kth smallest element of list S,

            for k from 1 to len(S).

        """

        S = list(set(S))

        if len(S) == 1:                                 # 1 step

            return S[0]                                 # 2 step

        pivot = S[0]                                    # 2 step         pick first element of S

        L = [x for x in S if x < pivot]                 # 3n steps       elements less than pivot

        E = [x for x in S if x == pivot]                # 3n steps       elements equal to pivot

        G = [x for x in S if pivot < x]                 # 3n steps       elements greater than pivot

        

        if k <= len(L):                                 # 2 steps

            return self.quick_select(L, k)              # 1 step         kth smallest lies in L

        elif k <= len(L) + len(E):                      # 4 steps

            return pivot                                # 1 step         kth smallest equal to pivot

        else:

            j = k - len(L) - len(E)                     # 5 steps        new selection parameter

            return self.quick_select(G, j)              # 1 step         kth smallest is jth in G

        # Total steps = 1+2+2+3n+3n+3n+2+1+4+1+5+1 = 9n + 19 = O(n)

        

    def frequency_element(self, a=None, object=""):

        """ Get the most common element in a column

            It is an adapted version of word_frequency algorithm

            a: a list of element

            object: name of the column

        """

        frequency = {}                                         # 1 step      create a dictionary

        for element in a:                                      # n steps

            frequency[element] = frequency.get(element, 0) +1  # 4n steps    get the frequen of an element with element is key.

        valueList = list(frequency.values())                   # n+1 steps   get all counts of all element.

       

        print("Most frequency of {}:".format(object))    

        for key, value in frequency.items():                   # n steps     (key, value) tuples represent (element, count)

            if value == max(valueList):                        # 2n steps

                print("{} {}".format(key, value))              

                return str(key)                                # 2n steps

        # Total steps = 1+n+4n+n+1+n+2n+2n = 11n + 2 = O(n)
def run_test():

    data = DataStructureClass()

    data.load_data()

    

    print('Use frequency_element Algorithm')

    prices = data.get_aColumn(9, data.data)

    frequency_price = data.frequency_element(prices, "price")



    print('Get the lowest of price: (Use heap data structure)')

    price = data.convert_to_heap(9, data.data)

    lowest_price = price[0]

    print(lowest_price)



    print('Get the fifth lowest of price: (Use quick_select Algorithm)')

    price_fifty = data.quick_select(prices, 5)

    print(price_fifty)



    reviews = data.get_aColumn(11, data.data)

    frequency_review = data.frequency_element(reviews, "review")



    nights = data.get_aColumn(10, data.data)

    frequency_night = data.frequency_element(nights, "night")



    minReviews_rows = data.specific_row(11, frequency_review, data.data)



    # Rows have most frequency reviews and frequency_night

    double_Specific = data.specific_row(10, frequency_night, minReviews_rows)



    print('Selected row: frequency_review, frequency_night.')

    print('Get the lowest of price in selected_rows: (Use heap data structure)')

    selected_price = data.convert_to_heap(9, double_Specific)

    lowest_price = selected_price[1]

    print(lowest_price)



    print('Get the fifth lowest of price in selected_rows: (Use quick_select Algorithm)')

    int_price = data.get_aColumn(9, double_Specific)

    price_fif = data.quick_select(int_price, 5)

    print(price_fif)



    price_list = data.get_aColumn(9, double_Specific)

    frequency_price = data.frequency_element(price_list, "price")





run_test()
import pandas as pd

import numpy as np

import seaborn as sns

# import scipy

import matplotlib.pyplot as plt

# import statsmodels.formula.api as smf

# import random

from sklearn import datasets, linear_model

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
pd.set_option('display.float_format', lambda x:'%.2f'%x)



ab = pd.read_csv('../input/nycairbnb/AB_NYC_2019.csv', low_memory=False)

ab.head()
ab.describe()
ab.isnull().sum()
ab['name']=ab['name'].replace(np.nan, '', regex=True)

ab['host_name']=ab['host_name'].replace(np.nan, '', regex=True)

ab['reviews_per_month']=ab['reviews_per_month'].replace(np.nan, 0)

ab['last_review']=ab['last_review'].replace(np.nan, '2019-01-01')
ab.isnull().sum()
plt.figure(figsize=(10,6))

sns.distplot(ab['price'], rug=True)
ab['room_type'] = ab['room_type'].astype('category')

ab['price'] = pd.to_numeric(ab['price'])
%matplotlib notebook

sns.boxplot(x='room_type', y='price', data=ab)

plt.xlabel('Room type')

plt.ylabel('Price')
%matplotlib notebook

fig = plt.figure()

plt.scatter(ab['longitude'],ab['latitude'], c=ab['price'], cmap='cool', alpha=0.5) 

plt.xlabel('longitude')

plt.ylabel('latitude')

%matplotlib notebook

plt.figure()

scat1 = sns.regplot(x='number_of_reviews', y='price', fit_reg=False, data=ab)



plt.xlabel('reviews')

plt.ylabel('price')
%matplotlib notebook

plt.figure()

scat1 = sns.regplot(x='reviews_per_month', y='price', fit_reg=False, data=ab)



plt.xlabel('reviews_per_month')

plt.ylabel('price')
%matplotlib notebook

plt.figure()

scat1 = sns.regplot(x='calculated_host_listings_count', y='price', fit_reg=False, data=ab)



plt.xlabel('calculated_host_listings_count')

plt.ylabel('price')


%matplotlib notebook

plt.figure()

scat1 = sns.regplot(x='availability_365', y='price', fit_reg=False, data=ab)



plt.xlabel('availability_365')

plt.ylabel('price')


%matplotlib notebook

plt.figure()

scat1 = sns.regplot(x='minimum_nights', y='price', fit_reg=False, data=ab)



plt.xlabel('minimum_nights')

plt.ylabel('price')
X = ab[['id','minimum_nights', 'number_of_reviews', 'calculated_host_listings_count', 'reviews_per_month','latitude','longitude']].dropna()

y = ab['price'].copy()
reg = linear_model.LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print(y_pred)
print(y_test)
print(np.mean(np.square(y_test - y_pred)))
print(np.sqrt(mean_squared_error(y_test, y_pred)))