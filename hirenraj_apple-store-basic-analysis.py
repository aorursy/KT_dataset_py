# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from csv import reader



#-- function to get dataset

def get_dataset(dataset_file):

    opened_file = open(dataset_file, encoding='UTF-8')

    read_file = reader(opened_file)

    return list(read_file)





#-- lets get apple store data

ios_dataset = get_dataset('/kaggle/input/app-store-apple-data-set-10k-apps/AppleStore.csv')

ios_header = ios_dataset[0]

ios = ios_dataset[1:]

#-- function to explore given dataset

def explore_data(dataset, start, end, rows_and_columns=False):

    dataset_slice = dataset[start:end]

    for row in dataset_slice:

        print(row)

        print('\n')  # adds new line after every row

    if rows_and_columns:

        print('Number of Rows :', len(dataset))

        print('Number of Columns :', len(dataset[0]))
print(ios_header)

explore_data(ios,0, 3, True)
#-- lets be smart about it and remove all the rows which doesn't match 

#-- header length

def remove_dirty_rows(dataset, header, counter=0):

    for row in dataset:

        if len(row) != len(header):

            del dataset[counter]

        counter += 1

    return dataset



ios = remove_dirty_rows(ios,ios_header)
def is_this_string_non_english(string):

    non_ascii = 0

    for character in string:

        if ord(character) > 127:

            non_ascii += 1

    if non_ascii > 3:

        return False

    else:

        return True



        

print(is_this_string_non_english('Instagram'))

print(is_this_string_non_english('爱奇艺PPS -《欢乐颂2》电视剧热播'))

print(is_this_string_non_english('Docs To Go™ Free Office Suite'))

print(is_this_string_non_english('Instachat 😜'))


def get_english_apps(dataset):

    english_apps = []

    non_english_apps = []

    for row in dataset:

        if is_this_string_non_english(row[0]):

            english_apps.append(row)

        else:

            non_english_apps.append(row)

    return english_apps, non_english_apps





ios_english = []

for app in ios:

    name = app[1]

    if is_this_string_non_english(name):

        ios_english.append(app)



(ios_english, ios_non_english) = get_english_apps(ios)





print('\n')

explore_data(ios_english, 0, 3, True)
def freq_table(dataset, index):

    table = {}

    total = 0

    

    for row in dataset:

        total += 1

        value = row[index]

        if value in table:

            table[value] += 1

        else:

            table[value] = 1

    

    table_percentages = {}

    for key in table:

        percentage = (table[key] / total) * 100

        table_percentages[key] = percentage 

    

    return table_percentages





def display_table(dataset, index):

    table = freq_table(dataset, index)

    table_display = []

    for key in table:

        key_val_as_tuple = (table[key], key)

        table_display.append(key_val_as_tuple)

        

    table_sorted = sorted(table_display, reverse = True)

    for entry in table_sorted:

        print(entry[1], ':', entry[0])

    
display_table(ios_english,-5)
genres_ios = freq_table(ios_english, -5)



for genre in genres_ios:

    total = 0

    len_genre = 0

    for app in ios_english:

        genre_app = app[-5]

        if genre_app == genre:            

            n_ratings = float(app[5])

            total += n_ratings

            len_genre += 1

    avg_n_ratings = total / len_genre

    print(genre, ':', avg_n_ratings)
for app in ios_english:

    if app[-5] == 'Navigation':

        print(app[1], ':', app[5]) # printing name and number of ratings
for app in ios_english:

    if app[-5] == 'Reference':

        print(app[1], ':', app[5])