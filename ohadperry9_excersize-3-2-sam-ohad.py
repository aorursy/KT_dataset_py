import pandas as pd 

import matplotlib.pyplot as plt 





# Reading the data set into the pandas object for processing

dataset = pd.read_csv('/kaggle/input/17k-apple-app-store-strategy-games/appstore_games.csv')



# settings for numbers display 

pd.set_option('float_format', '{:f}'.format)

pd.options.display.float_format = "{:.2f}".format
# SPECIFIY THE UN CHOSEN IDEAS 



# selected idea: find the human potential in the market



# load the data into pandas as a data frame 

data_frame = pd.DataFrame(dataset)







def calc_revenue(row):

    '''

       estimate each app revene

    :param row:

    :return:

    '''

    average_in_app_purchase_price = 0

    min_downloads = row['User Rating Count']

    estimated_number_of_in_app_purchases = min_downloads * 0.05

    if type(row['In-app Purchases']) is list:

        average_in_app_purchase_price = sum(row['In-app Purchases'])



    estimated_in_app_purchase = average_in_app_purchase_price * estimated_number_of_in_app_purchases



    return min_downloads * row['Price'] + estimated_in_app_purchase





def join_languages(languages):

    '''



    :param languages:

    :return:

    '''

    return ' '.join(languages)







data_frame['Price'].fillna(0, inplace=True)

data_frame['User Rating Count'].fillna(0, inplace=True)

data_frame['In-app Purchases'].fillna(0, inplace=True)

data_frame['Languages'].fillna('', inplace=True)





# 1. calc revenue for each app

data_frame['revenue'] = data_frame.apply(lambda row: calc_revenue(row), axis=1)

data_frame['count_apps'] = data_frame.apply(lambda row: 1, axis=1)



df = data_frame

# 2. sum revenue grouped by developer



developers = data_frame.groupby('Developer').agg({'count_apps':'sum', 'revenue':'sum', 'Languages': join_languages}).sort_values(by=['revenue', 'count_apps'], ascending=False)



print(developers.head(20))

small_developers = developers[developers['count_apps'] <= 5][developers['count_apps'] >1][developers['revenue'] > 0]



# developers who built between 2 and 5 apps

#count  318.000000     318.000000

# mean     2.751572    8748.579057

# std      0.975042   29897.381448

# min      2.000000       4.950000

# 25%      2.000000      32.725000

# 50%      2.000000     187.110000

# 75%      3.000000    2963.535000

# max      5.000000  238927.440000



# target group -

# 1. small developers - who developer 2-5 apps

# 2. earned medium size money

# 3. non chinese - different market ?

small_third_precentile_non_chinese_developers = developers[developers['count_apps'] <= 5][developers['count_apps'] >1][developers['revenue'] > 333][developers['revenue'] < 2963][developers['Languages'].str.contains('ZH')]

print(small_third_precentile_non_chinese_developers.head(20))

print(small_third_precentile_non_chinese_developers.describe())






