#import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Reading the data set into the pandas object for processing

dataset = pd.read_csv('/kaggle/input/17k-apple-app-store-strategy-games/appstore_games.csv')



# settings for numbers display 

pd.set_option('float_format', '{:f}'.format)

pd.options.display.float_format = "{:.2f}".format
data_frame = pd.DataFrame(dataset)

 

# Check that data is read by viewing the data frame head    

data_frame.head()
# Use the method `columns` on the dataframe

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.columns.html

data_frame.columns
# How many Rows are in the data frame 

# https://stackoverflow.com/a/32103678/1574104

total_rows = len(data_frame.axes[0])

print(f'number of rows in apps dataset is {total_rows}')
# https://stackoverflow.com/questions/15998491/how-to-convert-ipython-notebooks-to-pdf-and-html/25942111

columns_with_nan_values = data_frame.columns[data_frame.isna().any()].tolist()

print(f'columns with nan values are \n {columns_with_nan_values}')



USER_RATING_COUNT_COLUMN_NAME = 'User Rating Count'

USER_RATING_COLUMN_NAME = 'Average User Rating'

# 1.3. What is the mean, minimum and maximum of user rating count?

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html

# https://stackoverflow.com/a/41328899/1574104

user_rating_statistics = data_frame[USER_RATING_COUNT_COLUMN_NAME].describe()



# 1.5. What are the 4 largest size games that are free?

# • This is a challenge question – you will need to search stack overflow for an answer



data = user_rating_statistics

# print(f'user_rating_statistics: \n{data} \n\n')

print(f'mean = {data["mean"]}')

print(f'minimum = {data["min"]}')

print(f'maximum = {data["max"]}')

# 1.4. What is the mean, minimum and maximum of user rating count, in games with average user rating above 4?

column_data = data_frame[data_frame[USER_RATING_COLUMN_NAME] > 4.0][USER_RATING_COUNT_COLUMN_NAME]

user_rating_statistics_above_four = column_data.describe()



data = user_rating_statistics_above_four

# print(f'user_rating_statistics_above_four: \n{data} \n\n')

print(f'mean = {data["mean"]}')

print(f'minimum = {data["min"]}')

print(f'maximum = {data["max"]}')

# 1.5. What are the 4 largest size games that are free?

PRICE_COLUMN_NAME = 'Price'

APP_SIZE_COLUMN_NAME = 'Size'

NUMBER_OF_APPS_TO_SHOW = 4

all_free_games = data_frame[data_frame[PRICE_COLUMN_NAME] == 0.0]



# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html

# sorting by size. decending

all_free_games_sorted_by_size = all_free_games.sort_values(by=[APP_SIZE_COLUMN_NAME], ascending=False)

all_free_games_sorted_by_size.head(4)


