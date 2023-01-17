import pandas as pd # package for high-performance, easy-to-use data structures and data analysis

import numpy as np # fundamental package for scientific computing with Python

# for plotting

import matplotlib.pyplot as plt

import seaborn as sns



# Supress unnecessary warnings so that presentation looks clean

import warnings

warnings.filterwarnings("ignore")



%matplotlib inline
data = pd.read_csv('../input/en.openfoodfacts.org.products.tsv',  sep='\t')
print ("The number of rows in the dataset is " + str(data.shape[0]))

print ("The number of columns in the dataset is " + str(data.shape[1]))
data.head(2)
# Print all rows and columns

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
data.head(2)
# how much percentage of missing values are there

format(round(data.isnull().values.ravel().sum()/(data.shape[0] * data.shape[1]), 2))
# how many categorical and continuous features are there in the data

data.get_dtype_counts()
data.info() #it will also show the memory usage
# Lets explore the countries features

countries_columns = ['countries','countries_tags','countries_en']
data[countries_columns].head()
data['countries_en'] = data['countries_en'].str.lower()
data['countries_en'].head()
data.countries_en.str.extract(r"(united states.*)").unique()
data['countries_en'].isnull().values.ravel().sum()
data = data[data.countries_en.notnull()]

data['countries_en'].isnull().values.ravel().sum()
dataset= {}

dataset['us'] = data[data['countries_en'].str.contains("united states")]

dataset['india'] = data[data['countries_en'].str.contains("india")]

dataset['france'] = data[data['countries_en'].str.contains("france")]

dataset['uk'] = data[data['countries_en'].str.contains("united kingdom")]

dataset['spain'] = data[data['countries_en'].str.contains("spain")]

dataset['aus'] = data[data['countries_en'].str.contains("australia")]

dataset['brazil'] = data[data['countries_en'].str.contains("brazil")]
country_food_info_values = []
for key in dataset.keys():

    country_food_info_values.append([key, 

                             round(dataset[key]['energy_100g'].mean(), 2),

                             round(dataset[key]['fat_100g'].mean(), 2),

                             round(dataset[key]['cholesterol_100g'].mean(), 2),

                             round(dataset[key]['nutrition-score-fr_100g'].mean(), 2)]

                            )
country_food_info_values
country_food_info = pd.DataFrame(country_food_info_values, columns=['country','energy_100g', 'fat_100g', 'cholesterol_100g', 'nutrition_score_fr_100g'])
country_food_info
#Lets plot the meann values for the seven countries below

columns=['energy_100g', 'fat_100g', 'cholesterol_100g', 'nutrition_score_fr_100g']



fig, axis = plt.subplots(4,1,figsize=[20,20])



for i, col in enumerate(columns):

    sns.barplot(x= 'country', y = col, data = country_food_info, ax = axis[i])

                         