%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
import ast
from scipy import stats
food_data = pd.read_table('../input/world-food-facts/en.openfoodfacts.org.products.tsv', low_memory=False)
print("The Open Food Data dataframe has "+ str(food_data.shape[0]) + ' rows and ' + str(food_data.shape[1]) + ' columns.')
food_data.head()
#Splitting the columns in two, as info() hides column attributes for 100+ column datasets without enlarging the notebook window
food_data.iloc[:,0:100].info() 
food_data.iloc[:,100:163].info()
columns_with_data = []
for column in food_data.columns:
    if food_data[column].notnull().sum()> food_data.shape[0]*0.1:
        columns_with_data.append(column)
food_data_main_columns = food_data[columns_with_data]
food_data_main_columns.head()
food_data_main_columns.groupby('creator').size().sort_values(ascending = False).head(10)
top_creators = ['usda-ndb-import', 'openfoodfacts-contributors', 'kiliweb', 'date-limite-app', 'openfood-ch-import']

for creator in top_creators:
    time_created = pd.to_datetime(food_data_main_columns[food_data_main_columns.creator == creator].created_datetime.str.slice(0,10), format = "%Y/%m/%d")
    plt.figure(figsize=(10, 0.5), facecolor='w', edgecolor='k')
    plt.title(creator)
    plt.yticks([])
    plt.xlabel("Timeline")
    plt.plot_date(time_created, np.full((len(time_created),1), 1))
    
plt.show()
top_products = food_data_main_columns.groupby('product_name').size().sort_values(ascending = False) 
top30_products = food_data_main_columns.groupby('product_name').size().sort_values(ascending = False).head(30)
top30_products.sort_values(ascending = True).plot(kind='barh', figsize = (10,6.5), title = "Top products in terms of number of occurances")
plt.show()
top_brands = food_data_main_columns.groupby('brands_tags').size().sort_values(ascending = False)
top30_brands = food_data_main_columns.groupby('brands_tags').size().sort_values(ascending = False).head(30)
top30_brands.sort_values(ascending = True).plot(kind='barh', figsize = (10,6.5), title = "Top brands in terms of number of occurances")
plt.show()
coca_cola = food_data_main_columns[(food_data_main_columns.product_name.str.lower() == "coca cola")|(food_data_main_columns.product_name.str.lower() == "coca-cola")]
pepsi = food_data_main_columns[food_data_main_columns.product_name.str.lower() == "pepsi"]
pepsi.groupby('brands_tags').size().sort_values(ascending = False)
coca_cola.groupby('brands_tags').size().sort_values(ascending = False)
def column_to_chr_counter(column):
    """
    The function gets a column from a DataFrame and returns a dataframe, sorted by the most frequent group of characters.
    The function is built to handle columns with multiple tags.
    """
    col_list=column.dropna().tolist()
    col_string = ', '.join(col_list)
    col_string = col_string.lower()
    word_counter = Counter(re.split(", |,", col_string))
    char_counter = Counter(word_counter)
    dict_counter = ast.literal_eval(str(char_counter.elements)[42:-2])
    dataframe_counter = pd.DataFrame.from_dict(dict_counter, orient='index', columns = ['Count of occurances'])
    
    return dataframe_counter
columns_with_tags = ['countries_tags',  'manufacturing_places_tags','categories_tags' , 'labels_tags']
for col in columns_with_tags:
    
    #column_to_chr_counter(food_data[col]).head(10).plot(kind='bar', figsize = (7,3.5), title = ("\nTop " + str(col)) )
    #plt.show()
    print("\nTop " + str(col))
    print(column_to_chr_counter(food_data[col]).head(10))
    
uk_sweets_data_annually = pd.read_csv('../input/uk-household-purchases/uk_household_purchases_food.csv')
uk_sweets_data_annually.head()
uk_sweets_data_annually.info()
uk_food_data = food_data[food_data.countries_tags.str.contains('en:united-kingdom') == True]
uk_food_data.shape #Making sure UK has sufficient entries
pepsi_and_cola_uk = uk_food_data[(uk_food_data.product_name.str.lower() == "coca cola")|(uk_food_data.product_name.str.lower() == "coca-cola")|(uk_food_data.product_name.str.lower() == "pepsi")]
column_to_chr_counter(pepsi_and_cola_uk.categories_tags)
uk_food_data_soft_drinks = uk_food_data[uk_food_data.categories_tags.str.contains("en:carbonated-drink|en:sodas|en:sugared-beverages") == True]
uk_soft_drinks_carbo = uk_food_data_soft_drinks.carbohydrates_100g.mean() # 11.58 per 100g
column_to_chr_counter(uk_food_data[uk_food_data.product_name.str.contains("chocolate|cookies") == True].categories_tags)
uk_food_data_sweets = uk_food_data[uk_food_data.categories_tags.str.contains("en:sugary-snacks|chocolates|biscuits|cakes") == True]
uk_sweets_carbo = uk_food_data_sweets.carbohydrates_100g.mean() # 56.38 per 100g
#As the UK subset turned out to have only 1 ice cream, I used instead the whole one to assess the carbo intake
food_data_ice_cream = food_data[food_data.product_name.str.contains("ice cream") == True]
uk_ice_cream_carbo = food_data_ice_cream.carbohydrates_100g.mean() # 28.89 per 100g
#3 columns with the mean values will be added to uk_sweets_data_annually table
dict_uk = dict(((k, eval(k)) for k in ('uk_soft_drinks_carbo', 'uk_ice_cream_carbo', 'uk_sweets_carbo')))
uk_sweets_data_annually = uk_sweets_data_annually.assign(**dict_uk)
uk_sweets_data_annually.head()
uk_sweets_data_annually['who_recommendation'] = 25
uk_sweets_data_annually['sum_of_the_categories'] = uk_sweets_data_annually.soft_drinks_per_day_in_gr*uk_sweets_data_annually.uk_soft_drinks_carbo/100 + uk_sweets_data_annually.ice_cream_per_day_in_gr*uk_sweets_data_annually.uk_ice_cream_carbo/100+uk_sweets_data_annually.confectionery_per_day_in_gr*uk_sweets_data_annually.uk_sweets_carbo/100
uk_sweets_data_annually.head()
plt.bar(uk_sweets_data_annually.year, uk_sweets_data_annually.sum_of_the_categories, color = "grey")
#plt.plot(uk_sweets_data_annually.year, uk_sweets_data_annually.sum_of_the_categories)
plt.plot(uk_sweets_data_annually.year, uk_sweets_data_annually.who_recommendation, c="red" )
plt.legend()
plt.ylabel("Carbohydr. intake per day in gr.")
plt.xlabel('Year')
plt.ylim(0,50)
plt.show()
stats.ttest_1samp(uk_sweets_data_annually.sum_of_the_categories,25.0)
additives_with_warnings = pd.read_csv("../input/food-additives-with-warnings/additives_with_warnings.csv")
additives_with_warnings.head(5)
additives_with_warnings['full_information'] = additives_with_warnings.ingredient+"/"+additives_with_warnings.eu_nr + ": " + additives_with_warnings.information
additives_with_warnings_slice1 = additives_with_warnings[['ingredient', 'full_information']]
additives_with_warnings_slice2 = additives_with_warnings[['eu_nr', 'full_information']]
additives_with_warnings_slice2 = additives_with_warnings_slice2.rename(index=str, columns={"eu_nr": 'ingredient'})
frames = [additives_with_warnings_slice1, additives_with_warnings_slice2]
additives_with_warnings_final = pd.concat(frames, ignore_index = True)
additives_with_warnings_final.head()
food_data['warnings'] = np.nan #Creating a column to contain the warnings
for row in range(additives_with_warnings_final.shape[0]):
    food_data.loc[food_data.additives_en.str.contains(additives_with_warnings_final.ingredient[row], case = False) == True, 'warnings'] = additives_with_warnings_final.full_information[row]
food_data.warnings.unique()
food_data.groupby('warnings').size().sort_values(ascending = False)
food_data[food_data.warnings.str.contains('Banned')==True].countries.unique()
food_data[food_data.warnings.str.contains('Banned')==True].loc[food_data.countries == 'Belgique']
column_to_chr_counter(food_data[food_data.warnings.str.contains('adverse')==True].categories_tags)