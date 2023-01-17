import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as sm
import re
import sqlite3
from datetime import datetime
from pandas import Series, DataFrame
sns.set()
%matplotlib inline
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 200)
data = pd.read_csv('../input/en.openfoodfacts.org.products.tsv', sep='\t', low_memory=False)
print('There are {:,} rows '.format(data.shape[0]) + 'and {} columns in our data'.format(data.shape[1]))
data.head()
data.set_index('code', inplace=True)
print("This data is really messy! \nthe 'countries_en' column contians {} \"unique\" countries ".format(data['countries_en'].dropna().unique().shape[0]))
# We drop all columns that we deem not interesting and we don't intend to use for our investigation.
data = data.drop(columns=['creator',
                          'brands',
                          'brands_tags',
                          'categories',
                          'main_category',
                          'countries',
                          'countries_tags',
                          'additives',
                          'additives_tags',
                          'categories_tags',
                          'states',
                          'states_en',
                          'states_tags',
                          'url',
                          'quantity',
                          'packaging_tags',
                          'packaging',
                          'created_t',
                          'last_modified_t',
                          'ingredients_from_palm_oil_n', 
                          'ingredients_that_may_be_from_palm_oil_n',
                          'pnns_groups_1',
                          'pnns_groups_2',
                          'image_url',
                          'image_small_url',
                         ])

print('There are {:,} rows '.format(data.shape[0]) + 'and {} columns left in our data'.format(data.shape[1]))
data.isnull().sum().sort_values()

# We rename all columns that contain a "-" since in some scenarios this can cause problems with python 3
data = data.rename(columns={'nutrition-score-fr_100g': 'nutrition_score_fr_100g',
                            'nutrition-score-uk_100g': 'nutrition_score_uk_100g',
                            'vitamin-c_100g': 'vitamin_c_100g',
                            'vitamin-a_100g': 'vitamin_a_100g',
                            'saturated-fat_100g': 'saturated_fat_100g',
                            'trans-fat_100g': 'trans_fat_100g'})
data.isnull().sum().plot(kind='hist', figsize=(15,10))
plt.title('Distribution of NaNs')
plt.xlabel('NaNs')

plt.show()
# We drop all columns that contain less than 20% usable data
data = data.dropna(axis=1, thresh= len(data)*0.2, how='all') 
# drop all rows that (after dropping some columns) only contain NaNs
data = data.dropna(axis=0, how='all') 
print('There are now {:,} rows '.format(data.shape[0]) + 'and {} columns left in our data'.format(data.shape[1]))
data.isnull().sum().sort_values()
data['product_name'].fillna(value='Product name unavailable', inplace=True)
data.head()
# We check whether there are any duplicates in our data (this excludes out index-column)
data.duplicated().sum()
data[data.duplicated(keep=False)]
# We drop all duplicates from our data
data.drop_duplicates(inplace=True)
data.dtypes
data['created_datetime'] = pd.to_datetime(data['created_datetime'], format='%Y-%m-%dT%H:%M:%SZ', errors='coerce')
data['last_modified_datetime'] = pd.to_datetime(data['last_modified_datetime'], format='%Y-%m-%dT%H:%M:%SZ', errors='coerce')
data[['created_datetime', 'last_modified_datetime']].isnull().sum()
# We fills the nulls
data['created_datetime'].fillna(method='ffill', inplace=True)
data['last_modified_datetime'].fillna(method='ffill', inplace=True)
data.describe()
data.select_dtypes(include=float).plot(kind='box', subplots=True, title='Our Data before Outlier-Elimination', figsize=(20,20), layout=(6,4))

plt.show()
sns.pairplot(data, x_vars=['nutrition_score_fr_100g'], y_vars=['nutrition_score_uk_100g'], size=5)

plt.show()
data = data[(data[data.columns.difference(['energy_100g'])] <= 100).all(1)]
data = data[(data[data.columns.difference(['nutrition_score_fr_100g', 'nutrition_score_uk_100g'])] >= 0).all(1)]
data = data[(data['fat_100g'] + data['carbohydrates_100g'] + data['proteins_100g'])<=100]
data[data['nutrition_score_uk_100g'] != data['nutrition_score_fr_100g']].shape
data[data['nutrition_score_uk_100g'] == data['nutrition_score_fr_100g']].shape
sns.pairplot(data, x_vars=['nutrition_score_fr_100g'], y_vars=['nutrition_score_uk_100g'], size=5)

plt.show()
data['nutrition_score_100g'] = data['nutrition_score_uk_100g'][data['nutrition_score_uk_100g'] == data['nutrition_score_fr_100g']]
data = data.drop(columns=['nutrition_score_uk_100g', 'nutrition_score_fr_100g'])
data.describe()
# We eliminate all values (outliers) that are more than 3 std's from the mean
data = data[np.abs(data['additives_n']-data['additives_n'].mean())<=(3*data['additives_n'].std())]
data = data[np.abs(data['energy_100g']-data['energy_100g'].mean())<=(3*data['energy_100g'].std())]
data = data[np.abs(data['fat_100g']-data['fat_100g'].mean())<=(3*data['fat_100g'].std())]
data = data[np.abs(data['saturated_fat_100g']-data['saturated_fat_100g'].mean())<=(3*data['saturated_fat_100g'].std())]
data = data[np.abs(data['trans_fat_100g']-data['trans_fat_100g'].mean())<=(3*data['trans_fat_100g'].std())]
data = data[np.abs(data['cholesterol_100g']-data['cholesterol_100g'].mean())<=(3*data['cholesterol_100g'].std())]
data = data[np.abs(data['carbohydrates_100g']-data['carbohydrates_100g'].mean())<=(3*data['carbohydrates_100g'].std())]
data = data[np.abs(data['sugars_100g']-data['sugars_100g'].mean())<=(3*data['sugars_100g'].std())]
data = data[np.abs(data['fiber_100g']-data['fiber_100g'].mean())<=(3*data['fiber_100g'].std())]
data = data[np.abs(data['proteins_100g']-data['proteins_100g'].mean())<=(3*data['proteins_100g'].std())]
data = data[np.abs(data['salt_100g']-data['salt_100g'].mean())<=(3*data['salt_100g'].std())]
data = data[np.abs(data['sodium_100g']-data['sodium_100g'].mean())<=(3*data['sodium_100g'].std())]
data = data[np.abs(data['vitamin_a_100g']-data['vitamin_a_100g'].mean())<=(3*data['vitamin_a_100g'].std())]
data = data[np.abs(data['vitamin_c_100g']-data['vitamin_c_100g'].mean())<=(3*data['vitamin_c_100g'].std())]
data = data[np.abs(data['calcium_100g']-data['calcium_100g'].mean())<=(3*data['calcium_100g'].std())]
data = data[np.abs(data['iron_100g']-data['iron_100g'].mean())<=(3*data['iron_100g'].std())]
data = data[np.abs(data['nutrition_score_100g']-data['nutrition_score_100g'].mean())<=(3*data['nutrition_score_100g'].std())]
print("There are {:,} rows left in our data ".format(data.shape[0]))
data.describe()
data.select_dtypes(include=float).plot(kind='box', subplots=True, title='Our Data after Outlier-Elimination', figsize=(20,20), layout=(6,4))

plt.show()
# We create a new column 'time_delta' that represents the difference in time between creation and modification
data['time_delta'] = (data['created_datetime'] - data['last_modified_datetime'])
# We use the column 'serving_size' to indicate whether the product is liquid or solid (1/0)
data['is_liquid_binary'] = data['serving_size'].str.contains('l|oz', case=False).dropna().astype(int)
data['liquid/solid'] = data['is_liquid_binary'].map({1:'liquid',0:'solid'})
# We use a regular expression to extract the numeric amount of grams from the 'serving_size' column
data['serving_size_in_g'] = data['serving_size'].str.extract('(\d?\d?\d)', expand=True ).dropna().astype(int)
data.isnull().sum()
data.dropna(subset=['is_liquid_binary', 'liquid/solid', 'serving_size_in_g'], inplace=True)
data.head()
# We split all entries in the 'additives_en' column and create a new row for each in one new dataframe
exp_additives = data['additives_en'].str.split(',').apply(Series, 1).stack()
exp_additives.index = exp_additives.index.droplevel(-1)
exp_additives.name = 'additives_exp'
data_exp_additives = data.join(exp_additives)
# We split all entries in the 'ingredients_text' column and create a new row for each in one new dataframe
exp_ingredients = data['ingredients_text'].str.split(',').apply(Series, 1).stack()
exp_ingredients.index = exp_ingredients.index.droplevel(-1)
exp_ingredients.name = 'ingredients_exp'
data_exp_ingredients = data.join(exp_ingredients)
# We split all entries in the 'categories_en' column and create a new row for each in one new dataframe
# We use the 'categories_en' column rather than th 'main_column' because there are many NaNs and this way we get more data
exp_categories = data['categories_en'].str.split(',').apply(Series, 1).stack()
exp_categories.index = exp_categories.index.droplevel(-1)
exp_categories.name = 'categories_exp'
data_exp_categories = data.join(exp_categories)
# We split all entries in the 'countries_en' column and create a new row for each in one new dataframe
exp_countries = data['countries_en'].str.split(',').apply(Series, 1).stack()
exp_countries.index = exp_countries.index.droplevel(-1)
exp_countries.name = 'countries_exp'
data_exp_countries = data.join(exp_countries)
stand_data = data.select_dtypes(include=float).transform(lambda x: (x - x.mean()) / x.std())
stand_data.describe()
stand_data.dropna(inplace=True)
print('There are {:,} rows '.format(stand_data.shape[0]) + 'and {} columns in our standardized data'.format(stand_data.shape[1]))
data.describe()
data.hist(figsize=(20,20), layout=(6,4))

plt.show()
data.reset_index(inplace=True)
data.set_index('created_datetime', inplace=True)
data['time_delta'].describe()
print('The mean difference between the time an entry was createted and modified is {} days'.format(data['time_delta'].mean().days))
products_per_months = data['product_name'].groupby(data.index.month)
products_per_months.describe()
# We resample our data 
products_per_month = (data.resample('M')['product_name'].count())
# Amount of Products created each month
products_per_month.plot(kind='bar', figsize=(20,10), title='Products created by Month')

plt.show()
mean_products_per_month = products_per_month.groupby(products_per_month.index.month).mean()
mean_products_per_month
# Mean amount of Product created by month
mean_products_per_month.plot(kind='bar', figsize=(20,10), color='blue', title='Mean Products created by Month')

plt.show()
data.set_index('code', inplace=True)
print('There are {:,} unique additives in our data'.format(data_exp_additives['additives_exp'].dropna().unique().shape[0]))
print('These are the 10 most common additives in our data:\n{}'.format(data_exp_additives['additives_exp'].value_counts().head(10)))
data_exp_additives['additives_exp'].value_counts().head(10).sort_values().plot(kind='barh', figsize=(10,10))
plt.xlabel('Frequency')
plt.ylabel('Additives')
plt.title('The 10 most used Additives')

plt.show()
print('There are {:,} unique ingredients in our data'.format(data_exp_ingredients['ingredients_exp'].dropna().unique().shape[0]))
print('Thesee are five most common ingredients in our data:\n{}'.format(data_exp_ingredients['ingredients_exp'].value_counts().head(5)))
data_exp_ingredients['ingredients_exp'].value_counts().head(10).sort_values().plot(kind='barh', figsize=(10,10))
plt.xlabel('Frequency')
plt.ylabel('Ingredients')
plt.title('The 10 most used Ingredients')

plt.show()
data_exp_categories.reset_index(inplace=True)
print("There are {:,} unique categories in our data".format(data_exp_categories['categories_exp'].dropna().unique().shape[0]))
print('Thesee are five most common categories in our data:\n{}'.format(data_exp_categories['categories_exp'].value_counts().head(5)))
# Excluding Categories that appear less then 10 times in our data.
categories_filtered = data_exp_categories.groupby('categories_exp').filter(lambda x: len(x) >= 10)
# We group this dataframe by their categories
categories_grouped = categories_filtered.groupby('categories_exp')
categories_grouped.describe()
fat_top_ten = categories_grouped['fat_100g'].describe().sort_values(by='mean',ascending=False).head(10)
fat_top_ten
index = fat_top_ten.index
mean = fat_top_ten['mean'] 
std = fat_top_ten['std']

plt.figure(figsize=(15,10))
x_pos = [i for i, _ in enumerate(index)]
plt.figsize=(20,10)
plt.bar(x_pos, mean, yerr=std)
plt.xlabel("Categories")
plt.ylabel("Mean Fat Content per 100g")
plt.title("The 10 Categories with the highest average Fat Content per 100g")
plt.xticks(x_pos, index, rotation=30, horizontalalignment="right")


plt.show()
protein_top_ten = categories_grouped['proteins_100g'].describe().sort_values(by='mean',ascending=False).head(10)
protein_top_ten
index = protein_top_ten.index
mean = protein_top_ten['mean'] 
std = protein_top_ten['std']

plt.figure(figsize=(15,10))
x_pos = [i for i, _ in enumerate(index)]
plt.figsize=(20,10)
plt.bar(x_pos, mean, yerr=std)
plt.xlabel("Categories")
plt.ylabel("Mean Protein Content per 100g")
plt.title("The 10 Categories with the highest average Protein Content per 100g")
plt.xticks(x_pos, index, rotation=30, horizontalalignment="right")


plt.show()
carbs_top_ten = categories_grouped['carbohydrates_100g'].describe().sort_values(by='mean',ascending=False).head(10)
carbs_top_ten
index = carbs_top_ten.index
mean = carbs_top_ten['mean'] 
std = carbs_top_ten['std']

plt.figure(figsize=(15,10))
x_pos = [i for i, _ in enumerate(index)]
plt.figsize=(20,10)
plt.bar(x_pos, mean, yerr=std)
plt.xlabel("Categories")
plt.ylabel("Mean Carbohydrate Content per 100g")
plt.title("The 10 Categories with the highest average Carbohydrate Content per 100g")
plt.xticks(x_pos, index, rotation=30, horizontalalignment="right")


plt.show()
sugar_top_10 = categories_grouped['sugars_100g'].describe().sort_values(by='mean',ascending=False).head(10)
sugar_top_10
index = sugar_top_10.index
mean = sugar_top_10['mean'] 
std = sugar_top_10['std']

plt.figure(figsize=(15,10))
x_pos = [i for i, _ in enumerate(index)]
plt.figsize=(20,10)
plt.bar(x_pos, mean, yerr=std)
plt.xlabel("Categories")
plt.ylabel("Mean Sugar Content per 100g")
plt.title("The 10 Categories with the highest average Suger Content per 100g")
plt.xticks(x_pos, index, rotation=30, horizontalalignment="right")


plt.show()
un_healthy_top_10 = categories_grouped['nutrition_score_100g'].describe().sort_values(by='mean',ascending=False).head(10)
un_healthy_top_10
healthy_top_10 = categories_grouped['nutrition_score_100g'].describe().sort_values(by='mean',ascending=True).head(10)
healthy_top_10
index = healthy_top_10.index
mean = healthy_top_10['mean'] 
std = healthy_top_10['std']

plt.figure(figsize=(15,10))
x_pos = [i for i, _ in enumerate(index)]
plt.figsize=(20,10)
plt.bar(x_pos, mean, yerr=std)
plt.xlabel("Categories")
plt.ylabel("Mean Nutrition Score per 100g")
plt.title("The 10 unhealthiest Categories")
plt.xticks(x_pos, index, rotation=30, horizontalalignment="right")
plt.gca().invert_yaxis()



plt.show()
index = un_healthy_top_10.index
mean = un_healthy_top_10['mean'] 
std = un_healthy_top_10['std']

plt.figure(figsize=(15,10))
x_pos = [i for i, _ in enumerate(index)]
plt.figsize=(20,10)
plt.bar(x_pos, mean, yerr=std)
plt.xlabel("Categories")
plt.ylabel("Mean Nutrition Score per 100g")
plt.title("The 10 unhealthiest Categories")
plt.xticks(x_pos, index, rotation=30, horizontalalignment="right")


plt.show()
print("There are {:,} unique countries in our data".format(data_exp_countries['countries_exp'].dropna().unique().shape[0]))
print('This is the how often each Country appears in our data: \n{}'.format(data_exp_countries['countries_exp'].value_counts())) 
# We will focus on The US, France and Canada since there is just not enough data for all the other Countries
data_exp_countries = data_exp_countries[(data_exp_countries['countries_exp']=='United States')|
                                        (data_exp_countries['countries_exp']=='France')|
                                        (data_exp_countries['countries_exp']=='Canada')]
countries_grouped = data_exp_countries.groupby('countries_exp')
countries_grouped.describe()
country_fat_top_ten = countries_grouped['fat_100g'].describe().sort_values(by='mean',ascending=False).head(10)
country_fat_top_ten
index = country_fat_top_ten.index
mean = country_fat_top_ten['mean'] 
std = country_fat_top_ten['std']

plt.figure(figsize=(15,10))
x_pos = [i for i, _ in enumerate(index)]
plt.figsize=(20,10)
plt.bar(x_pos, mean, yerr=std)
plt.xlabel("Countries")
plt.ylabel("Mean Fat Content per 100g")
plt.title("Mean Fat Content per Country")
plt.xticks(x_pos, index)


plt.show()
country_carbs_top_ten = countries_grouped['carbohydrates_100g'].describe().sort_values(by='mean',ascending=False).head(10)
country_carbs_top_ten
index = country_carbs_top_ten.index
mean = country_carbs_top_ten['mean'] 
std = country_carbs_top_ten['std']

plt.figure(figsize=(15,10))
x_pos = [i for i, _ in enumerate(index)]
plt.figsize=(20,10)
plt.bar(x_pos, mean, yerr=std)
plt.xlabel("Countries")
plt.ylabel("Mean Carbohydrate Content per 100g")
plt.title("Mean Carbohydrate Content per Country")
plt.xticks(x_pos, index)


plt.show()
country_protein_top_ten = countries_grouped['proteins_100g'].describe().sort_values(by='mean',ascending=False).head(10)
country_protein_top_ten
index = country_protein_top_ten.index
mean = country_protein_top_ten['mean'] 
std = country_protein_top_ten['std']

plt.figure(figsize=(15,10))
x_pos = [i for i, _ in enumerate(index)]
plt.figsize=(20,10)
plt.bar(x_pos, mean, yerr=std)
plt.xlabel("Countries")
plt.ylabel("Mean Protein Content per 100g")
plt.title("Mean Protein Content per Country")
plt.xticks(x_pos, index)


plt.show()
country_sugar_top_ten = countries_grouped['sugars_100g'].describe().sort_values(by='mean',ascending=False).head(10)
country_sugar_top_ten
index = country_sugar_top_ten.index
mean = country_sugar_top_ten['mean'] 
std = country_sugar_top_ten['std']

plt.figure(figsize=(15,10))
x_pos = [i for i, _ in enumerate(index)]
plt.figsize=(20,10)
plt.bar(x_pos, mean, yerr=std)
plt.xlabel("Countries")
plt.ylabel("Mean Sugars Content per 100g")
plt.title("Mean Sugar Content per Country")
plt.xticks(x_pos, index)


plt.show()
country_healthy_top_10 = countries_grouped['nutrition_score_100g'].describe().sort_values(by='mean',ascending=True).head(10)
country_healthy_top_10
index = country_healthy_top_10.index
mean = country_healthy_top_10['mean'] 
std = country_healthy_top_10['std']

plt.figure(figsize=(15,10))
x_pos = [i for i, _ in enumerate(index)]
plt.figsize=(20,10)
plt.bar(x_pos, mean, yerr=std)
plt.xlabel("Countries")
plt.ylabel("Mean Nutrition Score per 100g")
plt.title("Mean Nutrition Score per Country")
plt.xticks(x_pos, index)

plt.show()
sns.pairplot(data, size=5, hue='liquid/solid',
            x_vars=['nutrition_score_100g'],
            y_vars= ['energy_100g', 'additives_n', 'fat_100g', 'carbohydrates_100g', 'proteins_100g', 'sugars_100g',
                      'saturated_fat_100g', 'trans_fat_100g', 'cholesterol_100g', 'fiber_100g', 
                     'salt_100g', 'sodium_100g', 'vitamin_a_100g', 'vitamin_c_100g', 'calcium_100g', 
                     'iron_100g', 'nutrition_score_100g','is_liquid_binary', 'serving_size_in_g'])

plt.show()
sns.pairplot(data, size=5, hue='nutrition_grade_fr',
            x_vars=['energy_100g'],
            y_vars= ['additives_n', 'fat_100g', 'carbohydrates_100g', 'proteins_100g', 'sugars_100g',
                      'saturated_fat_100g', 'trans_fat_100g', 'cholesterol_100g', 'fiber_100g', 
                     'salt_100g', 'sodium_100g', 'vitamin_a_100g', 'vitamin_c_100g', 'calcium_100g', 
                     'nutrition_score_100g','iron_100g', 'is_liquid_binary', 'serving_size_in_g'])

plt.show()
data.corr(method = "pearson")
corr = data.corr(method = "pearson")

f, ax = plt.subplots(figsize=(15, 15))

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), 
            cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax)
# Source: https://code.i-harness.com/en/q/10f46da
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]
print('The top 5 Correlations between variables are: \n{}'.format(get_top_abs_correlations(data.select_dtypes(include=float), n=5)))
# We exclude the variable 'sodium_100g' for our regression since it is so highly correlated with 'salt_100g'
energy_regression = sm.ols(formula= """energy_100g ~ additives_n
                                   + fat_100g
                                   + fiber_100g
                                   + saturated_fat_100g
                                   + trans_fat_100g
                                   + cholesterol_100g
                                   + carbohydrates_100g
                                   + sugars_100g
                                   + proteins_100g
                                   + salt_100g
                                   + vitamin_a_100g
                                   + vitamin_c_100g
                                   + calcium_100g
                                   + iron_100g
                                   + nutrition_score_100g
                                   + is_liquid_binary
                                   + serving_size_in_g""", data = stand_data)
res = energy_regression.fit()
print(res.summary())
nutrition_score_regression = sm.ols(formula= """nutrition_score_100g ~ additives_n
                                   + energy_100g
                                   + fat_100g
                                   + fiber_100g
                                   + saturated_fat_100g
                                   + trans_fat_100g
                                   + cholesterol_100g
                                   + carbohydrates_100g
                                   + sugars_100g
                                   + proteins_100g
                                   + salt_100g
                                   + vitamin_a_100g
                                   + vitamin_c_100g
                                   + calcium_100g
                                   + iron_100g
                                   + is_liquid_binary
                                   + serving_size_in_g""", data = stand_data)
res = nutrition_score_regression.fit()
print(res.summary())
#Creating an empty database
db = sqlite3.connect("product.db")
cursor = db.cursor()
query_1 = "CREATE TABLE macrn_code (code TEXT PRIMARY KEY, carbohydrates_100g FLOAT, fat_100g FLOAT, proteins_100g FLOAT, nutrition_score_100g INT);"
query_2 = "CREATE TABLE enrgy_code (code TEXT PRIMARY KEY, energy_100g FLOAT, product_name FLOAT, sugars_100g FLOAT, additives_n INT);"
cursor.execute(query_1)
cursor.execute(query_2)
# We reduce our data to 1000 randomly selected rows and reset the index
data_sample = data.reset_index().sample(n=1000)
data_macr = data_sample[['code', 'carbohydrates_100g','fat_100g', 'proteins_100g', 'nutrition_score_100g']]
data_nrgy = data_sample[['code', 'energy_100g', 'product_name', 'sugars_100g', 'additives_n']]
data_macr.to_sql(name='macrn_code',con=db,if_exists='append',index=False)
data_nrgy.to_sql(name='enrgy_code',con=db,if_exists='append',index=False)
def run_query(query):
    return pd.read_sql_query(query,db)
query ='''
SELECT carbohydrates_100g FROM macrn_code
INNER JOIN enrgy_code
ON macrn_code.code = enrgy_code.code
WHERE macr_score.nutrition_score_100g > 0
'''
#run_query(query)