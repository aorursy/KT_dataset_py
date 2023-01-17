# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# reading file

from pandas import DataFrame

food = pd.read_csv('../input/en.openfoodfacts.org.products.tsv',sep='\t',low_memory=False)

food.shape



import warnings

warnings.filterwarnings('ignore')
## Some data cleaning - Removing records and not necessary columns

food_name = food[food['product_name'].notnull()]

food_clean = food_name.drop(food_name.columns[0:6], axis =1 )

list1 = ['no_nutriments','ingredients_from_palm_oil', 'ingredients_that_may_be_from_palm_oil','nutrition_grade_uk']

list2 = ['-butyric-acid_100g','-caproic-acid_100g','-caprylic-acid_100g','-lignoceric-acid_100g','-cerotic-acid_100g']

list3 = ['-melissic-acid_100g','-elaidic-acid_100g','-mead-acid_100g','-erucic-acid_100g','-nervonic-acid_100g']

list4 = ['chlorophyl_100g','glycemic-index_100g','water-hardness_100g','last_modified_datetime','packaging_tags']

list5 = ['brands_tags','manufacturing_places_tags','labels', 'labels_tags','emb_codes', 'emb_codes_tags','categories', 'categories_tags']

list6 = ['origins_tags','countries', 'countries_tags','traces', 'traces_tags','additives', 'additives_tags','cities_tags','allergens']

list7 = ['states', 'states_tags']

list_final = list1 + list2 + list3 + list4 + list5 + list6 + list7

food_clean.drop(list_final, axis = 1, inplace = True)

food_clean = food_clean[food_clean.countries_en.notnull()]
# Selecting the countries that has more than 100 prodcts in file and visulazing the distribution

temp = food_clean['countries_en'].groupby(food_clean.countries_en).count()

food_countries = temp[temp > 100].sort_values(ascending = False)

food_countries.plot(kind = 'bar', title = "No of food product vs Countries", x = "Countries", y = "No. of products")
# Selecting the alcoholic products

alcoholic_pdts = food_clean[food_clean['alcohol_100g'].notnull()]

# Grouping the products by brands

alcohold_grp_by_brand = alcoholic_pdts['brands'].groupby(alcoholic_pdts.brands).count()

# Selcting only those brands that produce more than 25 products

top_alcohol_brands = alcohold_grp_by_brand[alcohold_grp_by_brand > 25].sort_values(ascending = False)

# Plotting the distribution of alcoholic product by brands;

top_alcohol_brands.plot(kind = 'pie', figsize = (5,5), autopct = '%1.1f%%')
##Finding the top alcoholic countries and their average alcohol content in prodcuts

alcohol_pdts_grp_cont = alcoholic_pdts['countries_en'].groupby(alcoholic_pdts['countries_en']).count()

topAlcoholicCountries = alcohol_pdts_grp_cont[alcohol_pdts_grp_cont >= 10]

topAlcoholicCountriesList = topAlcoholicCountries.index.tolist()

meanAlcoholCountry = []

for country in topAlcoholicCountriesList:

    temp = alcoholic_pdts[alcoholic_pdts['countries_en'] == country]

    meanAlcoholCountry.append((country,np.mean(temp['alcohol_100g'])))

df = pd.DataFrame(meanAlcoholCountry)

df = df.set_index(df[0])

df.drop(df.columns['0'], axis = 1, inplace = True)

df.columns = ['mean_alcohol_concentration']

df.plot(kind = 'bar')
