# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
nutrient = pd.read_csv('/kaggle/input/mz-open-health-data/food_nutrient.csv', index_col='fdc_id')

del(nutrient['id'], nutrient['data_points'], nutrient['derivation_id'], nutrient['footnote'], nutrient['min_year_acquired'])

nutrient.head()
food = pd.read_csv('/kaggle/input/mz-open-health-data/food.csv', index_col='fdc_id')

del(food['data_type'], food['publication_date'])

food.head()
portion = pd.read_csv('/kaggle/input/mz-open-health-data/food_portion.csv', index_col='fdc_id')

del(portion['data_points'], portion['min_year_acquired'], portion['footnote'], portion['id'], portion['seq_num'])

portion.head()
branded = pd.read_csv('/kaggle/input/open-usda-nutrition-data-zena2/branded_food.csv', index_col='fdc_id')

del(branded['data_source'], branded['modified_date'],

branded['available_date'], branded['market_country'], branded['discontinued_date'], branded['brand_owner'], branded['gtin_upc'])

branded.head()
df1 = pd.merge(pd.merge(food, nutrient, on='fdc_id'), portion, on='fdc_id')

print(df1.head())
branded.head()