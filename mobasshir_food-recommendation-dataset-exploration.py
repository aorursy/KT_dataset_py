# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', 100) # Setting pandas to display a N number of columns

pd.set_option('display.max_rows', 10) # Setting pandas to display a N number rows

pd.set_option('display.width', 1000) # Setting pandas dataframe display width to N

from scipy import stats # statistical library

from statsmodels.stats.weightstats import ztest # statistical library for hypothesis testing

import plotly.graph_objs as go # interactive plotting library

import pandas_profiling # library for automatic EDA

%pip install autoviz # installing and importing autoviz, another library for automatic data visualization

from autoviz.AutoViz_Class import AutoViz_Class

from IPython.display import display # display from IPython.display

from itertools import cycle # function used for cycling over values



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)

#     for filename in filenames:

#         pass

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# for k,v in df.items():

#     print(k)

#     print('#' * len(k))

#     print(v.columns.to_list())

#     print('*' * 10)

#     print('*' * 10)
# df = dict()

# for dirname, _, filenames in os.walk('/kaggle/input/zomato-restaurants-data'):

#     for filename in filenames:

#         if filename.endswith('.csv'):

#             name = filename.split('.')[0]

#             df[name] = pd.read_csv(os.path.join(dirname, filename))

# #         elif filename.endswith('.json'):

# #             import json

# #             from pandas.io.json import json_normalize  

# #             with open(os.path.join(dirname, filename)) as f: 

# #                 _ = json.load(f) 

# #             name = filename.split('.')[0]

# #             print(_)

# #             df[name] = pd.read_json(_, orient='index')

            

#         print(os.path.join(dirname, filename))
zomato_restaurent_df = pd.read_csv('/kaggle/input/zomato-restaurants-data/zomato.csv', encoding='latin-1')

display(zomato_restaurent_df.head(4))
# report = pandas_profiling.ProfileReport(zomato_restaurent_df)
# display(report)
# AV = AutoViz_Class()

# report_2 = AV.AutoViz('/kaggle/input/zomato-restaurants-data/zomato.csv')
df = dict()

import os

for dirname, _, filenames in os.walk('/kaggle/input/restaurant-recommendation-challenge'):

    for filename in filenames:

        if filename.endswith('.csv'):

            name = filename.split('.')[0]

            df[name] = pd.read_csv(os.path.join(dirname, filename))

        print(os.path.join(dirname, filename))
display(df['orders'].head(4))
display(df['vendors'].head(4))
display(df['train_locations'].head(4))
display(df['train_customers'].head(4))
# display(df['train_full'].head(4))
display(df['train_full'].head(4))
!cat /kaggle/input/restaurant-recommendation-challenge/VariableDefinitions.txt
df = dict()

import os

for dirname, _, filenames in os.walk('/kaggle/input/19560-indian-takeaway-orders'):

    for filename in filenames:

        if filename.endswith('.csv'):

            name = filename.split('.')[0]

            df[name] = pd.read_csv(os.path.join(dirname, filename))

        print(os.path.join(dirname, filename))
display(df['restaurant-1-orders'].head(4))
display(df['restaurant-2-orders'].head(4))
display(df['restaurant-1-products-price'].head(4))
display(df['restaurant-2-products-price'].head(4))
df = dict()

import os

for dirname, _, filenames in os.walk('/kaggle/input/zomato-restaurants-hyderabad'):

    for filename in filenames:

        if filename.endswith('.csv'):

            name = filename.split('.')[0]

            df[name] = pd.read_csv(os.path.join(dirname, filename))

        print(os.path.join(dirname, filename))
display(df['Restaurant names and Metadata'].head(4))
display(df['Restaurant reviews'].head(4))
df = dict()

for dirname, _, filenames in os.walk('/kaggle/input/amazon-fine-food-reviews'):

    for filename in filenames:

        if filename.endswith('.csv'):

            name = filename.split('.')[0]

            df[name] = pd.read_csv(os.path.join(dirname, filename))

        print(os.path.join(dirname, filename))
display(df['Reviews'].head(4))
df = dict()

for dirname, _, filenames in os.walk('/kaggle/input/food-demand-forecasting'):

    for filename in filenames:

        if filename.endswith('.csv'):

            name = filename.split('.')[0]

            df[name] = pd.read_csv(os.path.join(dirname, filename))

        print(os.path.join(dirname, filename))
display(df['meal_info'].head(4))
display(df['train'].head(4))
display(df['fulfilment_center_info'].head(4))
df = dict()

for dirname, _, filenames in os.walk('/kaggle/input/food-preference'):

    for filename in filenames:

        if filename.endswith('.csv'):

            name = filename.split('.')[0]

            name = dirname.split('/')[-1] + '_' + name

            df[name] = pd.read_csv(os.path.join(dirname, filename))

        print(os.path.join(dirname, filename))
display(df['food-preference_food_coded'].head(4))
display(df['food-choices_food_coded'].head(4))
# df = dict()

# for dirname, _, filenames in os.walk('/kaggle/input/av-genpact-hack-dec2018'):

#     for filename in filenames:

#         if filename.endswith('.csv'):

#             name = filename.split('.')[0]

#             df[name] = pd.read_csv(os.path.join(dirname, filename))

#         print(os.path.join(dirname, filename))
# df['meal_info'].columns
# df['train'].columns
# df['fulfilment_center_info'].columns
!wget https://raw.githubusercontent.com/altosaar/food2vec/master/dat/kaggle_and_nature.csv
with open('/kaggle/working/kaggle_and_nature.csv') as f:

    print(f.read())