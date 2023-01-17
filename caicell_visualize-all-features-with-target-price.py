# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

pal = sns.color_palette()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
house_train = pd.read_csv('../input/train.csv')



house_train.sample(3)
set(house_train.dtypes.unique())
cat_var = list(house_train.select_dtypes(include = ['O']).columns)

int_var = list(house_train.select_dtypes(include = ['int64']).columns)

float_var = list(house_train.select_dtypes(include = ['float64']).columns)

int_var.remove('price_doc')

print("Categorical variable length : {}".format(len(cat_var)))

print("Int variable length : {}".format(len(int_var)))

print("Float variable length : {}".format(len(float_var)))
def cate_full_mean_graph(data= None, x_list = None, y = None):

    row = len(x_list)

    

    fig, ax = plt.subplots(row, 1, figsize = (14, 5*row))

    for itr, x in enumerate(x_list):

        x_y_data = data[[x, y]].copy()

        x_y_name = list(set(x_y_data[x].unique()))

    

        x_y_mean = x_y_data.groupby(x)[y].mean()

        x_y_mean = x_y_mean.sort_values()

    

        sns.barplot(x=x_y_mean.keys(), y=x_y_mean, ax = ax[itr])

        plt.xlabel(x)



def scatter_full_x_y(data = None, x_list = None, y = None):

    number_int = len(x_list)

    num_row = number_int // 3 + number_int %3

    

    fig, ax = plt.subplots(num_row, 3, figsize = (15, 5*num_row))

    for index in range(num_row):

        row = 3*index

        x_sample = x_list[row:row+3]

        for col in range(3):

            x = x_sample[col]

            x_y_data = data[[x,y]].copy()

            x_y_data = x_y_data.sort_values(x)

            ax_temp = ax[index, col]

            ax_temp.scatter(x_y_data[x], x_y_data[y], alpha = 0.1)

            ax_temp.set_title( x ,fontsize = 15)

cate_full_mean_graph(data = house_train, x_list = cat_var, y = 'price_doc')
#scatter_full_x_y(data = house_train, x_list = int_var, y = 'price_doc')

#scatter_full_x_y(data = house_train, x_list = float_var, y = 'price_doc')
essential = ['full_sq', 'work_all', 'floor', 'num_room', 

'children_school', 'ID_railroad_terminal', 'trc_count_500', 

'mosque_count_500', 'market_count_1500']

trivial = ['build_year', 'full_all',  'healthcare_centers_raion',  'university_top_20_raion', 

 'office_raion', 'young_all']
scatter_full_x_y(data = house_train, x_list = essential, y = 'price_doc')
scatter_full_x_y(data = house_train, x_list = trivial, y = 'price_doc')
build_year = set(house_train['build_year'].unique())

build_year
build_strange_row = [4965, 0, 1, 3, 20, 71] #nan

build_year_list = house_train['build_year']

for x in build_strange_row:

    print('Number of house built in {} : {}'.format(x,(build_year_list == x).sum()))
for x in [1900.0, 1901.0, 1902.0, 1903.0, 1904.0, 1905.0]:

    print('Number of house built in {} : {}'.format(x,(build_year_list == x).sum()))

print('\n')

for x in [1919.0, 1920.0, 1921.0, 1922.0, 1923.0, 1924.0]:

    print('Number of house built in {} : {}'.format(x,(build_year_list == x).sum()))

print('\n')

for x in [1969.0, 1970.0, 1971.0, 1972.0, 1973.0, 1974.0]:

    print('Number of house built in {} : {}'.format(x,(build_year_list == x).sum()))

print('\n')