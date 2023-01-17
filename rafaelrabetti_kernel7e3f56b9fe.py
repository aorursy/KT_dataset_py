# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor

from fastai import *



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_file_path = '../input/120642-manhattan/train.csv'

exemplo_file_path = '../input/120642-manhattan/exemplo.csv'

valid_file_path = '../input/120642-manhattan/valid.csv'

test_file_path = '../input/120642-manhattan/test.csv'



train_data = pd.read_csv(train_file_path, low_memory=False, parse_dates=["sale_date"])

exemplo_data = pd.read_csv(exemplo_file_path)

valid_data = pd.read_csv(valid_file_path, low_memory=False, parse_dates=["sale_date"])

test_data = pd.read_csv(test_file_path, low_memory=False, parse_dates=["sale_date"])



train_data.tail()





valid_data.describe()


train_data.columns
train_data = train_data.dropna(axis=0)



train_data = train_data.query("sale_id <= 21588")



y = train_data.sale_price

y.describe()
this_data = pd.concat([valid_data, test_data], ignore_index=True)

this_data.head()
this_data = this_data.drop(columns=["ease-ment", "address", "apartment_number", "building_class_category", "building_class_at_present"])

train_data = train_data.drop(columns=["ease-ment", "address", "apartment_number", "sale_price", "building_class_category", "building_class_at_present"])



train_data.gross_square_feet = train_data.gross_square_feet.str.extract('(\d+)', expand=True).fillna(0)

train_data.land_square_feet = train_data.land_square_feet.str.extract('(\d+)', expand=True).fillna(0)

this_data.gross_square_feet = this_data.gross_square_feet.str.extract('(\d+)', expand=True).fillna(0)

this_data.land_square_feet = this_data.land_square_feet.str.extract('(\d+)', expand=True).fillna(0)



train_data.land_square_feet = train_data.land_square_feet.astype(str).astype(int)

train_data.gross_square_feet = train_data.gross_square_feet.astype(str).astype(int)

this_data.land_square_feet = this_data.land_square_feet.astype(str).astype(int)

this_data.gross_square_feet = this_data.gross_square_feet.astype(str).astype(int)



this_data.sale_date = pd.to_timedelta(this_data["sale_date"])

train_data.sale_date =  pd.to_timedelta(train_data["sale_date"])

this_data.sale_date = this_data.sale_date.dt.total_seconds()

train_data.sale_date = train_data.sale_date.dt.total_seconds()



train_data.info()
x = train_data

this = this_data



train_objs_num = len(x)

dataset = pd.concat(objs=[x, this_data], axis=0, sort=False)

dataset = pd.get_dummies(dataset)

x = dataset[:train_objs_num]

this = dataset[train_objs_num:]



x
x.info()
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



# Define model. Specify a number for random_state to ensure same results each run

model = RandomForestRegressor(random_state=0)



# Fit model

model.fit(x, y)



predict = model.predict(this)

print(predict)





#forest_model = RandomForestRegressor(random_state=1)

#forest_model.fit(x, y)

#melb_preds = forest_model.predict(this)

# print(mean_absolute_error(val_y, melb_preds))
from sklearn.metrics import mean_absolute_error

from sklearn.tree import DecisionTreeRegressor



def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)



for max_leaf_nodes in [10, 100, 1000, 10000, 100000, 1000000]:

    my_mae = get_mae(max_leaf_nodes,  x, this, y, predict)

    print("Max leaf nodes: %d  \t\t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
# submission = exemplo_data.query("sale_id <= 47842")

# resto = exemplo_data.query("sale_id > 47842")

exemplo_data.sale_price = predict

# bigdata = pd.concat([submission, resto], ignore_index=True)

# bigdata.to_csv("submission.csv", index = False)

exemplo_data.to_csv("submission.csv", index = False)
# from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype

# from sklearn.tree import export_graphviz

# import IPython

# import graphviz

# from sklearn.ensemble import forest



# def train_cats(df):

#     """Change any columns of strings in a panda's dataframe to a column of

#     categorical values. This applies the changes inplace.

#     Parameters:

#     -----------

#     df: A pandas dataframe. Any columns of strings will be changed to

#         categorical values.

#     Examples:

#     ---------

#     >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})

#     >>> df

#        col1 col2

#     0     1    a

#     1     2    b

#     2     3    a

#     note the type of col2 is string

#     >>> train_cats(df)

#     >>> df

#        col1 col2

#     0     1    a

#     1     2    b

#     2     3    a

#     now the type of col2 is category

#     """

#     for n,c in df.items():

#         if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()



# def proc_df(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None,

#             preproc_fn=None, max_n_cat=None, subset=None, mapper=None):

#     """ proc_df takes a data frame df and splits off the response variable, and

#     changes the df into an entirely numeric dataframe. For each column of df 

#     which is not in skip_flds nor in ignore_flds, na values are replaced by the

#     median value of the column.

#     Parameters:

#     -----------

#     df: The data frame you wish to process.

#     y_fld: The name of the response variable

#     skip_flds: A list of fields that dropped from df.

#     ignore_flds: A list of fields that are ignored during processing.

#     do_scale: Standardizes each column in df. Takes Boolean Values(True,False)

#     na_dict: a dictionary of na columns to add. Na columns are also added if there

#         are any missing values.

#     preproc_fn: A function that gets applied to df.

#     max_n_cat: The maximum number of categories to break into dummy values, instead

#         of integer codes.

#     subset: Takes a random subset of size subset from df.

#     mapper: If do_scale is set as True, the mapper variable

#         calculates the values used for scaling of variables during training time (mean and standard deviation).

#     Returns:

#     --------

#     [x, y, nas, mapper(optional)]:

#         x: x is the transformed version of df. x will not have the response variable

#             and is entirely numeric.

#         y: y is the response variable

#         nas: returns a dictionary of which nas it created, and the associated median.

#         mapper: A DataFrameMapper which stores the mean and standard deviation of the corresponding continuous

#         variables which is then used for scaling of during test-time.

#     Examples:

#     ---------

#     >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})

#     >>> df

#        col1 col2

#     0     1    a

#     1     2    b

#     2     3    a

#     note the type of col2 is string

#     >>> train_cats(df)

#     >>> df

#        col1 col2

#     0     1    a

#     1     2    b

#     2     3    a

#     now the type of col2 is category { a : 1, b : 2}

#     >>> x, y, nas = proc_df(df, 'col1')

#     >>> x

#        col2

#     0     1

#     1     2

#     2     1

#     >>> data = DataFrame(pet=["cat", "dog", "dog", "fish", "cat", "dog", "cat", "fish"],

#                  children=[4., 6, 3, 3, 2, 3, 5, 4],

#                  salary=[90, 24, 44, 27, 32, 59, 36, 27])

#     >>> mapper = DataFrameMapper([(:pet, LabelBinarizer()),

#                           ([:children], StandardScaler())])

#     >>>round(fit_transform!(mapper, copy(data)), 2)

#     8x4 Array{Float64,2}:

#     1.0  0.0  0.0   0.21

#     0.0  1.0  0.0   1.88

#     0.0  1.0  0.0  -0.63

#     0.0  0.0  1.0  -0.63

#     1.0  0.0  0.0  -1.46

#     0.0  1.0  0.0  -0.63

#     1.0  0.0  0.0   1.04

#     0.0  0.0  1.0   0.21

#     """

#     if not ignore_flds: ignore_flds=[]

#     if not skip_flds: skip_flds=[]

#     if subset: df = get_sample(df,subset)

#     else: df = df.copy()

#     ignored_flds = df.loc[:, ignore_flds]

#     df.drop(ignore_flds, axis=1, inplace=True)

#     if preproc_fn: preproc_fn(df)

#     if y_fld is None: y = None

#     else:

#         if not is_numeric_dtype(df[y_fld]): df[y_fld] = pd.Categorical(df[y_fld]).codes

#         y = df[y_fld].values

#         skip_flds += [y_fld]

#     df.drop(skip_flds, axis=1, inplace=True)



#     if na_dict is None: na_dict = {}

#     else: na_dict = na_dict.copy()

#     na_dict_initial = na_dict.copy()

#     for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)

#     if len(na_dict_initial.keys()) > 0:

#         df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)

#     if do_scale: mapper = scale_vars(df, mapper)

#     for n,c in df.items(): numericalize(df, c, n, max_n_cat)

#     df = pd.get_dummies(df, dummy_na=True)

#     df = pd.concat([ignored_flds, df], axis=1)

#     res = [df, y, na_dict]

#     if do_scale: res = res + [mapper]

#     return res

  

# def fix_missing(df, col, name, na_dict):

#     """ Fill missing data in a column of df with the median, and add a {name}_na column

#     which specifies if the data was missing.

#     Parameters:

#     -----------

#     df: The data frame that will be changed.

#     col: The column of data to fix by filling in missing data.

#     name: The name of the new filled column in df.

#     na_dict: A dictionary of values to create na's of and the value to insert. If

#         name is not a key of na_dict the median will fill any missing data. Also

#         if name is not a key of na_dict and there is no missing data in col, then

#         no {name}_na column is not created.

#     Examples:

#     ---------

#     >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})

#     >>> df

#        col1 col2

#     0     1    5

#     1   nan    2

#     2     3    2

#     >>> fix_missing(df, df['col1'], 'col1', {})

#     >>> df

#        col1 col2 col1_na

#     0     1    5   False

#     1     2    2    True

#     2     3    2   False

#     >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})

#     >>> df

#        col1 col2

#     0     1    5

#     1   nan    2

#     2     3    2

#     >>> fix_missing(df, df['col2'], 'col2', {})

#     >>> df

#        col1 col2

#     0     1    5

#     1   nan    2

#     2     3    2

#     >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})

#     >>> df

#        col1 col2

#     0     1    5

#     1   nan    2

#     2     3    2

#     >>> fix_missing(df, df['col1'], 'col1', {'col1' : 500})

#     >>> df

#        col1 col2 col1_na

#     0     1    5   False

#     1   500    2    True

#     2     3    2   False

#     """

#     if is_numeric_dtype(col):

#         if pd.isnull(col).sum() or (name in na_dict):

#             df[name+'_na'] = pd.isnull(col)

#             filler = na_dict[name] if name in na_dict else col.median()

#             df[name] = col.fillna(filler)

#             na_dict[name] = filler

#     return na_dict

  

# def numericalize(df, col, name, max_n_cat):

#     """ Changes the column col from a categorical type to it's integer codes.

#     Parameters:

#     -----------

#     df: A pandas dataframe. df[name] will be filled with the integer codes from

#         col.

#     col: The column you wish to change into the categories.

#     name: The column name you wish to insert into df. This column will hold the

#         integer codes.

#     max_n_cat: If col has more categories than max_n_cat it will not change the

#         it to its integer codes. If max_n_cat is None, then col will always be

#         converted.

#     Examples:

#     ---------

#     >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})

#     >>> df

#        col1 col2

#     0     1    a

#     1     2    b

#     2     3    a

#     note the type of col2 is string

#     >>> train_cats(df)

#     >>> df

#        col1 col2

#     0     1    a

#     1     2    b

#     2     3    a

#     now the type of col2 is category { a : 1, b : 2}

#     >>> numericalize(df, df['col2'], 'col3', None)

#        col1 col2 col3

#     0     1    a    1

#     1     2    b    2

#     2     3    a    1

#     """

#     if not is_numeric_dtype(col) and ( max_n_cat is None or len(col.cat.categories)>max_n_cat):

#         df[name] = pd.Categorical(col).codes+1

        

# def get_sample(df,n):

#     """ Gets a random sample of n rows from df, without replacement.

#     Parameters:

#     -----------

#     df: A pandas data frame, that you wish to sample from.

#     n: The number of rows you wish to sample.

#     Returns:

#     --------

#     return value: A random sample of n rows of df.

#     Examples:

#     ---------

#     >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})

#     >>> df

#        col1 col2

#     0     1    a

#     1     2    b

#     2     3    a

#     >>> get_sample(df, 2)

#        col1 col2

#     1     2    b

#     2     3    a

#     """

#     idxs = sorted(np.random.permutation(len(df))[:n])

#     return df.iloc[idxs].copy()



# def draw_tree(t, df, size=10, ratio=0.6, precision=0):

#     """ Draws a representation of a random forest in IPython.

#     Parameters:

#     -----------

#     t: The tree you wish to draw

#     df: The data used to train the tree. This is used to get the names of the features.

#     """

#     s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True,

#                       special_characters=True, rotate=True, precision=precision)

#     IPython.display.display(graphviz.Source(re.sub('Tree {',

#        f'Tree {{ size={size}; ratio={ratio}', s)))

    

# def set_rf_samples(n):

#     """ Changes Scikit learn's random forests to give each tree a random sample of

#     n random rows.

#     """

#     forest._generate_sample_indices = (lambda rs, n_samples:

#         forest.check_random_state(rs).randint(0, n_samples, n))

# def reset_rf_samples():

#     """ Undoes the changes produced by set_rf_samples.

#     """

#     forest._generate_sample_indices = (lambda rs, n_samples:

#         forest.check_random_state(rs).randint(0, n_samples, n_samples))