# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from fastai.tabular import *

from fastai.imports import *
path=Path('/kaggle/input/bigquery-geotab-intersection-congestion/')
train=pd.read_csv(path/'train.csv');train.head()
test=pd.read_csv(path/'test.csv');test.head()
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype

def apply_cats(df, trn):

    

    """Changes any columns of strings in df into categorical variables using trn as

    a template for the category codes.

    Parameters:

    -----------

    df: A pandas dataframe. Any columns of strings will be changed to

        categorical values. The category codes are determined by trn.

    trn: A pandas dataframe. When creating a category for df, it looks up the

        what the category's code were in trn and makes those the category codes

        for df.

    Examples:

    ---------

    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})

    >>> df

       col1 col2

    0     1    a

    1     2    b

    2     3    a

    note the type of col2 is string

    >>> train_cats(df)

    >>> df

       col1 col2

    0     1    a

    1     2    b

    2     3    a

    now the type of col2 is category {a : 1, b : 2}

    >>> df2 = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['b', 'a', 'a']})

    >>> apply_cats(df2, df)

           col1 col2

        0     1    b

        1     2    a

        2     3    a

    now the type of col is category {a : 1, b : 2}

    """

    for n,c in df.items():

        if (n in trn.columns) and (trn[n].dtype.name=='category'):

            df[n] = c.astype('category').cat.as_ordered()

            df[n].cat.set_categories(trn[n].cat.categories, ordered=True, inplace=True)
def fix_missing(df, col, name, na_dict):

    

    """ Fill missing data in a column of df with the median, and add a {name}_na column

    which specifies if the data was missing.

    Parameters:

    -----------

    df: The data frame that will be changed.

    col: The column of data to fix by filling in missing data.

    name: The name of the new filled column in df.

    na_dict: A dictionary of values to create na's of and the value to insert. If

        name is not a key of na_dict the median will fill any missing data. Also

        if name is not a key of na_dict and there is no missing data in col, then

        no {name}_na column is not created.

    Examples:

    ---------

    >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})

    >>> df

       col1 col2

    0     1    5

    1   nan    2

    2     3    2

    >>> fix_missing(df, df['col1'], 'col1', {})

    >>> df

       col1 col2 col1_na

    0     1    5   False

    1     2    2    True

    2     3    2   False

    >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})

    >>> df

       col1 col2

    0     1    5

    1   nan    2

    2     3    2

    >>> fix_missing(df, df['col2'], 'col2', {})

    >>> df

       col1 col2

    0     1    5

    1   nan    2

    2     3    2

    >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})

    >>> df

       col1 col2

    0     1    5

    1   nan    2

    2     3    2

    >>> fix_missing(df, df['col1'], 'col1', {'col1' : 500})

    >>> df

       col1 col2 col1_na

    0     1    5   False

    1   500    2    True

    2     3    2   False

    """

    import pandas as pd

    if is_numeric_dtype(col):

        if pd.isnull(col).sum() or (name in na_dict):

            df[name+'_na'] = pd.isnull(col)

            filler = na_dict[name] if name in na_dict else col.median()

            df[name] = col.fillna(filler)

            na_dict[name] = filler

    return na_dict
def numericalize(df, col, name, max_n_cat):

    

    """ Changes the column col from a categorical type to it's integer codes.

    Parameters:

    -----------

    df: A pandas dataframe. df[name] will be filled with the integer codes from

        col.

    col: The column you wish to change into the categories.

    name: The column name you wish to insert into df. This column will hold the

        integer codes.

    max_n_cat: If col has more categories than max_n_cat it will not change the

        it to its integer codes. If max_n_cat is None, then col will always be

        converted.

    Examples:

    ---------

    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})

    >>> df

       col1 col2

    0     1    a

    1     2    b

    2     3    a

    note the type of col2 is string

    >>> train_cats(df)

    >>> df

       col1 col2

    0     1    a

    1     2    b

    2     3    a

    now the type of col2 is category { a : 1, b : 2}

    >>> numericalize(df, df['col2'], 'col3', None)

       col1 col2 col3

    0     1    a    1

    1     2    b    2

    2     3    a    1

    """

    if not is_numeric_dtype(col) and ( max_n_cat is None or len(col.cat.categories)>max_n_cat):

        df[name] = pd.Categorical(col).codes+1
def scale_vars(df, mapper):

    

    warnings.filterwarnings('ignore', category=sklearn.exceptions.DataConversionWarning)

    if mapper is None:

        map_f = [([n],StandardScaler()) for n in df.columns if is_numeric_dtype(df[n])]

        mapper = DataFrameMapper(map_f).fit(df)

    df[mapper.transformed_names_] = mapper.transform(df)

    return mapper
def proc_df(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None,

            preproc_fn=None, max_n_cat=None, subset=None, mapper=None):

    

    """ proc_df takes a data frame df and splits off the response variable, and

    changes the df into an entirely numeric dataframe. For each column of df 

    which is not in skip_flds nor in ignore_flds, na values are replaced by the

    median value of the column.

    Parameters:

    -----------

    df: The data frame you wish to process.

    y_fld: The name of the response variable

    skip_flds: A list of fields that dropped from df.

    ignore_flds: A list of fields that are ignored during processing.

    do_scale: Standardizes each column in df. Takes Boolean Values(True,False)

    na_dict: a dictionary of na columns to add. Na columns are also added if there

        are any missing values.

    preproc_fn: A function that gets applied to df.

    max_n_cat: The maximum number of categories to break into dummy values, instead

        of integer codes.

    subset: Takes a random subset of size subset from df.

    mapper: If do_scale is set as True, the mapper variable

        calculates the values used for scaling of variables during training time (mean and standard deviation).

    Returns:

    --------

    [x, y, nas, mapper(optional)]:

        x: x is the transformed version of df. x will not have the response variable

            and is entirely numeric.

        y: y is the response variable

        nas: returns a dictionary of which nas it created, and the associated median.

        mapper: A DataFrameMapper which stores the mean and standard deviation of the corresponding continuous

        variables which is then used for scaling of during test-time.

    Examples:

    ---------

    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})

    >>> df

       col1 col2

    0     1    a

    1     2    b

    2     3    a

    note the type of col2 is string

    >>> train_cats(df)

    >>> df

       col1 col2

    0     1    a

    1     2    b

    2     3    a

    now the type of col2 is category { a : 1, b : 2}

    >>> x, y, nas = proc_df(df, 'col1')

    >>> x

       col2

    0     1

    1     2

    2     1

    >>> data = DataFrame(pet=["cat", "dog", "dog", "fish", "cat", "dog", "cat", "fish"],

                 children=[4., 6, 3, 3, 2, 3, 5, 4],

                 salary=[90, 24, 44, 27, 32, 59, 36, 27])

    >>> mapper = DataFrameMapper([(:pet, LabelBinarizer()),

                          ([:children], StandardScaler())])

    >>>round(fit_transform!(mapper, copy(data)), 2)

    8x4 Array{Float64,2}:

    1.0  0.0  0.0   0.21

    0.0  1.0  0.0   1.88

    0.0  1.0  0.0  -0.63

    0.0  0.0  1.0  -0.63

    1.0  0.0  0.0  -1.46

    0.0  1.0  0.0  -0.63

    1.0  0.0  0.0   1.04

    0.0  0.0  1.0   0.21

    """

    if not ignore_flds: ignore_flds=[]

    if not skip_flds: skip_flds=[]

    if subset: df = get_sample(df,subset)

    else: df = df.copy()

    ignored_flds = df.loc[:, ignore_flds]

    df.drop(ignore_flds, axis=1, inplace=True)

    if preproc_fn: preproc_fn(df)

    if y_fld is None: y = None

    else:

        if not is_numeric_dtype(df[y_fld]): df[y_fld] = pd.Categorical(df[y_fld]).codes

        y = df[y_fld].values

        skip_flds += [y_fld]

    df.drop(skip_flds, axis=1, inplace=True)



    if na_dict is None: na_dict = {}

    else: na_dict = na_dict.copy()

    na_dict_initial = na_dict.copy()

    for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)

    if len(na_dict_initial.keys()) > 0:

        df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)

    if do_scale: mapper = scale_vars(df, mapper)

    for n,c in df.items(): numericalize(df, c, n, max_n_cat)

    df = pd.get_dummies(df, dummy_na=True)

    df = pd.concat([ignored_flds, df], axis=1)

    res = [df, y, na_dict]

    if do_scale: res = res + [mapper]

    return res
dep_var=['TotalTimeStopped_p20','TotalTimeStopped_p40','TotalTimeStopped_p50', 'TotalTimeStopped_p60', 'TotalTimeStopped_p80','TimeFromFirstStop_p20', 'TimeFromFirstStop_p40','TimeFromFirstStop_p50', 'TimeFromFirstStop_p60','TimeFromFirstStop_p80', 'DistanceToFirstStop_p20','DistanceToFirstStop_p40', 'DistanceToFirstStop_p50','DistanceToFirstStop_p60', 'DistanceToFirstStop_p80']

cat_names=['EntryStreetName','ExitStreetName', 'EntryHeading', 'ExitHeading', 'City','Path']

columns=train.columns;

cont_names=[];

cont_names=[x for x in columns if x not in cat_names];

for x in dep_var:cont_names.remove(x);

len(cont_names)
procs = [FillMissing, Categorify, Normalize];

test_tab = TabularList.from_df(test, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
data = (TabularList.from_df(train, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)

        

        .split_by_rand_pct(valid_pct=0.2,seed=40)

        

        .label_from_df(cols=dep_var,label_cls=FloatList)

        

        .add_test(test_tab)



        .databunch()

    )
data.show_batch()
learn = tabular_learner(data, layers=[20,10], metrics=rmse)

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, 1e-2)
target={

0:'TotalTimeStopped_p20',

1:'TotalTimeStopped_p50',

2:'TotalTimeStopped_p80',

3:'DistanceToFirstStop_p20',

4:'DistanceToFirstStop_p50',

5:'DistanceToFirstStop_p80',

}
indx=[]

for key in target.keys():

  xx=dep_var.index(target[key])

  indx.append(xx)

print(indx)
preds,_=learn.get_preds(ds_type=DatasetType.Test)
prediction_total=[]



for pred in range(len(preds)):

    preds_array=[x.item() for x in preds[pred]]

    for id in indx:prediction_total.append(preds_array[id])





len(prediction_total)==(len(indx)*len(test))
# Read submission file

submission=pd.read_csv(path/'sample_submission.csv');submission.head()
submission['Target']=prediction_total

submission.to_csv("DL_test_1.csv", index = False)