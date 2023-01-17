# basic imports

import pandas as pd

import numpy as np



# plotting libraries and magics

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline



# garbage collector

import gc



# preprocessing

from sklearn.preprocessing import LabelEncoder



# modeling requirements

from sklearn.model_selection import train_test_split # to split the data into train and validation sets

from sklearn.metrics import r2_score # eval metric for this competetion



from sklearn.ensemble import RandomForestRegressor



# enabling multiple outputs for Jupyter cells

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity='all'
# read the data

train = pd.read_csv('../input/mercedes-benz-greener-manufacturing/train.csv', index_col='ID')

test = pd.read_csv('../input/mercedes-benz-greener-manufacturing/test.csv', index_col='ID')

sub = pd.read_csv('../input/mercedes-benz-greener-manufacturing/sample_submission.csv')
# train.head(3)

# test.head(3)

# sub.head(3)
# let's first get our target extracted

y = train['y']

# y.head(3)



# drop the target from the train set 

train.drop('y', axis=1, inplace=True)
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return(np.transpose(tt))
def get_num_cat_cols(df):

    """

    Returns two lists, one for categorical variables and one for numeric variables

    """

    cat_vars = [col for col in df.columns if df[col].dtype == 'object']

    num_vars = [col for col in df.columns if df[col].dtype != 'object']

    

    return cat_vars, num_vars
# # let's see what's the missing values look like

# missing_data(train)

# # wow! No missing values!
# let's see how many categorical and how many numeric variables we have

cat_vars, num_vars = get_num_cat_cols(train)

print('Categorical variables: ', cat_vars)

print('Numeric variables: ', num_vars)
# let's see what the numeric data looks like

train[num_vars].head()



# we could add a new feature which is the count of 1's in the num_vars per row

train['1_count'] = train[num_vars].sum(axis=1)

test['1_count'] = test[num_vars].sum(axis=1)

train.head(3)
# let's also get the percentage of 1's to total numeric columns

train['1_count_percent'] = (train[num_vars].sum(axis=1) * 100) / len(num_vars)

test['1_count_percent'] = (test[num_vars].sum(axis=1) * 100) / len(num_vars)



train.head()
# let's see what kind of data is present in the columns

cols_with_less_data = [c for c in num_vars if train[c].nunique()<2]

# for c in cols_with_less_data:

#     train[c].unique()



# these rows only have zeroes, that's not helping us gain any information, let's drop them - that reduces 12 columns.

train.drop(cols_with_less_data, axis=1, inplace=True)

test.drop(cols_with_less_data, axis=1, inplace=True)

# train.head(3)
gc.collect()
# let's check the categorical variables

[print('There are {1} unique values in {0} column'.format(c, train[c].nunique())) for c in cat_vars];
# X4 has only 4 values. let's see what they are

train.X4.unique() # could be label encoded
# X3 has only 7 values. let's see what they are

train.X3.unique() # could be label encoded
# let's check X6

train.X6.unique() # could be label encoded
# let's check X2

train.X2.unique() # could be label encoded
# let's get the num_cols once again since we removed some of them

cat_vars, num_vars = get_num_cat_cols(train)
# # help(sns.heatmap)

# plt.figure(figsize=(15, 5))

# sns.heatmap(train_without_trgt[num_vars].corr());
def label_encode_columns(df):

    """

    Given a dataframe, this will label encode all the categorical columns

    """

    cat_cols, _ = get_num_cat_cols(df)

    le = LabelEncoder()

    for c in cat_cols:

        le.fit(df[c])

        df[c] = le.transform(df[c])

    

    return df



# posibly, implement inplace=True functionality too
train = label_encode_columns(train)

test = label_encode_columns(test)



train[cat_vars].head()

test[cat_vars].head()
# let's split the train and test set

train_x, val_x, train_y, val_y = train_test_split(train, y, test_size=0.3)
important_features = ['X314','X315','X119','X118','X263','X136','X29','X279','X5','X232','X76','X54','X189','X47','X104','X8','1_count_percent','1_count','X2',

                       'X3','X275','X65','X26','X6','X127','X1','X267','X311','X283','X0','X77','X341','X287','X13','X241','X46','X162','X117','X82','X105']
# params = {n_estimators: [100, 200, 400, 500],

#           max_depth: [3, 4, 5],

          

#          }



# for now, let's run it with default param values. We'll tune things later.

reg = RandomForestRegressor(n_estimators = 500, 

                            max_depth = 5,

                            random_state=42)

reg.fit(train_x[important_features], train_y);



val_pred = reg.predict(val_x[important_features])



# now, let's check the R2 score

r2_score(val_y, val_pred)
# # let's see what features are important and what are junk - we'll get rid of the junk to maybe make the performance better

# plt.figure(figsize=(20, 10))

# feat_importances = pd.Series(reg.feature_importances_, index=train.columns);

# feat_importances.nlargest(40).plot(kind='barh');



# important_features = list(feat_importances.nlargest(40).index)
sub['y'] = reg.predict(test[important_features])

sub.to_csv('quick_and_dirty_Random_forest.csv', index=False);