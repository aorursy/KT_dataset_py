### Here are the packages used in this notebook



import numpy as np

import pandas as pd

import math

import datetime



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns             # https://seaborn.pydata.org/index.html

from wordcloud import WordCloud   # https://pypi.org/project/wordcloud/



import pandasql                    # https://github.com/yhat/pandasql

from pandasql import sqldf

pysqldf = lambda q: sqldf(q, globals())



from sklearn.preprocessing import MultiLabelBinarizer   #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html

import ast                          # https://docs.python.org/3/library/ast.html



from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.linear_model import LinearRegression,Lasso

from sklearn.model_selection import KFold, cross_val_score, train_test_split,GridSearchCV

from sklearn.metrics import mean_squared_error, mean_squared_log_error

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.feature_selection import SelectKBest,chi2

import xgboost as xgb

import lightgbm as lgb

from catboost import CatBoostRegressor

from xgboost.sklearn import XGBRegressor

from xgboost import plot_importance

from types import FunctionType

from fastai.imports import *





import os

print(os.listdir("../input"))



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
# First let's read the files in

# I like to do all my transformations on one file to prevent issues so I will create a file with both and tag as train/test

# Since I am doing log and normalize transformations across both files, results may vary if done only on the train file.

# release_date is in a horrible format - parse it on the way in 



train = pd.read_csv("../input/train.csv",parse_dates=["release_date"])

train['file']='train'

train = train.set_index('id')

print('train')

print(train.head(2))



test = pd.read_csv("../input/test.csv",parse_dates=["release_date"])

test['file']='test'

test = test.set_index('id')

print('test')

print(test.head(2))



both = pd.concat([train, test], sort=False)



print('both')

print(both.head(2))

      

submission = pd.read_csv("../input/sample_submission.csv")

print('train data types')  

train.dtypes

print('test data types')  

test.dtypes

print('both data types')  

both.dtypes

both.info()
both.describe(include='all')
# Show me the nulls across both files - what am I dealing with here?

print(both.isnull().sum())
# Unsure if we are concerned WHICH collection, but we will assume that they wouldn't make another in the series if the first one bombed

# Using pandasql, one of my favorites though perforance can suffer. Dataset is small so let's give it a go

belongs_to_q = """

select 

count(*) as count, substr(belongs_to_collection, instr(belongs_to_collection,'name'), (instr(belongs_to_collection,'name')+40)) 

from both 

group by substr(belongs_to_collection, instr(belongs_to_collection,'name'), (instr(belongs_to_collection,'name')+40))

order by count(*) desc 

"""

belongs_to_o = pysqldf(belongs_to_q)

print(belongs_to_o)



# We see James Bond is the big leader. 

# I do see many collections of 1, meaning others in the collection are either spelled differently or not in this dataset.

# I would error on the side of just one-hotting the existence of a collection
# Since we are only interested in the existence of a name, let's one-hot it into a new column part_of_a_collection

both['part_of_a_collection'] = np.where(both['belongs_to_collection'].isna(), 0, 1)



collection_counts_q = """

select 

part_of_a_collection, count(*) as count

from both

group by part_of_a_collection

"""

collection_counts_o = pysqldf(collection_counts_q)

print(collection_counts_o)
# What is the distribution of the budget?



# Viz with seaborn. It's visually pleasing

sns.set(style="white", palette="muted", color_codes=True)



# Set up the matplotlib figure

f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)

sns.despine(left=True)



# Plot a simple histogram with binsize determined automatically

sns.distplot(both['budget'],kde=False, color="b", ax=axes[0, 0])



# Plot a kernel density estimate and rug plot

sns.distplot(both['budget'], hist=False, rug=True, color="r", ax=axes[0, 1])



# Plot a filled kernel density estimate

sns.distplot(both['budget'], hist=False, color="g", kde_kws={"shade": True}, ax=axes[1, 0])



# Plot a historgram and kernel density estimate

sns.distplot(both['budget'], color="m", ax=axes[1, 1])



plt.setp(axes, yticks=[])

plt.tight_layout()

# budget is very skewed. Makes sense. 

# I don't want to remove the outliers because those are the blockbusters (isn't that what we want to really know?)

# The evaluation also logs to minimize the effect of the blockbusters anyway

# How many of those budgets are zero (we know they aren't really zero...and are the same missing data)



budgets0_q = """

select 

 status, count(*) as count

from both

where budget = 0

group by status

"""

budgets0_o = pysqldf(budgets0_q)

print(budgets0_o)



# I though that the 0 values might correspond to films not yet released, that this isn't the case. This is a difficulty for sure
# I will try a Sigmoid function to help

def sigmoid(x):

    e = np.exp(1)

    y = 1/(1+e**(-x))

    return y

sigmoid_results = sigmoid(both['budget'])

sigmoid_results.describe()



# That doesn't look good
# try log function

budget_log = np.log(both['budget'])

budget_log.describe



# divide by zero...nope
# Next up, Log + 1

budget_log1 = np.log(both['budget'] + 1)

budget_log1.describe()



# better - include a normalize

def normalize(column):

    upper = column.max()

    lower = column.min()

    y = (column - lower)/(upper-lower)

    return y



budget_log1_normalized = normalize(np.log(both['budget'] + 1))

budget_log1_normalized.describe()



# looks good

# set a new column as budget_log1_norm

both['budget_log1_norm'] = normalize(np.log(both['budget'] + 1))
genres_q = """

select 

count(*) as count, substr(genres, instr(genres,'name'), (instr(genres,'name')+40)) 

from both 

group by substr(genres, instr(genres,'name'), (instr(genres,'name')+40))

order by count(*) desc 

"""

genres_o = pysqldf(genres_q)

print(genres_o)



# I see that a movie can have multiple genres.

# Do I simply 1-hot out the results or dig deeper into the combinations?
# While I would like to explore the combinations, let's start simply with 1-hots.

# Compliments to https://www.kaggle.com/liviuasnash/predict-movies-step-by-step/notebook

both['genres'] = both['genres'].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x))

both['genres']
# The EDA and feature engineering for the dict columns seems similar. Create a few basic common utilities

def print_few_values(col_name):

    print("Sample values for", col_name)

    both[col_name].head(5).apply(lambda x: print(x))

    

def dictionary_sizes(col_name):

    return (both[col_name].apply(lambda x: len(x)).value_counts())



def print_dictionary_sizes(col_name):

    print("\n===================================================")

    print("Distribution of sizes for", col_name)

    print(dictionary_sizes(col_name))

    

# returns a list of tuples of names for a given row of a column

def dict_name_list(d, name="name"):

    return ([i[name] for i in d] if d != {} else [])



# returns a list of tuples of the (id,name) pairs for a given row of a column

def dict_id_name_list(d, name="name"):

    return ([(i["id"],i[name]) for i in d] if d != {} else [])



# returns a list of names for a given column

def col_name_list(col_name, name="name"):

    # Get a list of lists of names

    name_list_list = list(both[col_name].apply(lambda x: dict_name_list(x, name)).values)

    # Merge into 1 list

    return ([i for name_list in name_list_list for i in name_list])



# returns a list of tuples of the (id,name) pairs for a given column

def col_id_name_list(col_name, name="name"):

    # Get a list of lists of (id,name) tuples

    tuple_list_list = list(both[col_name].apply(lambda x: dict_id_name_list(x, name)).values)

    # Merge into 1 list

    return ([i for tuple_list in tuple_list_list for i in tuple_list])



def get_names_counter(col_name, name="name"):

    name_list = col_name_list(col_name, name)

    return (collections.Counter(name_list))

    

def print_top_names(col_name, name="name"):

    print("\n===================================================")

    print("Top {0}s for {1}".format(name, col_name))

    c = get_names_counter(col_name, name)

    print(c.most_common(20))

    

def EDA_dict(col_name):

    print_few_values(col_name)

    print_dictionary_sizes(col_name)

    print_top_names(col_name)

    

def add_dict_size_column(col_name):

    both[col_name + "_size"] = both[col_name].apply(lambda x: len(x) if x != {} else 0)



def add_dict_id_column(col_name):

    c = col_name + "_id"

    both[c] = both[col_name].apply(lambda x: x[0]["id"] if x != {} else 0)

    both[c] = both[c].astype("category")



# for each of the top values in the dictionary, add an column indicating if the row belongs to it

def add_dict_indicator_columns(col_name):

    c = get_names_counter(col_name)

    top_names = [x[0] for x in c.most_common(20)]

    for name in top_names:

        both[col_name + "_" + name] = both[col_name].apply(lambda x: name in dict_name_list(x))

        

def drop_column(col_name):

    both.drop([col_name], axis=1, inplace=True)

    

def feature_engineer_dict(col_name):

    add_dict_size_column(col_name)

    max_size = dictionary_sizes(col_name).index.max()

    if max_size == 1:

        add_dict_id_column(col_name)

    else:

        add_dict_indicator_columns(col_name)

    drop_column(col_name)

    

def encode_column(col_name):

    lbl = LabelEncoder()

    lbl.fit(list(both[col_name].values)) 

    both[col_name] = lbl.transform(list(both[col_name].values))

    

col_name="genres"

EDA_dict(col_name)



feature_engineer_dict(col_name)
both.columns
# get rid of the sames in science fiction and tv movie...not helpful

both.columns = ['belongs_to_collection', 'budget', 'homepage', 'imdb_id',

       'original_language', 'original_title', 'overview', 'popularity',

       'poster_path', 'production_companies', 'production_countries',

       'release_date', 'runtime', 'spoken_languages', 'status', 'tagline',

       'title', 'Keywords', 'cast', 'crew', 'revenue', 'file',

       'part_of_a_collection', 'budget_log1_norm', 'genres_size',

       'genres_Drama', 'genres_Comedy', 'genres_Thriller', 'genres_Action',

       'genres_Romance', 'genres_Adventure', 'genres_Crime',

       'genres_Science_Fiction', 'genres_Horror', 'genres_Family',

       'genres_Fantasy', 'genres_Mystery', 'genres_Animation',

       'genres_History', 'genres_Music', 'genres_War', 'genres_Documentary',

       'genres_Western', 'genres_Foreign', 'genres_TV_Movie']
# Convert T/F to 1/0 for future matching

both.genres_Drama = both.genres_Drama.astype(int)

both.genres_Comedy = both.genres_Comedy.astype(int)

both.genres_Thriller = both.genres_Thriller.astype(int)

both.genres_Action = both.genres_Action.astype(int)

both.genres_Romance = both.genres_Romance.astype(int)

both.genres_Adventure = both.genres_Adventure.astype(int)

both.genres_Crime = both.genres_Crime.astype(int)

both.genres_Science_Fiction = both.genres_Science_Fiction.astype(int)

both.genres_Horror = both.genres_Horror.astype(int)

both.genres_Family = both.genres_Family.astype(int)

both.genres_Fantasy = both.genres_Fantasy.astype(int)

both.genres_Mystery = both.genres_Mystery.astype(int)

both.genres_Animation = both.genres_Animation.astype(int)

both.genres_History = both.genres_History.astype(int)

both.genres_Music = both.genres_Music.astype(int)

both.genres_War = both.genres_War.astype(int)

both.genres_Documentary = both.genres_Documentary.astype(int)

both.genres_Western = both.genres_Western.astype(int)

both.genres_Foreign = both.genres_Foreign.astype(int)

both.genres_TV_Movie = both.genres_TV_Movie.astype(int)

both.head(10)
# 20 top revenue genre combos

g_rev = """

with list as 

(select avg(revenue) as avg_revenue

,    genres_Drama, genres_Comedy, genres_Thriller, genres_Action,

       genres_Romance, genres_Adventure, genres_Crime,

       genres_Science_Fiction, genres_Horror, genres_Family,

       genres_Fantasy, genres_Mystery, genres_Animation,

       genres_History, genres_Music, genres_War, genres_Documentary,

       genres_Western, genres_Foreign, genres_TV_Movie

       from both 

       group by

       genres_Drama, genres_Comedy, genres_Thriller, genres_Action,

       genres_Romance, genres_Adventure, genres_Crime,

       genres_Science_Fiction, genres_Horror, genres_Family,

       genres_Fantasy, genres_Mystery, genres_Animation,

       genres_History, genres_Music, genres_War, genres_Documentary,

       genres_Western, genres_Foreign, genres_TV_Movie

  )

  select row_number()  over (

       order by avg_revenue asc) as row_number

  , genres_Drama, genres_Comedy, genres_Thriller, genres_Action,

       genres_Romance, genres_Adventure, genres_Crime,

       genres_Science_Fiction, genres_Horror, genres_Family,

       genres_Fantasy, genres_Mystery, genres_Animation,

       genres_History, genres_Music, genres_War, genres_Documentary,

       genres_Western, genres_Foreign, genres_TV_Movie

       from list

       

       

"""

genre_revenue = pysqldf(g_rev)

genre_revenue.genres_Drama = genre_revenue.genres_Drama.astype(int)

genre_revenue.genres_Comedy = genre_revenue.genres_Comedy.astype(int)

genre_revenue.genres_Thriller = genre_revenue.genres_Thriller.astype(int)

genre_revenue.genres_Action = genre_revenue.genres_Action.astype(int)

genre_revenue.genres_Romance = genre_revenue.genres_Romance.astype(int)

genre_revenue.genres_Adventure = genre_revenue.genres_Adventure.astype(int)

genre_revenue.genres_Crime = genre_revenue.genres_Crime.astype(int)

genre_revenue.genres_Science_Fiction = genre_revenue.genres_Science_Fiction.astype(int)

genre_revenue.genres_Horror = genre_revenue.genres_Horror.astype(int)

genre_revenue.genres_Family = genre_revenue.genres_Family.astype(int)

genre_revenue.genres_Fantasy = genre_revenue.genres_Fantasy.astype(int)

genre_revenue.genres_Mystery = genre_revenue.genres_Mystery.astype(int)

genre_revenue.genres_Animation = genre_revenue.genres_Animation.astype(int)

genre_revenue.genres_History = genre_revenue.genres_History.astype(int)

genre_revenue.genres_Music = genre_revenue.genres_Music.astype(int)

genre_revenue.genres_War = genre_revenue.genres_War.astype(int)

genre_revenue.genres_Documentary = genre_revenue.genres_Documentary.astype(int)

genre_revenue.genres_Western = genre_revenue.genres_Western.astype(int)

genre_revenue.genres_Foreign = genre_revenue.genres_Foreign.astype(int)

genre_revenue.genres_TV_Movie = genre_revenue.genres_TV_Movie.astype(int)

genre_revenue.tail(20)
olang_q = """

select 

count(*) as count, original_language

from both 

group by original_language

order by count(*) desc 

"""

olang_o = pysqldf(olang_q)

print(olang_o)



# overwhelmingly English. I'm not sure if this is enough variation to be useful.
both['is_English'] = 0 

both.loc[ both['original_language'] == "en" ,"is_English"] = 1
# What is the distribution of the popularity?



# Viz with seaborn. It's visually pleasing

sns.set(style="white", palette="muted", color_codes=True)



# Set up the matplotlib figure

f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)

sns.despine(left=True)



# Plot a simple histogram with binsize determined automatically

sns.distplot(both['popularity'],kde=False, color="b", ax=axes[0, 0])



# Plot a kernel density estimate and rug plot

sns.distplot(both['popularity'], hist=False, rug=True, color="r", ax=axes[0, 1])



# Plot a filled kernel density estimate

sns.distplot(both['popularity'], hist=False, color="g", kde_kws={"shade": True}, ax=axes[1, 0])



# Plot a historgram and kernel density estimate

sns.distplot(both['popularity'], color="m", ax=axes[1, 1])



plt.setp(axes, yticks=[])

plt.tight_layout()
# commenting out for efficiency



# popularity in relation to revenue

#pop_rev = sns.swarmplot(x='popularity', y='revenue', data=both)

#pop_rev



# More popular = More revenue? Shocker!

# It is skewed so let's log1 normalize it to a new column popularity_log1_norm



both['popularity_log1_norm'] = normalize(np.log(both['popularity'] + 1))



# 1 test row without a date - that will probably need to be fixed or removed

# 3829

# Thinking about it, do not remove.  Then the shapes will not align. 

# Looking up the information:

# https://www.imdb.com/title/tt0210130/?ref_=nv_sr_1?ref_=nv_sr_1

# 20 March 2001



#both['release_date']=both['release_date'].fillna('3/20/2001')



# I want to break the date parts out..... 

both['release_date']

both['release_date'] = pd.to_datetime(both['release_date'], format='%m/%d/%y')

both['release_year'] = both.release_date.dt.year.fillna(2001)

both['release_month'] = both.release_date.dt.month.fillna(3)

both['release_day'] = both.release_date.dt.day.fillna(20)

both['release_day_of_week'] = both.release_date.dt.dayofweek.fillna(1)

both['release_quarter'] = both.release_date.dt.quarter.fillna(1) 
# This is a check to be sure I don't have any null date fields



print(both['release_quarter'].isnull().sum())

# Runtime and Revenue?

# Replace NaN with zeroes



both['runtime'] = both['runtime'].fillna





both.columns
# Create a dataframe with only the columns I want and move revenue to the end



both_final = both[['file', 

       'part_of_a_collection', 'budget_log1_norm', 'genres_size',

       'genres_Drama', 'genres_Comedy', 'genres_Thriller', 'genres_Action',

       'genres_Romance', 'genres_Adventure', 'genres_Crime',

       'genres_Science_Fiction', 'genres_Horror', 'genres_Family',

       'genres_Fantasy', 'genres_Mystery', 'genres_Animation',

       'genres_History', 'genres_Music', 'genres_War', 'genres_Documentary',

       'genres_Western', 'genres_Foreign', 'genres_TV_Movie',

       'popularity_log1_norm', 'release_year', 'release_month', 'release_day',

       'release_day_of_week', 'release_quarter','is_English','revenue']]







both_final
# Time to put split the data into train and test files

# Then drop the file column, we don't need that anymore



train_final_1 = both_final[(both_final['file'] == "train")]

train_final_1 = train_final_1.drop(columns=['file'])



test_final_1 = both_final[(both_final['file'] == "test")]

test_final_1 = test_final_1.drop(columns=['file'])



y_train = train_final_1[['revenue']]

y_train
# Do I have nulls in there??

# Yes, test revenue

test_final_1.sum(), test_final_1.min(), test_final_1.max()

test_final_1['revenue'] = 1000

test_final_1.sum(), test_final_1.min(), test_final_1.max()
y_train = y_train.values
# 1.) Remove table meta data, column names etc. → Just use values for prediction.

X_train = train_final_1.values

#y_train = y_train.values



X_test  = test_final_1.values



# Update:) Scale



X_scaler = StandardScaler()

X_train  = X_scaler.fit_transform(X_train)

X_test   = X_scaler.transform(X_test)

y_train  = np.log(y_train)

y_scaler = MinMaxScaler((0,1))
# Do I have nulls in there??



X_train.sum(), X_train.min(), X_train.max()

y_train.sum(), y_train.min(), y_train.max()

X_test.sum(), X_test.min(), X_test.max()

y_train  = y_scaler.fit_transform(y_train).ravel() # transform and convert column-vector y to a 1d array



# Update:) Create Validation Split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=56)
# The data is prepped, now to fit and predict.

# Nice kernel with more details : https://www.kaggle.com/alexandermelde/code-template-for-simple-regression-prediction/data



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import learning_curve

from sklearn.kernel_ridge import KernelRidge

from sklearn.svm import SVR

from sklearn import tree

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import SGDRegressor

import lightgbm as lgb



kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,

                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],

                              "gamma": np.logspace(-2, 2, 5)})



svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,

                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],

                               "gamma": np.logspace(-2, 2, 5)})



tr = tree.DecisionTreeRegressor()



est = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,

     max_depth=1, random_state=0, loss='ls').fit(X_train, y_train)



sgd1 = SGDRegressor(alpha=0.0001, fit_intercept=False, max_iter=100, early_stopping=True, n_iter_no_change=5, epsilon=0.1,

                   loss='squared_loss')



LGB = lgb.LGBMRegressor(n_estimators=10000, 

                             objective='regression', 

                             metric='rmse',

                             max_depth = 5,

                             num_leaves=30, 

                             min_child_samples=100,

                             learning_rate=0.01,

                             boosting = 'gbdt',

                             min_data_in_leaf= 10,

                             feature_fraction = 0.9,

                             bagging_freq = 1,

                             bagging_fraction = 0.9,

                             importance_type='gain',

                             lambda_l1 = 0.2,

                             bagging_seed=2729, 

                             subsample=.8, 

                             colsample_bytree=.9,

                             use_best_model=True)



# 2.) Calculate the coefficients of the linear regression / "Train"

#reg     = KNeighborsRegressor().fit(X_train, y_train)

reg     = LGB.fit(X_train, y_train)



# 3.) Define functions to calculate a score

def score_function(y_true, y_pred):

    # see https://www.kaggle.com/c/tmdb-box-office-prediction/overview/evaluation

    # we use Root Mean squared logarithmic error (RMSLE) regression loss

    assert len(y_true) == len(y_pred)

    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))



def score_function2(y_true, y_pred):

    # alternative implementation

    y_pred = np.where(y_pred>0, y_pred, 0)

    return np.sqrt(mean_squared_log_error(y_true, y_pred))

    

# 4.) Apply the regression model on the prepared train, validation and test set and invert the logarithmic scaling

y_train_pred  = np.exp(y_scaler.inverse_transform(np.reshape(reg.predict(X_train), (-1,1))))

y_val_pred    = np.exp(y_scaler.inverse_transform(np.reshape(reg.predict(X_val), (-1,1))))

y_test_pred   = np.exp(y_scaler.inverse_transform(np.reshape(reg.predict(X_test), (-1,1))))

                   

# 5.) Print the RMLS error on training, validation and test set. it should be as low as possible

print("RMLS Error on Training Dataset:\t", score_function(y_train , y_train_pred), score_function2(y_train , y_train_pred))

print("RMLS Error on Validation Dataset:\t", score_function(y_val , y_val_pred), score_function2(y_val , y_val_pred))

print("RMLS Error on Provided Test Dataset:\t mystery!")
# Make sure my shapes are in order

# If they are not, go back through your code. Did you remove any rows during prep?



test.shape

y_test_pred.shape

test.index
test.columns
# 1.) Add the predicted values to the original test data

df_test = test.assign(revenue=y_test_pred)



# 2.) Extract a table of ids and their revenue predictions



output = df_test.loc[:, lambda df_test: [ 'revenue']]

output



# 3.) save that table to a csv file

output.to_csv("submission1c.csv")



# 4.) take a look

pd.read_csv("submission1c.csv").head(5)