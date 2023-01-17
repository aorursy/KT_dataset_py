import numpy as np

import pandas as pd

from sklearn.metrics import mean_squared_error

from scipy.sparse import csr_matrix

import gc
# Memory reduction helper function:

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns: #columns

        col_type = df[col].dtypes

        if col_type in numerics: #numerics

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
data_pass = '/kaggle/input/m5-forecasting-accuracy/'



# Sales quantities:

sales = pd.read_csv(data_pass+'sales_train_validation.csv')



# Calendar to get week number to join sell prices:

calendar = pd.read_csv(data_pass+'calendar.csv')

calendar = reduce_mem_usage(calendar)



# Sell prices to calculate sales in USD:

sell_prices = pd.read_csv(data_pass+'sell_prices.csv')

sell_prices = reduce_mem_usage(sell_prices)
# Dataframe with only last 28 days:

cols = ["d_{}".format(i) for i in range(1914-28, 1914)]

data = sales[["id", 'store_id', 'item_id'] + cols]



# To long form:

data = data.melt(id_vars=["id", 'store_id', 'item_id'], 

                 var_name="d", value_name="sale")



# Add week of year column from 'calendar':

data = pd.merge(data, calendar, how = 'left', 

                left_on = ['d'], right_on = ['d'])



data = data[["id", 'store_id', 'item_id', "sale", "d", "wm_yr_wk"]]



# Add weekly price from 'sell_prices':

data = data.merge(sell_prices, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')

data.drop(columns = ['wm_yr_wk'], inplace=True)



# Calculate daily sales in USD:

data['sale_usd'] = data['sale'] * data['sell_price']

data.head()



#this part is correct
# List of categories combinations for aggregations as defined in docs:

dummies_list = [sales.state_id, sales.store_id, 

                sales.cat_id, sales.dept_id, 

                sales.state_id + sales.cat_id, sales.state_id + sales.dept_id,

                sales.store_id + sales.cat_id, sales.store_id + sales.dept_id, 

                sales.item_id, sales.state_id + sales.item_id, sales.id]





## First element Level_0 aggregation 'all_sales':

dummies_df_list =[pd.DataFrame(np.ones(sales.shape[0]).astype(np.int8), 

                               index=sales.index, columns=['all']).T]



# List of dummy dataframes:

for i, cats in enumerate(dummies_list):

    dummies_df_list +=[pd.get_dummies(cats, drop_first=False, dtype=np.int8).T]

    

# Concat dummy dataframes in one go:

## Level is constructed for free.

roll_mat_df = pd.concat(dummies_df_list, keys=list(range(12)), 

                        names=['level','id'])#.astype(np.int8, copy=False)



# Save values as sparse matrix & save index for future reference:

roll_index = roll_mat_df.index

roll_mat_csr = csr_matrix(roll_mat_df.values)

roll_mat_csr.shape
# Dump roll matrix to pickle:

roll_mat_df.to_pickle('roll_mat_df.pkl')
# Free some momory:

del dummies_df_list, roll_mat_df

gc.collect()
# Fucntion to calculate S weights:

def get_s(drop_days=0):

    

    """

    drop_days: int, equals 0 by default, so S is calculated on all data.

               If equals 28, last 28 days won't be used in calculating S.

    """

    

    # Rollup sales:

    d_name = ['d_' + str(i+1) for i in range(1913-drop_days)]

    sales_train_val = roll_mat_csr * sales[d_name].values



    # Find sales start index:

    start_no = np.argmax(sales_train_val>0, axis=1)

    

    # Replace days less than min day number with np.nan:

    # Next code line is super slow:

    flag = np.dot(np.diag(1/(start_no+1)) , np.tile(np.arange(1,1914-drop_days),(roll_mat_csr.shape[0],1)))<1

    sales_train_val = np.where(flag, np.nan, sales_train_val)



    # Denominator of RMSSE / RMSSE

    weight1 = np.nansum(np.diff(sales_train_val,axis=1)**2,axis=1)/(1913-start_no-1)

    

    return weight1
drop_days = 0

d_name = ['d_' + str(i+1) for i in range(1913-drop_days)]

sales_train_val = roll_mat_csr * sales[d_name].values



# Find sales start index:

start_no = np.argmax(sales_train_val>0, axis=1)



flag = np.dot(np.diag(1/(start_no+1)) , np.tile(np.arange(1,1914-drop_days),(roll_mat_csr.shape[0],1)))<1



print(sales_train_val[230][752])

print(flag[230][752]) # should be False
# Fucntion to calculate S weights:

def fixed_get_s(drop_days=0):

    

    """

    drop_days: int, equals 0 by default, so S is calculated on all data.

               If equals 28, last 28 days won't be used in calculating S.

    """

    

    # Rollup sales:

    d_name = ['d_' + str(i+1) for i in range(1913-drop_days)]

    sales_train_val = roll_mat_csr * sales[d_name].values



    no_sales = np.cumsum(sales_train_val, axis=1) == 0

    sales_train_val = np.where(no_sales, np.nan, sales_train_val)

    

    # Denominator of RMSSE / RMSSE

    weight1 = np.nanmean(np.diff(sales_train_val,axis=1)**2,axis=1)

    

    return weight1
%%timeit -n 3 -r 1

get_s(drop_days=0)
%%timeit -n 3 -r 1

fixed_get_s(drop_days=0)
S = fixed_get_s(drop_days=0)

S.shape
# S values from AGG & WRMSSE Evaluator:

# array([3.26268315e+05, 5.14239651e+05, 5.17917913e+05, ...,

#       1.71293871e-01, 6.98666667e-02, 2.81004710e-01])

# Good match:

S[10:]
# Functinon to calculate weights:

def get_w(sale_usd):

    """

    """

    # Calculate the total sales in USD for each item id:

    total_sales_usd = sale_usd.groupby(

        ['id'], sort=False)['sale_usd'].apply(np.sum).values

    

    # Roll up total sales by ids to higher levels:

    weight2 = roll_mat_csr * total_sales_usd

    

    return 12*weight2/np.sum(weight2)
def faster_get_w(sale_usd):

    """

    """

    # Calculate the total sales in USD for each item id:

    total_sales_usd = sale_usd.groupby(

        ['id'], sort=False)['sale_usd'].sum().values # use vectorized sum

    

    # Roll up total sales by ids to higher levels:

    weight2 = roll_mat_csr * total_sales_usd

    

    return 12*weight2/np.sum(weight2)
%%timeit -n 3 -r 1

get_w(data[['id','sale_usd']])
%%timeit -n 3 -r 1

faster_get_w(data[['id','sale_usd']])
W = faster_get_w(data[['id','sale_usd']])

W.shape
# Predicted weights

W_df = pd.DataFrame(W,index = roll_index,columns=['w'])



# Load the original weights:

data_pass = '/kaggle/input/weights-validation/'

W_original_df = pd.read_csv(data_pass+'weights_validation.csv')



# Set new index, calculate difference between original and predicted:

W_original_df = W_original_df.set_index(W_df.index)

W_original_df['Predicted'] = W_df.w

W_original_df['diff'] = W_original_df.Weight - W_original_df.Predicted



# See where we are off by more than e-6

m = W_original_df.Weight.values - W_df.w.values > 0.000001

W_original_df[m]
SW = W/np.sqrt(S)
sw_df = pd.DataFrame(np.stack((S, W, SW), axis=-1),index = roll_index,columns=['s','w','sw'])

sw_df.to_pickle('sw_df.pkl')
# Function to do quick rollups:

def rollup(v):

    '''

    v - np.array of size (30490 rows, n day columns)

    v_rolledup - array of size (n, 42840)

    '''

    return (v.T*roll_mat_csr.T).T





# Function to calculate WRMSSE:

def wrmsse(preds, y_true, score_only=False, s = S, w = W, sw=SW):

    '''

    preds - Predictions: pd.DataFrame of size (30490 rows, N day columns)

    y_true - True values: pd.DataFrame of size (30490 rows, N day columns)

    sequence_length - np.array of size (42840,)

    sales_weight - sales weights based on last 28 days: np.array (42840,)

    '''

    

    if score_only:

        return np.sum(

                np.sqrt(

                    np.mean(

                        np.square(rollup(preds.values-y_true.values))

                            ,axis=1)) * sw)/12 #<-used to be mistake here

    else: 

        score_matrix = (np.square(rollup(preds.values-y_true.values)) * np.square(w)[:, None])/ s[:, None]

        score = np.sum(np.sqrt(np.mean(score_matrix,axis=1)))/12 #<-used to be mistake here

        return score, score_matrix
# Define fold pass here:

file_pass = '/kaggle/working/'# '/kaggle/input/fast-wrmsse-and-sw-frame/'



# Load S and W weights for WRMSSE calcualtions:

sw_df = pd.read_pickle(file_pass+'sw_df.pkl')

S = sw_df.s.values

W = sw_df.w.values

SW = sw_df.sw.values



# Load roll up matrix to calcualte aggreagates:

roll_mat_df = pd.read_pickle(file_pass+'roll_mat_df.pkl')

roll_index = roll_mat_df.index

roll_mat_csr = csr_matrix(roll_mat_df.values)

del roll_mat_df
# Predictions:

sub = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')

sub = sub[sub.id.str.endswith('validation')]

sub.drop(['id'], axis=1, inplace=True)



DAYS_PRED = sub.shape[1]    # 28



# Ground truth:

dayCols = ["d_{}".format(i) for i in range(1914-DAYS_PRED, 1914)]

y_true = sales[dayCols]
%%timeit -n 100 -r 5

# n - execute the statement n times 

# r - repeat each loop r times and return the best



score = wrmsse(sub, y_true, score_only=True)
%%timeit -n 100 -r 5

# n - execute the statement n times 

# r - repeat each loop r times and return the best



score1, score_matrix = wrmsse(sub, y_true)
score = wrmsse(sub, y_true, score_only=True)

score
score1, score_matrix = wrmsse(sub, y_true)

score_df = pd.DataFrame(score_matrix, index = roll_index)

score_df.reset_index(inplace=True)

score_df.head()

score1