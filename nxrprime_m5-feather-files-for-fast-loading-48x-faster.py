import plotly.express as px

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Copy from ragnar's kernel to reduce memory usage

from pandas.api.types import is_datetime64_any_dtype as is_datetime



def reduce_mem_usage(df, use_float16=False):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        if is_datetime(df[col]):

            # skip datetime type

            continue

        col_type = df[col].dtype

        

        if col_type != object:

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

                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
%%time

stv = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')

sales = pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')

cal = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')

ss = pd.read_csv('../input/m5-forecasting-accuracy/sample_submission.csv')

# From michael mayer's kernel

from sklearn.preprocessing import OrdinalEncoder

def prep_calendar(df):

    df = df.drop(["date", "weekday"], axis=1)

    df = df.assign(d = df.d.str[2:].astype(int))

    df = df.fillna("missing")

    cols = list(set(df.columns) - {"wm_yr_wk", "d"})

    df[cols] = OrdinalEncoder(dtype="int").fit_transform(df[cols])

    df = reduce_mem_usage(df)

    return df



def prep_selling_prices(df):

    gr = df.groupby(["store_id", "item_id"])["sell_price"]

    df["sell_price_rel_diff"] = gr.pct_change()

    df["sell_price_roll_sd7"] = gr.transform(lambda x: x.rolling(7).std())

    df["sell_price_cumrel"] = (gr.shift(0) - gr.cummin()) / (1 + gr.cummax() - gr.cummin())

    df = reduce_mem_usage(df)

    return df



def reshape_sales(df, drop_d = None):

    if drop_d is not None:

        df = df.drop(["d_" + str(i + 1) for i in range(drop_d)], axis=1)

    df = df.assign(id=df.id.str.replace("_validation", ""))

    df = df.reindex(columns=df.columns.tolist() + ["d_" + str(1913 + i + 1) for i in range(2 * 28)])

    df = df.melt(id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],

                 var_name='d', value_name='demand')

    df = df.assign(d=df.d.str[2:].astype("int16"))

    return df



def prep_sales(df):

    df['min'] = df['demand'].apply('min')

    df['max'] = df['demand'].apply('max')

    df['std'] = df['demand'].apply('std')

    df['mean'] = df['demand'].apply('mean')



    # Remove rows with NAs except for submission rows. rolling_mean_t180 was selected as it produces most missings

    df = reduce_mem_usage(df)



    return df
stv = reshape_sales(stv)

import gc

gc.collect()

stv = prep_sales(stv)

gc.collect()

stv.to_feather('sales_train_validation.feather')

del stv

sales = prep_selling_prices(sales)

gc.collect()

sales.to_feather('sell_prices.feather')

del sales

cal = prep_calendar(cal) ### does not help cal

gc.collect()

cal.to_feather('calendar.feather')

del cal

ss.to_feather('sample_submission.feather')
%%time

stv = pd.read_feather('sales_train_validation.feather')

sales = pd.read_feather('sell_prices.feather')

cal = pd.read_feather('calendar.feather')

ss = pd.read_feather('sample_submission.feather')