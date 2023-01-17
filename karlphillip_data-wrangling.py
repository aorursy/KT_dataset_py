import math
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
%matplotlib inline

air_reserve_df = pd.read_csv('../input/air_reserve.csv')
air_store_df = pd.read_csv('../input/air_store_info.csv')
air_visit_df = pd.read_csv('../input/air_visit_data.csv')

hpg_reserve_df = pd.read_csv('../input/hpg_reserve.csv')
hpg_store_df = pd.read_csv('../input/hpg_store_info.csv')

date_info_df = pd.read_csv('../input/date_info.csv')
store_ids_df = pd.read_csv('../input/store_id_relation.csv')
def visual_inspect(df, size=3):
    print('>>> Shape =', df.shape)
    display(df.head(size))    
    print('>>> Types\n', df.dtypes)
    #print('\n>>> Info')
    #df.info()
visual_inspect(air_reserve_df)
visual_inspect(air_store_df)
visual_inspect(air_visit_df)
visual_inspect(hpg_reserve_df)
visual_inspect(hpg_store_df)
visual_inspect(store_ids_df)
visual_inspect(date_info_df)
# Add a visit_date column for air_reserve
air_reserve_df['visit_datetime'] = pd.to_datetime(air_reserve_df['visit_datetime'])
air_reserve_df['visit_date'] = air_reserve_df['visit_datetime'].apply(lambda x: x.strftime('%Y-%m-%d'))

# Rename columns so they match during merge
air_reserve_df.rename(columns={'reserve_visitors': 'visitors'}, inplace=True)
print('>>> air_visit_df   shape=', air_visit_df.shape) 
# display(air_visit_df.head(2))
print('>>> air_reserve_df shape=', air_reserve_df.shape) 
# display(air_reserve_df.head(2))

print('>>> Merging...\n')
df = pd.merge(air_visit_df, air_reserve_df, on=["air_store_id", "visit_date", "visitors"], how="outer")

print('>>> df             shape=', df.shape) 
display(df.head())
print('>>> df             shape=', df.shape) 
# display(df.head(2))
print('>>> air_store_df   shape=', air_store_df.shape) 
# display(air_store_df.head(2))

print('>>> Merging...\n')
df = pd.merge(df, air_store_df, on="air_store_id", how="left")

print('>>> df             shape=', df.shape)
display(df.head(3))
# convert visit_datetime from string to datetime object to add a new column containing only the Y-M-D
hpg_reserve_df['visit_datetime'] = pd.to_datetime(hpg_reserve_df['visit_datetime'])
hpg_reserve_df['visit_date'] = hpg_reserve_df['visit_datetime'].apply(lambda x: x.strftime('%Y-%m-%d'))

print('>>> hpg_reserve_df shape=', hpg_reserve_df.shape)
display(hpg_reserve_df.head(1))
print('\n>>> store_ids_df   shape=', store_ids_df.shape)
display(store_ids_df.head(1))

# replace the HPG code in hpg_merged_df for its equivalent AIR code
print('>>> Merging...\n')
hpg_df = pd.merge(hpg_reserve_df, store_ids_df, on='hpg_store_id', how='left')

print('\n>>> hpg_df         shape=', hpg_df.shape)
display(hpg_df.head(3))

# DEBUG: print rows that have valid air_store_id and hpg_store_id
valid_ids_df = hpg_df[hpg_df.air_store_id.notnull() & hpg_df.hpg_store_id.notnull()]
valid_ids_percent = len(valid_ids_df.index) / len(hpg_df.index) * 100
print('>>> hpg_df rows with valid HPG & AIR ids: ',  round(valid_ids_percent, 1), '%')
hpg_df = pd.merge(hpg_df, hpg_store_df, on="hpg_store_id", how="left")

print('>>> hpg_df         shape=', hpg_df.shape)
display(hpg_df.head(3))
# rename columns in both dataframes so they blend in nicely during merge
hpg_df.rename(columns={'reserve_visitors': 'visitors'}, inplace=True)
hpg_df.rename(columns={'hpg_genre_name': 'genre_name'}, inplace=True)
hpg_df.rename(columns={'hpg_area_name' : 'area_name'}, inplace=True)
df.rename(columns={'air_genre_name': 'genre_name'}, inplace=True)
df.rename(columns={'air_area_name' : 'area_name'}, inplace=True)

print('>>> df             shape=', df.shape)
# display(df.head(2))
print('>>> hpg_df         shape=', df.shape)
# display(hpg_df.head(2))

print('\n>>> Merging...\n')
df = pd.concat([df, hpg_df], axis=0).reset_index(drop=True)

print('>>> df             shape=', df.shape)
display(df.head(3))
print('>>> date_info_df   shape=', date_info_df.shape)
# display(date_info_df.head(2))
print('>>> df             shape=', df.shape)
# display(df.tail(3))

# convert column visit_datetime from string to datetime object
df['visit_datetime'] = pd.to_datetime(df['visit_datetime'])

# combine both datasets to add columns 'day_of_week' and 'holiday_flg' to merged_df
print('\n>>> Merging...\n')
df = pd.merge(df, date_info_df, left_on='visit_date', right_on='calendar_date')
del df['calendar_date']

print('>>> df             shape=', df.shape)
display(df.tail(3))
def check_nan_rows(df):
    ## Store all rows with NaN values
    nan_df = df[pd.isnull(df).any(axis=1)]
    print('>>> Number of rows that have at least ONE NaN value:', len(nan_df.index))
    #display(nan_df.head())

    ## List only rows that have NaN values (in all columns)
    nan_rows_df = nan_df[nan_df.isnull().all(1)]      
    nan_rows_count = len(nan_rows_df.index)
    print('>>> Number of rows completely filled with NaN values:', nan_rows_count)    
    #display(nan_rows_df.head())

    if (nan_rows_count != 0):
        print('>>> Oopsie! Found', nan_rows_count, 'NaN rows:')
        display(nan_rows_df.head())        
check_nan_rows(df)  
def check_IDs(df):
    # Count how many IDs are NaNs
    print('>>> NaN count on air_store_id:', df.air_store_id.isnull().sum())
    print('>>> NaN count on hpg_store_id:', df.hpg_store_id.isnull().sum())

    # Count how many rows don't have a valid ID (i.e. air_store_id and hpg_store_id are both NaNs)
    null_ids_df = df[df.air_store_id.isnull() & df.hpg_store_id.isnull()]
    null_ids_count = len(null_ids_df.index)
    print('>>> Number of rows without any IDs:', null_ids_count)
    
    if (null_ids_count != 0):
        display(null_ids_df.head())

    # Check how many IDs don't have any data at all (i.e. all columns are NaNs)
    max_nan_per_row = len(df.columns) - 1    
    df['NaN_count'] = df.isnull().sum(axis=1)    
    invalid_ids_df = df[df['NaN_count'] == max_nan_per_row]    
    
    invalid_ids_count = len(invalid_ids_df.index)
    print('>>> Number of rows with IDs but NO DATA at all connected to them:', invalid_ids_count)
    
    if (invalid_ids_count != 0):
        print('>>> Found', invalid_ids_count, ' rows with IDs that have NO DATA at all')
        display(invalid_ids_df.head())
        
    del df['NaN_count']
check_IDs(df)
def get_data_types(df):    
    col_names = df.columns.tolist()
    data_types_df = pd.DataFrame(columns=col_names)    
    
    row = []
    for col in col_names:        
        row.append(str(df[col].dtype))

    data_types_df.loc[0] = row
    return data_types_df
dtypes_df = get_data_types(df)
display(dtypes_df.head())
df['reserve_datetime'] = pd.to_datetime(df['reserve_datetime'])
df['visit_date'] = pd.to_datetime(df['visit_date'])
dtypes_df = get_data_types(df)
display(dtypes_df.head())
# Checks numeric columns for negative numbers
def check_numeric(df):     
    neg_count_df = pd.DataFrame(columns=['Negative values count'])
    
    #print('>>> Number of negative values found in numeric columns:')
    num_col_list = list(df.select_dtypes(include=['int64', 'float64']).columns)    
    
    total_neg = 0
    for col_name in num_col_list:
        neg_count = df[df[col_name] < 0].shape[0]  # extract number of rows        
        #print('\t*' + col_name+ '* = ' + str(neg_count))
        neg_count_df.loc[col_name] = neg_count
        total_neg += neg_count
    
    return neg_count_df, total_neg
neg_count_df, total_neg_count = check_numeric(df)
display(neg_count_df)
# Checks columns separately for NaN values
def check_invalid(df): 
    nan_count_df = pd.DataFrame(columns=['NaN values count'])
    
    #print('>>> Number of NaN values found in each column:')
    col_list = list(df.columns.tolist())    
    
    total_nan = 0
    for col_name in col_list:
        nan_count = df[col_name].isnull().sum()        
        #print('\t' + col_name+ ' = ' + str(nan_count))
        nan_count_df.loc[col_name] = nan_count        
        total_nan += nan_count
    
    return nan_count_df, total_nan
# count how many NaNs are there
nan_count_df, total_nan_count = check_invalid(df)
display(nan_count_df)
# plots a pie chart to visualize data corruption
def plot_corruption(total_values, total_neg_count, total_nan_count):
    neg_fraction = total_neg_count / total_values * 100
    nan_fraction = total_nan_count / total_values * 100
    ok_fraction  =  (total_values - (neg_fraction + nan_fraction)) / total_values * 100

    # Data to plot
    labels = 'Negative values', 'NaN values', 'Good values'
    sizes = [neg_fraction, nan_fraction, ok_fraction]
    colors = ['orange', 'lightcoral', 'yellowgreen']
    explode = (0.2, 0.2, 0)  # explode 1st and 2nd slice

    # Plot
    plt.subplots(figsize=(9,4))
    patches, texts = plt.pie(sizes, explode=explode, labels=labels, colors=colors, shadow=True, startangle=330)
    plt.legend(patches, labels, loc="best")
    plt.axis('equal')
    plt.show()
    
    print('>>>', round(nan_fraction, 1), '% of the values are NaNs.')
    print('>>> There are', total_values, 'cells in the DataFrame. Shape =', df.shape)
total_values = df.shape[0] * df.shape[1]
plot_corruption(total_values, total_neg_count, total_nan_count)
df['area_name'].fillna(value='', inplace=True)
df['genre_name'].fillna(value='', inplace=True)
df.fillna(pd.concat([ df.hpg_store_id.map(store_ids_df.set_index('hpg_store_id').air_store_id),
                      df.air_store_id.map(store_ids_df.set_index('air_store_id').hpg_store_id),
                    ], axis=1, keys=['air_store_id', 'hpg_store_id']), inplace=True)
df['air_store_id'].fillna(value='', inplace=True)
df['hpg_store_id'].fillna(value='', inplace=True)
df['latitude'].fillna(value=-1, inplace=True)
df['longitude'].fillna(value=-1, inplace=True)
df['reserve_datetime'].fillna(value=-9999, inplace=True)
df['visit_datetime'].fillna(value=-9999, inplace=True)
nan_count_df, total_nan_count = check_invalid(df)
display(nan_count_df)
def plot_reservation_vs_visitors(df):
    # set a datetime index so resample() works properly
    df = df.set_index('visit_date')
    reservations_made_df = df.resample('W').apply({'visitors':'count'})
    visitors_df = df.resample('W').apply({'visitors':'sum'})

    activity_df = pd.concat([reservations_made_df, visitors_df], join='inner', axis=1)
    activity_df.columns = ['reservations_made', 'visitors']

    fig, ax = plt.subplots(figsize=(15,5))

    # There's a bug in Pandas' plot() that handles data differently than Matplotlib's plot()
    # What this means is that if you use Pandas' plot() you cannot change xticks values later.
    #ax = activity_df.plot(kind='area', ax=ax, stacked=True, title='Reservations made and number of visitors (per week)')

    # alternative method: use matplotlib's plot_date() and configure xticks at will
    plt.plot_date(activity_df.index.to_pydatetime(), activity_df, fmt='-')
    plt.title('Number of reservations made VS visitors (per week)')    
    ax = plt.gca()
    ax.xaxis.grid(True, which="major")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%y'))

    d = activity_df.index.to_pydatetime()
    h_v = plt.fill_between(d, activity_df['visitors'], activity_df['reservations_made'], 
                           activity_df['visitors'] >= activity_df['reservations_made'],
                           facecolor='orange', alpha=0.5, interpolate=True)
    h_r = plt.fill_between(d, 0, activity_df['reservations_made'], 
                           facecolor='blue', alpha=0.5, interpolate=True)
    
    plt.legend(handles=[h_v, h_r], labels=["Visitors", "Reservations"], loc='upper left')

    ax.set_ylabel('Number of people')
    ax.margins(0.005, 0) # set margins to avoid "whitespace" while showing the first x-tick
    plt.show()
plot_reservation_vs_visitors(df)
