#import necessary packages
import pandas as pd 
import numpy as np 

# Pandas_profiling is not necessary but invaluable
import pandas_profiling

# this notebook only requires plotly express - adding other for expandibility
import plotly.express as px 
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Dataset is not available to public, it requires a federally sponsored login to NPMRDS
# please see .csv for Alaska NPMRDS dataset, courtesy of Alaska Department of Transportation & Public Facilities

print('getting data')
ttr = '../input/alaska-travel-time-reliability/2019_Alaska_NPMRDS_MassiveData.csv'
print('creating dataframe')
df = pd.read_csv(ttr)
print('shape: ' + str(df.shape))
df.sample(5)
def assess_df(df):
    """Simple function to combine various pandas tools to view the data.
    includes info(), describe(), isnull().sum()"""
    
    pd.options.display.float_format = '{:.8f}'.format
    
    print("*DATAFRAME INFORMATION*")
    print(df.info())
    print()      
    # statistical description
    print("*STATISTICAL DESCRIPTION*")
    print(df.describe())
    print()
    # Count of null rows per column
    print("*MISSING VALUES COUNT*")
    print(df.isnull().sum())
    
assess_df(df)
    
# call pandas_profiling - will take some time to load due to size of data
df.profile_report()
def percentile_95(df,column):
    q95 = df[column].quantile(.95)
    print(f"{column} 95th percentile: {q95}")
    greater_95 = df.loc[df[column] > q95]
    print(f'Count of values in Dataframe: {df[column].count()}')
    print(f'Count of values in 95th percentile : {greater_95[column].count()}')
    print(f'Minimum value in 95th percentile : {greater_95[column].min()}')
    print(f'Maximum value in 95th percentile : {greater_95[column].max()}')
    
    return greater_95

df_95 = percentile_95(df, 'travel_time_seconds')
fig = px.histogram(df_95, x='travel_time_seconds', log_y='count')
fig.show()
def deviation_spread(df, column):
    """Function to take a DataFrame and target column 
    to return the standard deviation
    as well as the 3rd standard deviation according to zscore.
    Then prints the amount of row records that lie in the outer distribution 
    and additionally an output of a new dataframe ."""
    
    col = df[column]
    std = col.std()
    mean = col.mean()
    print(f"{column} Standard Deviation: {std}")
    print(f'{column} Mean: {mean}')
    
    #Compute the z score of each value in the sample, relative to the sample mean and standard deviation.
    from scipy import stats
    df['z'] = z = np.abs(stats.zscore(col))
    
    print(f'Average z score: {df.z.mean()}')
    print(f'Total row records: {col.count()}')
    z_less3 = str(len(df[z<3]))
    print(f'Total row records within three standard deviations: {z_less3}')
    z_greater3 = str(len(df[z>3]))
    print(f'Count of values in outer distribution:  {z_greater3}')
    df_outliers = df.loc[df['z'] > 3]
    df_outliers_max = df_outliers[column].max()
    df_outliers_min = df_outliers[column].min()
    print(f'Outliers min value: {df_outliers_min}')
    print(f'Outliers max value: {df_outliers_max}')
    
    return df_outliers

df_outliers = deviation_spread(df, 'travel_time_seconds')
fig = px.histogram(df_outliers, x='travel_time_seconds', log_y='count')
fig.show()
def iqr_value(df, column):
    """Function to take a DataFrame and target column 
    to return the Interquartile Range (IQR).
    Then prints the amount of row records that lie beyond the IQR 
    and additionally an output of a new dataframe with a cutoff threshold."""
    
    #Compute the IQR, relative to the 25th percentile and 75th percentile
    from scipy.stats import iqr
    iqr_val = iqr(df[column])
    
    col = df[column]
    q25 = col.quantile(.25)
    q75 = col.quantile(.75)
    print(f"{column} IQR: {iqr_val}")
    print(f'{column} 25th percentile: {q25}')
    print(f'{column} 75th percentile: {q75}')
    
    cut_off = iqr_val * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    
    df_iqr = df.loc[df[column] > iqr_val]
    
    df_cutoff = df.loc[df[column] > upper]

    print(f'Count of values in Dataframe: {df[column].count()}')
    print(f'Count of values beyond the IQR value: {df_iqr[column].count()}')
    print(f'Minimum value in IQR DataFrame : {df_iqr[column].min()}')
    print(f'Maximum value in IQR DataFrame : {df_iqr[column].max()}')
    print()
    print(f'travel_time_seconds cutoff value: {upper}')
    print(f'Count of values beyond the Cutoff Threshold: {df_cutoff[column].count()}')
    print(f'Minimum value in IQR DataFrame : {df_cutoff[column].min()}')
    print(f'Maximum value in IQR DataFrame : {df_cutoff[column].max()}')

    return df_cutoff

df_cutoff = iqr_value(df, 'travel_time_seconds')
fig = px.histogram(df_cutoff, x='travel_time_seconds', log_y='count', title='Cutoff Threshold')
fig.show()
def count_outliers(df_whole, df_subset, column):
    """Function to count outliers from a sebset dataframe (must be created already),
    and compare the 'outliers' dataframet to another dataframe (the original)"""
    
    # quick view of how many tmc_codes are responsible
    outliers = df_subset[column].value_counts().count()
    original_count = df_whole[column].value_counts().count()
    outliers_percent = (df_subset[column].value_counts().count()/df_whole[column].value_counts().count())*100
    
    print(f'TMC outliers count: {outliers}')
    print(f'Original Dataset Count: {original_count}')
    print(f'Percentage of TMC with outliers: {outliers_percent}')
    print()
    # and how many times each of them contributing outliers
    outliers_count = df_subset[column].value_counts()
    outliers_count.index.names = [column]
    outliers_count.rename('outliers_count')
    print(outliers_count.to_markdown())
    
count_outliers(df, df_outliers, 'tmc_code')
# quick look at the distance between the mean and mode of individual features in the dataframe

from scipy import stats

dfgroup = df.groupby('tmc_code')['travel_time_seconds'].agg(['mean', lambda x: stats.mode(x)[0], 'max','min','median', lambda x: x.quantile(.95)])
dfgroup

dfgroup_meanless = dfgroup.loc[dfgroup['mean'] < dfgroup['<lambda_0>']]

dfgroup['distance'] = np.abs(dfgroup['mean']-dfgroup['<lambda_0>'])

dfgroup_meanless.shape[0]
# provided by StackOverflow user:3218693, edited and implemented by author
# https://stackoverflow.com/questions/64185797/how-to-accomplish-row-selection-criteria-per-unique-id-with-pandas-groupby-cal

def clean_outliers(df, features, column):
    """Function that takes DataFrame, groups by feature (category, despite cardinality),
    generates mean and 95th percentile by feature values,
    creates a threshold by adding mean and 95th percentile,
    creates a nested function to filter rows above threshold,
    then maps the filter against the DataFrame features and selected column values
    """
    
    # 1. aggregate everything at once
    df_agg = df.groupby(features).agg(
        mean=(column, pd.Series.mean),
        q95=(column, lambda x: np.quantile(x, .95))
    )

    # 2. construct a lookup table for the thresholds
    threshold = {}
    for feature, row in df_agg.iterrows():  # slow but only 1.2k rows
        threshold[feature] = np.max(row["mean"]) + row["q95"]

    # 3. filtering
    def f(feature, values):
        return values <= threshold[feature]

    df_so = df[list(map(f, df[features].values, df[column].values))]
    return df_so

df_so = clean_outliers(df, 'tmc_code','travel_time_seconds')
assess_df(df_so)
# CAREFUL
# This takes a while to generate
fig = px.histogram(df_so, x='travel_time_seconds', log_y='count')
fig.show()
#dictionary for function

convert_dict={'speed': int,
                  'reference_speed': int,
                  'data_density': int,
                 'month': int,
                  'day_of_year': int,
                  'hour':int}

def transform_df(df, nulls, drop, categorical, convert__dict, time_measure):
    """Function that takes a dataframe, 
    cleans up nuls in identified column (by row),
    drops identified columns,
    converts columns datatype according to dictionary,
    converts time measurement column to datetime,
    then returns new clean DataFrame"""
    
    
    # drop nulls and reduce dimensionality/size
    print('dropping nulls')
    df_features = df.dropna(axis=0, subset=[nulls])
    print('dropping "drop columns"')
    df_features = df_features.drop([drop], axis=1)
    print('dropped shape: ' + str(df_features.shape))
    print('original shape: ' + str(df_features.shape))
    
    #convert categorical column with unique values to numerical coded values
    categ_list = list(df_features[categorical].unique())
    cat_dict = {}
    i = 0
    for i in range(len(categ_list)):
        cat_dict.update({categ_list[i]: i+1})
        i += 1

    # replace string values for integers
    df_features[categorical].replace(cat_dict, inplace=True)

    # transform values from measurement_tstamp to datetime
    print('datetime transform')
    df_features[time_measure] = pd.to_datetime(df_features[time_measure])
    df_features = df_features
    print('indexing datetime')
    df_features.index = df_features[time_measure]

    # visualizations & computations are enhanced by values parsed out
    print('adding month/day columns')
    df_features['month'] = df_features[time_measure].dt.month
    df_features['day_of_year'] = df_features[time_measure].dt.dayofyear
    df_features['day_name'] = df_features[time_measure].dt.day_name()
    df_features['hour'] = df_features[time_measure].dt.hour

    #create dictionary for converting values
    print('converting types')
    df_features= df_features.astype(convert_dict)
    print('conversion complete')
    print('Complete')
    print()
    df_features.info()
    
    return df_features
    
df_features = transform_df(df_so, 'reference_speed', 'z', 'data_density', convert_dict, 'measurement_tstamp')

tmc_id = r'C:\Users\ejmason\JupyterNotebooks\NPMRDS\2019_Alaska_NPMRDS_MassiveData\TMC_Identification.csv'

convert_dict_xy={'start_latitude': 'float32',
              'start_longitude': 'float32',
              'zip': 'int32',
              'miles':'float32'}

def df_xy_merge(xy, df, nulls, convert_dict):
    """
    function to take related TMC_Identification sheet,
    format for easy merging with travel time data,
    transform and convert,
    and generate new dataframe
    """
    print('Getting additional data')
    # add in additional sheet for getting coordinates to tmc's
    xy = pd.read_csv(xy)

    # rename column for easier merging
    print('renaming columns for merge')
    xy = xy.rename(columns={'tmc':'tmc_code'})

    # there are duplicates for start/end active dates, 
    # unnecessary for the scope of appending spatial values

    print(xy.tmc_code.count())
    xy.drop_duplicates(subset=['tmc_code'],inplace=True)

    print(xy.tmc_code.count())

    # merge for future use involving mapping or geospatial components
    print('Merge started')
    df_xy = df.merge(xy[['tmc_code',
                         'county',
                         'zip',
                         'miles',
                         'start_latitude',
                         'start_longitude']], 
                     on='tmc_code', 
                     how='left')
    df_xy.dropna(axis=0, subset=[nulls])
    print('Merge complete')
    
    print('converting types')
    df_xy= df_xy.astype(convert_dict)
    df_xy.info()
    
    return df_xy

df_xy = df_xy_merge(tmc_id, df_features, 'miles', convert_dict_xy)
# Travel Time Relability Metrics (TTR)


def ttr_create(data, df, feature):
    '''
    Function recreates the process of calculation to generate Level of Travel Time Reliability.
    It takes the dataset (travel_time_seconds),
    the dataframe comtaining the TMC_codes and attributes,
    and identifies the tmc station column.
    Currently, the function only works with the data that has been processed 
    with this notebook (or comparable)
    but could be extended in the future
    '''
    
    # read in the data to dataframe, small reformatting and dropping duplicates
    tmc_df = pd.read_csv(data)
    tmc_df = tmc_df.rename(columns={feature:'tmc_code'})
    tmc_df.drop_duplicates(subset=['tmc_code'],inplace=True)
    
    print("starting TTR process")
    ########################################################################################
    # AM period LOTTR Process

    print("starting AM process")
    lottr_amp_tf = df.loc[(df['hour'].between(6,9,inclusive=True))
                                        & (df['day_name'].isin(['Monday', 
                                                                  'Tuesday',
                                                                  'Wednesday',
                                                                  'Thursday',
                                                                  'Friday']))]

    lottr_amp_tf_p80 = lottr_amp_tf.groupby('tmc_code')['travel_time_seconds'].quantile(0.8)
    lottr_amp_tf_p50 = lottr_amp_tf.groupby('tmc_code')['travel_time_seconds'].quantile(0.5)
    lottr_amp = lottr_amp_tf_p80 / lottr_amp_tf_p50
    lottr_amp = round(lottr_amp,2) 
    
    lottr_amp = lottr_amp.to_frame()
    lottr_amp = lottr_amp.reset_index()
    lottr_amp = lottr_amp.rename(columns={"travel_time_seconds": "LOTTR_AMP"})

    print(f'Merge for AM started')
    tmc_df = tmc_df.merge(lottr_amp, on='tmc_code', how='left')
    print('Merge complete')

    ########################################################################################
    # MID period LOTTR Process

    print("starting MID process")
    lottr_mid_tf = df.loc[(df['hour'].between(10,15,inclusive=True))
                                        & (df['day_name'].isin(['Monday', 
                                                                  'Tuesday',
                                                                  'Wednesday',
                                                                  'Thursday',
                                                                  'Friday']))]

    lottr_mid_tf_p80 = lottr_mid_tf.groupby('tmc_code')['travel_time_seconds'].quantile(0.8)
    lottr_mid_tf_p50 = lottr_mid_tf.groupby('tmc_code')['travel_time_seconds'].quantile(0.5)
    lottr_mid = lottr_mid_tf_p80 / lottr_mid_tf_p50
    lottr_mid = round(lottr_mid,2)
    
    lottr_mid = lottr_mid.to_frame()
    lottr_mid = lottr_mid.reset_index()
    lottr_mid = lottr_mid.rename(columns={"travel_time_seconds": "LOTTR_MID"})

    print(f'Merge for Mid started')
    tmc_df = tmc_df.merge(lottr_mid, on='tmc_code', how='left')
    print('Merge complete')

    ########################################################################################
    # PM period LOTTR Process

    print("starting PM process")
    lottr_pm_tf = df.loc[(df['hour'].between(16,19,inclusive=True))
                                        & (df['day_name'].isin(['Monday', 
                                                                  'Tuesday',
                                                                  'Wednesday',
                                                                  'Thursday',
                                                                  'Friday']))]

    lottr_pm_tf_p80 = lottr_pm_tf.groupby('tmc_code')['travel_time_seconds'].quantile(0.8)
    lottr_pm_tf_p50 = lottr_pm_tf.groupby('tmc_code')['travel_time_seconds'].quantile(0.5)
    lottr_pm = lottr_pm_tf_p80 / lottr_pm_tf_p50
    lottr_pm = round(lottr_pm,2) 
    
    lottr_pm = lottr_pm.to_frame()
    lottr_pm = lottr_pm.reset_index()
    lottr_pm = lottr_pm.rename(columns={"travel_time_seconds": "LOTTR_PM"})

    print(f'Merge for PM started')
    tmc_df = tmc_df.merge(lottr_pm, on='tmc_code', how='left')
    print('Merge complete')

    ########################################################################################
    # Weekend period LOTTR Process

    print("starting Weekend process")
    lottr_we_tf = df.loc[(df['hour'].between(6,19,inclusive=True))
                                        & (df['day_name'].isin(['Saturday', 
                                                                  'Sunday']))]

    lottr_we_tf_p80 = lottr_we_tf.groupby('tmc_code')['travel_time_seconds'].quantile(0.8)
    lottr_we_tf_p50 = lottr_we_tf.groupby('tmc_code')['travel_time_seconds'].quantile(0.5)
    lottr_we = lottr_we_tf_p80 / lottr_we_tf_p50
    lottr_we = round(lottr_we,2)

    lottr_we = lottr_we.to_frame()
    lottr_we = lottr_we.reset_index()
    lottr_we = lottr_we.rename(columns={"travel_time_seconds": "LOTTR_WE"})

    print(f'Merge for Weekend started')
    tmc_df = tmc_df.merge(lottr_we, on='tmc_code', how='left')
    print('Merge complete')

    ########################################################################################
    # Overnight period LOTTR Process
    # not used as a measure (only used for freight, which is a different ratio)

    print("starting Overnight process")
    lottr_ov_tf = df.loc[(df['hour'].isin([20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7]))
                                        & (df['day_name'].isin(['Monday', 
                                                                  'Tuesday',
                                                                  'Wednesday',
                                                                  'Thursday',
                                                                  'Friday',
                                                                  'Saturday', 
                                                                  'Sunday']))]

    lottr_ov_tf_p80 = lottr_ov_tf.groupby('tmc_code')['travel_time_seconds'].quantile(0.8)
    lottr_ov_tf_p50 = lottr_ov_tf.groupby('tmc_code')['travel_time_seconds'].quantile(0.5)
    lottr_ov = lottr_ov_tf_p80 / lottr_ov_tf_p50
    lottr_ov = round(lottr_ov,2) 

    lottr_ov = lottr_ov.to_frame()
    lottr_ov = lottr_ov.reset_index()
    lottr_ov = lottr_ov.rename(columns={"travel_time_seconds": "LOTTR_OV"})

    print(f'Merge for Overnight started')
    tmc_df = tmc_df.merge(lottr_ov, on='tmc_code', how='left')
    print('Merge complete')
    print('Process Complete')
    return tmc_df

tmc_df= ttr_create(tmc_id, df_xy, 'tmc')
lottr_list = list(tmc_df.columns[39:45])
lottr_list
tmc_df.dropna(subset=lottr_list)
df_features_orig = transform_df(df, 'reference_speed', 'z', 'data_density', convert_dict, 'measurement_tstamp')

tmc_df_orig = ttr_create(tmc_id, df_features_orig, 'tmc')
lottr_list = list(tmc_df.columns[39:45])
lottr_list

def diff_df(df, df_orig, features, diff_list):
    df_diff = pd.DataFrame()

    df_diff[features] = df[features]
    df_diff['AM_Diff'] = (df[diff_list[0]]-df_orig[diff_list[0]])/df_orig[diff_list[0]]*100
    df_diff['MID_Diff'] = (df[diff_list[1]]-df_orig[diff_list[1]])/df_orig[diff_list[1]]*100
    df_diff['PM_Diff'] = (df[diff_list[2]]-df_orig[diff_list[2]])/df_orig[diff_list[2]]*100
    df_diff['WE_Diff'] = (df[diff_list[3]]-df_orig[diff_list[3]])/df_orig[diff_list[3]]*100
    
    return df_diff
    
df_diff = diff_df(tmc_df, tmc_df_orig, 'tmc_code', lottr_list)

df_diff.describe()
df_increase = df_diff.loc[(df_diff['AM_Diff']>0) 
            | (df_diff['MID_Diff']>0)
            | (df_diff['PM_Diff']>0) 
            | (df_diff['WE_Diff']>0)]
df_increase.loc[(df_increase['AM_Diff']>1.5) 
            | (df_increase['MID_Diff']>1.5)
            | (df_increase['PM_Diff']>1.5) 
            | (df_increase['WE_Diff']>1.5)]
df_diff.loc[(df_diff['AM_Diff']>0) 
            & (df_diff['MID_Diff']>0)
            & (df_diff['PM_Diff']>0) 
            & (df_diff['WE_Diff']>0)]
df_diff.loc[(df_diff['AM_Diff']<0) 
            | (df_diff['MID_Diff']<0)
            | (df_diff['PM_Diff']<0) 
            | (df_diff['WE_Diff']<0)]
def check_random(df, df_compare):
    """
    function to check random feature,
    compare from original dataframe to cleaned dataframe,
    and produce a plot
    """
    
    # randomize feature
    import random
    i = random.randint(range(len(tmc_df['tmc_code']))[0],
                       range(len(tmc_df['tmc_code']))[-1])
    
    # get first df for figure
    df_fig = df.loc[
        df['tmc_code']==df['tmc_code'][i]
                   ][lottr_list]
    
    df_fig['tmc_code'] = df.loc[
        df['tmc_code']==df['tmc_code'][i]
                               ]['tmc_code']
    df_fig.dropna(subset=lottr_list)
    df_fig['state'] = "Cleaned"
    
    # get second df for figure
    df_fig2 = df_compare.loc[
        df_compare['tmc_code']
        ==df_compare['tmc_code'][i]
    ][lottr_list]
    
    df_fig2['tmc_code'] = df_compare.loc[
        df_compare['tmc_code']
        ==df_compare['tmc_code'][i]
    ]['tmc_code']
    df_fig2.dropna(subset=lottr_list)
    df_fig2['state'] = "Original"

    # add dataframe together and "melt" to reshape    
    df_fig = df_fig.append(df_fig2)

    df_fig_melt = df_fig.melt(id_vars=[
        'tmc_code', 'state'])
    print(df_fig_melt)

    # create plot and show
    fig = px.bar(df_fig_melt, 
                 x='variable', 
                 y='value', 
                 barmode='group', 
                 color="state", 
                 title=df_fig_melt['tmc_code'][0],
                 color_discrete_sequence=["green", "goldenrod"])
    return fig
    
check_random(tmc_df, tmc_df_orig)    
# drop all NA's from LOTTR fields
df_proj = tmc_df.dropna(subset=lottr_list)
pd.set_option('display.max_columns', None)
df_proj
# get list of field for removing columns
df_proj.columns
# make a list of fields to be removed
project_list = ['state','start_latitude', 'start_longitude', 'end_latitude',
       'end_longitude','timezone_name', 'type',
       'country', 'tmclinear','border_set', 'structype','route_sign',
       'route_qual','altrtename','isprimary',
       'active_start_date', 'active_end_date', 'LOTTR_OV']
# drop unnecessary columns
df_project = df_proj.drop(columns=project_list)
# create correlation heatmap for inspection
corr = df_project.corr()
fig_corr = px.imshow(corr)
fig_corr.show()
import datapane as dp
# host interactive visualization
r1 = dp.Report(dp.Plot(fig_corr))
r1.publish(name='Travel_Time_Project_HeatMap1', open=True)
# pare down attributes to most pertinent for project prioritization
df_proj_set = df_project.loc[(df_project["miles"]<.5) 
                          & (df_project["thrulanes"] < 3) 
                          & (df_project["aadt"]>10000)
                         & (df_project["nhs_pct"]==100)
                         & (df_project["f_system"]<4)]
# pare down LOTTR values to only view unreliable segments for project prioritization
df_proj_priority = df_proj_set.loc[(df_proj_set["LOTTR_AMP"] >=1.5) 
                | (df_proj_set["LOTTR_MID"] >=1.5) 
                | (df_proj_set["LOTTR_PM"] >=1.5) 
                | (df_proj_set["LOTTR_WE"] >=1.5)]
df_proj_priority_sub = df_proj_priority[['tmc_code', 'LOTTR_AMP', 'LOTTR_MID','LOTTR_PM','LOTTR_WE']]
df_proj_priority_sub

# host interactive visualization
r2 = dp.Report(dp.Table(df_proj_priority_sub))
r2.publish(name='Travel_Time_Project_Table', open=True)
tmc_list = list(df_proj.columns)
for i in lottr_list:
    tmc_list.remove(i)
tmc_list
# Verify correlation
corr_priority = df_proj_priority.corr()
fig_corr2 = px.imshow(corr_priority)
fig_corr2.show()
# Clarify correlation heatmap
corr3 = corr_priority[corr_priority.iloc[:,:] > 0.2]

fig = px.imshow(corr3)
fig.show()
# create and add ranking
rank = df_proj_priority.loc[:,set(lottr_list[:-1])].sum(axis=1).rank(ascending=False).astype(int)

df_proj_priority = df_proj_priority.assign(rank=rank.values)

# create subset for "melting" and visualization
df_proj_sub = df_proj_priority[['tmc_code','f_system','aadt', 'rank']]

df_proj_melt = pd.melt(df_proj_priority, id_vars=['tmc_code'], value_vars=lottr_list[:-1])
df_proj_melt_merge = pd.merge(df_proj_melt, df_proj_sub, on='tmc_code')

# visualize tmc segment project prioritization
rank_list = list(df_proj_melt_merge['rank'].values)
inverse = [1/i for i in rank_list]

fig_scatter = px.scatter(df_proj_melt_merge,x='variable', y='value',
                        color='tmc_code',
                        size=inverse,
                        color_continuous_scale='tealgrn',
                        hover_data=['tmc_code'])
fig_scatter.show()
# host interactive visualization
r3 = dp.Report(dp.Plot(fig_scatter))
r3.publish(name='Travel_Time_Project_Rank', open=True)
# scatter matrix to compare the tmc's across periods

fig_lottr = px.scatter_matrix(df_proj_priority, dimensions=lottr_list[:-1],
                        color='aadt', size="f_system",
                        color_continuous_scale='tealgrn',
                        hover_data=['tmc_code'])
fig_lottr.update_traces(diagonal_visible=False)
fig_lottr.show()

df_styled = df_diff.dropna(subset=['AM_Diff','MID_Diff','PM_Diff','WE_Diff'])
df_styled = df_styled.style.bar(subset=['AM_Diff','MID_Diff','PM_Diff','WE_Diff'], align='mid', color=['#d65f5f', '#5fba7d'])
df_styled
df_fig = tmc_df.loc[:,lottr_list]
df_fig['tmc_code'] = tmc_df['tmc_code']
df_fig = df_fig.drop(columns=['LOTTR_OV'])
df_fig = df_fig.dropna()
df_fig['state'] = "Cleaned"
df_fig
df_fig = tmc_df.loc[:,lottr_list]
df_fig['tmc_code'] = tmc_df['tmc_code']
df_fig = df_fig.drop(columns=['LOTTR_OV'])
df_fig = df_fig.dropna()
df_fig['state'] = "Cleaned"

# get second df for figure
df_fig2 = tmc_df_orig.loc[:,lottr_list]
df_fig2['tmc_code'] = tmc_df_orig['tmc_code']
df_fig2 = df_fig2.drop(columns=['LOTTR_OV'])
df_fig2 = df_fig2.dropna()
df_fig2['state'] = "Original"

# add dataframe together and "melt" to reshape    
df_fig = df_fig.append(df_fig2)

df_fig_melt = df_fig.melt(id_vars=[
    'tmc_code', 'state'])

# create plot and show
fig_melt = px.bar(df_fig_melt, 
             x='variable', 
             y='value', 
             barmode='group', 
             color="state",
             animation_frame='tmc_code',
             title=df_fig_melt['tmc_code'][0],color_discrete_sequence=["green", "goldenrod"])
fig_melt.show()
r = dp.Report(dp.Table(df_diff),
              dp.Plot(fig_melt))
r.publish(name='Travel_Time_Reliability_Outliers', open=True)