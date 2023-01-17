# for data manipulation

import numpy as np 

import pandas as pd 

import pandas_profiling as pp

pd.set_option('display.max_columns', 50)

pd.set_option('display.float_format', lambda x: '%.3f' % x)



# for date manipulation

from datetime import datetime



# for visualization: matplotlib

from matplotlib import pyplot as plt

from IPython.core.pylabtools import figsize

%matplotlib inline

# to display visuals in the notebook



# for visualization: seaborn

import seaborn as sns

sns.set_context(font_scale=2)



# for visualization: plotly

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objects as go

import plotly.express as px

import plotly.figure_factory as ff

from plotly.subplots import make_subplots

from plotly.offline import iplot



# to cleanup memory usage

import gc



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# path

path = "/kaggle/input/ashrae-energy-prediction"



# train  data

building = pd.read_csv(path + "/building_metadata.csv")

weather_train = pd.read_csv(path + "/weather_train.csv", 

                            index_col=1, parse_dates = True)

train = pd.read_csv( path + "/train.csv", 

                    index_col=2, parse_dates = True)
# look at the number of rows and columns

print('Size of the building dataset is', building.shape)

print('Size of the weather_train dataset is', weather_train.shape)

print('Size of the train dataset is', train.shape)
# test data

weather_test = pd.read_csv(path + "/weather_test.csv", 

                           index_col=1, parse_dates = True)

test = pd.read_csv(path + "/test.csv", 

                   index_col=3, parse_dates = True)

# submission data

sample_submission = pd.read_csv( path + "/sample_submission.csv")
# look at the number of rows and columns

print('Size of the weather_test dataset is', weather_test.shape)

print('Size of the test dataset is', test.shape)

print('Size of the sample_submission is', sample_submission.shape)
del sample_submission

gc.collect()
## Function to reduce the DF size

def reduce_memory_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

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
reduce_memory_usage(building)

reduce_memory_usage(weather_train)

reduce_memory_usage(train)



reduce_memory_usage(weather_test)

reduce_memory_usage(test)
pp.ProfileReport(building)
building.sort_values(by="square_feet", ascending=True).tail()
pp.ProfileReport(weather_train)
train.info()
print("Percentage of missing values in the train dataset")

train.isna().sum()
train.describe(include="all")
train.head()
test.describe(include="all")
pp.ProfileReport(weather_test)
del weather_test, test

gc.collect()
# set the plot size

figsize(12,10)



# set the histogram, mean and median

sns.distplot(train['meter_reading'],

             kde=True)

plt.axvline(x=train.meter_reading.mean(), 

            linewidth=3, color='g', label="mean", alpha=0.5)

plt.axvline(x=train.meter_reading.median(), 

            linewidth=3, color='y', label="median", alpha=0.5)



# set title, legends and labels

plt.title("Distribution of Meter Reading", size=14)

plt.legend(["mean", "median"])
# set the plot size

figsize(12,10)



# set the histogram, mean and median

sns.distplot(np.log1p(train['meter_reading']),

             kde=True)

plt.axvline(x=np.log1p(train.meter_reading.mean()), 

            linewidth=3, color='g', label="mean", alpha=0.5)

plt.axvline(x=np.log1p(train.meter_reading.median()), 

            linewidth=3, color='y', label="median", alpha=0.5)



# set title, legends and labels

plt.title("Distribution of Logarithm(Meter Reading + 1) ", size=14)

plt.legend(["mean", "median"])
# create dataframe excluding 0 measurements of meter_reading and take the natural logarithm

# np.log is used this time because we don't have 0 values in the meter_reading

positive_train = train[train['meter_reading'] != 0]

positive_train['log_meter_reading'] = np.log(positive_train['meter_reading'])
# set the plot size

figsize(12,10)



# set the histogram, mean and median

sns.distplot(positive_train['log_meter_reading'], 

             kde=True)

plt.axvline(x=positive_train['log_meter_reading'].mean(),

            linewidth=3, color='g', label="mean", alpha=0.5)

plt.axvline(x=positive_train['log_meter_reading'].median(),

            linewidth=3, color='y', label="median", alpha=0.5)



# set title, legends and labels

plt.title("Distribution of Logarithm(Meter Reading) w/o 0 Measurements", size=14)

plt.legend(["mean", "median"])
def outlier_function(df, col_name):

    ''' this function detects first and third quartile and interquartile range for a given column of a dataframe

    then calculates upper and lower limits to determine outliers conservatively

    returns the number of lower and uper limit and number of outliers respectively

    '''

    first_quartile = np.percentile(

        np.array(df[col_name].tolist()), 25)

    third_quartile = np.percentile(

        np.array(df[col_name].tolist()), 75)

    IQR = third_quartile - first_quartile

                      

    upper_limit = third_quartile+(3*IQR)

    lower_limit = first_quartile-(3*IQR)

    outlier_count = 0

                      

    for value in df[col_name].tolist():

        if (value < lower_limit) | (value > upper_limit):

            outlier_count +=1

    return lower_limit, upper_limit, outlier_count
# percentage of outliers in the meter_reading

print("{} percent of {} are outliers."

      .format((

              (100 * outlier_function(train, 'meter_reading')[2])

               / len(train['meter_reading'])),

              'meter_reading'))
train['meter_reading'].sort_values().tail()
positive_train['meter_reading'].sort_values().head()
# distribution of the meter reading in meters without zeros

figsize(12,10)



#list of different meters

meters = sorted(train['meter'].unique().tolist())



# plot meter_reading distribution for each meter

for meter_type in meters:

    subset = train[train['meter'] == meter_type]

    sns.kdeplot(np.log1p(subset["meter_reading"]), 

                label=meter_type, linewidth=2)



# set title, legends and labels

plt.ylabel("Density")

plt.xlabel("Meter_reading")

plt.legend(['electricity', 'chilled water', 'steam', 'hot water'])

plt.title("Density of Logartihm(Meter Reading + 1) Among Different Meters", size=14)
# distribution of the meter reading in meters without zeros

figsize(12,10)



# plot meter_reading distribution for each meter

for meter_type in meters:

    subset = positive_train[positive_train['meter'] == meter_type]

    sns.kdeplot(subset["log_meter_reading"], 

                label=meter_type, linewidth=2)



# set title, legends and labels

plt.ylabel("Density")

plt.xlabel("Log_meter_reading")

plt.legend(['electricity', 'chilled water', 'steam', 'hot water'])

plt.title("Density of Positive Logarithm(Meter Reading) Among Different Meters", size=14)
# upsample hourly observations to daily and aggregate by meter category

train_daily_avg_by_meter = (train.

                            groupby('meter').

                            meter_reading.

                            resample('d').mean().

                            reset_index())
# assign meter values as column headers to create tidy-form dataframe

tidy_train_daily_avg = (train_daily_avg_by_meter.

                        pivot(index='timestamp', 

                              columns='meter', 

                              values='meter_reading').

                        reset_index())
# rename column header back to meter categories

tidy_train_daily_avg.rename(columns = {0: "electricity",

                                       1: "chilled_water",

                                       2: "steam",

                                       3: "hot_water"},

                           inplace=True)
# create meter and color dictionary

meter_dict = {'electricity': 'darkblue',

              'chilled_water':'orange',

              'steam': 'green',

              'hot_water': 'red'

             }



# create figure object and plot each meter category

fig = go.Figure()



for key in meter_dict:

    fig.add_trace(go.Line(

        x=tidy_train_daily_avg.timestamp, 

        y=tidy_train_daily_avg[key], 

        mode='lines',

        name=key,

        line_color=meter_dict[key]))



# add title and show figure

fig.update_layout(

    title_text='Average Daily Energy Consumption in kWh',

    xaxis_rangeslider_visible=True)

fig.show()
# upsample weather_train dataframe to get daily means

weather_train_daily_avg = (weather_train.

                           resample('d').

                           mean())
# align weather train dataframe with the train_daily_avg dataframe

weather_train_daily_avg.reset_index(inplace=True)
weather_vs_meter_reading = (train_daily_avg_by_meter.

                            merge(weather_train_daily_avg, 

                                  on='timestamp', 

                                  how='left'))
# rename meter column

weather_vs_meter_reading['meter'] = (weather_vs_meter_reading['meter'].

                                     map({0: 'electricity',

                                          1: 'chilled_water',

                                          2: 'steam',

                                          3: 'hot_water'}))
# create weather variables and color dictionary

weather_dict = {"air_temperature": "red",

                "cloud_coverage": "orange",

                "dew_temperature": "coral",

                "precip_depth_1_hr": "olive",

                "sea_level_pressure": "teal",

                "wind_direction": "purple",

                "wind_speed": "navy" 

               }
# create plotly object and plot weather variables against dates

fig = go.Figure()

    

for key in weather_dict:

    fig.add_trace(go

                  .Line(x=weather_vs_meter_reading['timestamp'],

                        y=weather_vs_meter_reading[key], 

                        name=key,

                        line_color=weather_dict[key]))

    

fig.update_layout(title_text='Time Series of Weather Variables')

fig.show()      
# fig = ff.create_scatterplotmatrix(

#    weather_vs_meter_reading[["meter_reading",

#                              "air_temperature",

#                              "cloud_coverage",

#                              "dew_temperature",

#                              "precip_depth_1_hr",

#                              "sea_level_pressure",

#                              "wind_direction",

#                              "wind_speed",

#                              "meter"]], diag='histogram', index='meter',

#                                  height=1400, width=1400)

# fig.update_layout(

#    title='Weather Varaibles and Meter Reading',

#    dragmode='select'

#)



# fig.show()
fig = px.scatter_matrix(weather_vs_meter_reading,

                        dimensions=["meter_reading",

                                    "air_temperature",

                                    "cloud_coverage",

                                    "dew_temperature",

                                    "precip_depth_1_hr",

                                    "sea_level_pressure",

                                    "wind_direction",

                                    "wind_speed"],

                        color="meter")



fig.update_layout(

    title='Weather Varaibles and Meter Reading',

    dragmode='select',

    width=1400,

    height=1400,

    hovermode='closest')



fig.update_traces(diagonal_visible=True)

fig.show()
# group train dataset per building and meter category

train_by_building = (train.

                     groupby(["building_id", "meter"]).

                     meter_reading.mean().

                     reset_index())
# merge grouped train dataframe with building dataset

building_w_meter_reading = (train_by_building.

                            merge(building, 

                                  on='building_id', 

                                  how='left'))
# add log_meter_reading to visualize meter_reading distribution

building_w_meter_reading['log_meter_reading'] = np.log1p(building_w_meter_reading['meter_reading'])
# map primary use column 

building_w_meter_reading['primary_use_mapped'] = (building_w_meter_reading['primary_use'].

                                                  map({'Office': 'Office',

                                                        'Education': 'Education',

                                                        'Entertainment/public assembly':'Entertainment/public',

                                                        'Lodging/residential': 'Residential',

                                                        'Public services': 'Public services'

                                                       }))
# replace the rest with Other

building_w_meter_reading['primary_use_mapped'].replace(np.nan, 

                                                       'Other', 

                                                       regex=True, 

                                                       inplace=True)
building_w_meter_reading['meter'] = (building_w_meter_reading['meter'].

                                     map({0: 'electricity',

                                          1: 'chilled_water',

                                          2: 'steam',

                                          3: 'hot_water'

                                         }))
# split bıilding_w_meter_reading dataset per primary use category

education = (building_w_meter_reading[building_w_meter_reading[

    'primary_use_mapped'] == 'Education'])



office = (building_w_meter_reading[building_w_meter_reading[

    'primary_use_mapped'] == 'Office'])



entertainment_public = (building_w_meter_reading[building_w_meter_reading[

    'primary_use_mapped'] == 'Entertainment/public'])



residential = (building_w_meter_reading[building_w_meter_reading[

    'primary_use_mapped'] == 'Residential'])



public_services = (building_w_meter_reading[building_w_meter_reading[

    'primary_use_mapped'] == 'Public services'])



other = (building_w_meter_reading[building_w_meter_reading[

    'primary_use_mapped'] == 'Other'])
# create distplot parameters as lists

hist_data = [education['log_meter_reading'], 

             office['log_meter_reading'],

             entertainment_public['log_meter_reading'],

             residential['log_meter_reading'],

             public_services['log_meter_reading'],

             other['log_meter_reading']]



group_labels = ['education', 'office', 'entertainment_public',

               'residential', 'public_services', 'other' ]



colors = ['#333F44', '#37AA9C', '#94F3E4', '#66CCFF', '#2C89AB', '#0324A9']
# create KDE plot of log_meter_reading 

fig = ff.create_distplot(hist_data, group_labels, 

                         show_hist=False, colors=colors, show_rug=True)

fig.update_layout(title_text='Distribution of Logarithm Meter Reading among Primary Use')

fig.show()
# histogram of site_ids

fig = px.histogram(building, x="site_id")

fig.update_layout(title_text='Distribution Site IDs')

fig.show()
# create site id list

site_ids = building_w_meter_reading.site_id.unique().tolist()



# create plotly object and visualize the distribution

fig = go.Figure()



# add a violin plot for each site_id

for site_id in site_ids:

    fig.add_trace(go.Violin(y=building_w_meter_reading

                            [building_w_meter_reading['site_id'] == site_id]

                            ['log_meter_reading'],

                            name=site_id,

                            box_visible=True))



# set title and show the object

fig.update_layout(title_text='Distribution of Logarithm Meter Reading among Site ID')

fig.show()
fig = px.scatter(building_w_meter_reading, x="square_feet", y="log_meter_reading", 

                 color="meter", hover_data=['meter_reading'])



fig.update_layout(title_text='Meter Reading VS Square Feet Among Different Meters')

fig.show()
currentYear = datetime.now().year

building_w_meter_reading['age'] = currentYear - building_w_meter_reading['year_built']
fig = px.scatter(building_w_meter_reading, x="age", y="log_meter_reading",

                 color="meter", hover_data=['meter_reading'])



fig.update_layout(title_text='Meter Reading VS Age of the Building Among Different Meters')

fig.show()
fig = px.scatter(building_w_meter_reading, x="floor_count", y="log_meter_reading", 

                 color="meter", hover_data=['meter_reading'])



fig.update_layout(title_text='Meter Reading VS Floor Count Among Different Meters')

fig.show()