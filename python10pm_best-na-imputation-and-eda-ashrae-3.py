import datetime

import gc

import warnings

warnings.filterwarnings("ignore")



# several prints in one cell

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"





import numpy as np

import pandas as pd

import pandas_profiling

# don't truncate the DF

pd.set_option("display.max_columns", None)



from sklearn.preprocessing import LabelEncoder

from sklearn.impute import SimpleImputer



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        file_name = os.path.join(dirname, filename)

        print(file_name)

        

train_df = pd.read_csv("/kaggle/input/ashrae-energy-prediction/train.csv")

train_df.shape

train_df.head()



weather_df = pd.read_csv("/kaggle/input/ashrae-energy-prediction/weather_train.csv")

weather_df.shape

weather_df.head()



building_metadata_df = pd.read_csv("/kaggle/input/ashrae-energy-prediction/building_metadata.csv")

building_metadata_df.shape

building_metadata_df.head()

def report_nulls(df):

    '''

    Show a fast report of the DF.

    '''

    rows = df.shape[0]

    columns = df.shape[1]

    null_cols = 0

    list_of_nulls_cols = []

    for col in list(df.columns):

        null_values_rows = df[col].isnull().sum()

        null_rows_pcn = round(((null_values_rows)/rows)*100, 2)

        col_type = df[col].dtype

        if null_values_rows > 0:

            print("The column {} has {} null values. It is {}% of total rows.".format(col, null_values_rows, null_rows_pcn))

            print("The column {} is of type {}.\n".format(col, col_type))

            null_cols += 1

            list_of_nulls_cols.append(col)

    null_cols_pcn = round((null_cols/columns)*100, 2)

    print("The DataFrame has {} columns with null values. It is {}% of total columns.".format(null_cols, null_cols_pcn))

    return list_of_nulls_cols
print("Reporting the train DataFrame")

list_of_null_cols = report_nulls(train_df)

print()

print("Reporting the weather DataFrame")

list_of_null_cols = report_nulls(weather_df)

print()

print("Reporting the building metadata DataFrame")

list_of_null_cols = report_nulls(building_metadata_df)
# as we can see, the main DataFrame has no null values. The weather data contains null values in 77,78% of the columns, with the colum "cloud_coverage" and "precip_depth_1_hr" with most nulls.

# On the other side, the building metadata has a lot of null in the colum "year_built" and "floor_count"
plt.figure(figsize=(20, 10))

sns.heatmap(weather_df.isnull(), cbar = False, cmap = "inferno")

plt.title("Viz of null value for the weather DataFrame.");
plt.figure(figsize=(20, 10))

sns.heatmap(building_metadata_df.isnull(), cbar = False, cmap = "inferno")

plt.title("Viz of null value for the building metada DataFrame.");
gc.collect()

weather_df.sort_values(["site_id", "timestamp"], inplace = True)

weather_df.head()
# Plotting part

# Let's define some helper functions

def plot_distribution(x, y, title, xlabel, ylabel, flag = False):

    '''

    Function that help to plot the distribution of some variables and expects 5 arguments:

        x: data for x axis.

        y: data for y axis.

        title: the title you want for our plot

        xlabel: the message you want to put to the x axis

        ylabel: the message you want to put to the y axis

    --------------------------------------------------------------------------------------

    Plots the data and annotates the max, min, median and mean.

    '''

    if flag:

        y_mean = [np.mean(y) for i in range(len(y))]

        y_median = [np.median(y) for i in range(len(y))]



        # beautiful color pallets

        # http://everyknightshoulddesign.blogspot.com/2013/08/beautiful-color-palettes-their-hex-codes.html

        colors = "#0F5959   #17A697   #638CA6   #8FD4D9   #D93240".split()



        # basic plot

        plt.figure(figsize=(15,7)) # define the size of the plot

        plt.plot(x, y, color = colors[0])

        plt.plot(x, y_mean, color = colors[1])

        plt.plot(x, y_median, color = colors[4])

        if np.min(y) > 0:

            plt.ylim((np.min(y) - np.min(y)*-0.5) , np.max(y)*1.5)

        if np.min(y) == 0:

            plt.ylim((np.max(y)*-1) , np.max(y)*1.5)

        plt.title(title)

        plt.xlabel(xlabel)

        plt.ylabel(ylabel)



        # Using annotation to help understand and format a little the plot.



        plt.annotate('Mean value is {}'.format(np.mean(y)), # Message you want to put on the plot

                     xy=(150, np.mean(y)), # Coordinates were the arrows will point

                     xycoords='data', # 

                     xytext=(0.2, 0.90), # position of the text

                     textcoords='axes fraction', # specify the scale. If fraction the xytext has to be between 0 - 1

                     color = colors[1], # colors we have defined previously

                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color = colors[1])) # parameters for the arrow



        plt.annotate('Median value is {}'.format(np.median(y)),

                     xy=(250, np.median(y)),

                     xycoords='data',

                     xytext=(0.7, 0.90), 

                     textcoords='axes fraction', 

                     color = colors[4],

                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color = colors[4])) 



        # calculate the max and min to determine the index of the value so that we can dinamically plot them

        # on different charts

        max_y = np.max(y)

        #index_max = y.index(max_y)



        # getting the value for the max value from a numpy array.

        index_max = np.where(y == max_y)[0][0]



        min_y = np.min(y)

        #index_min = y.index(min_y)

        # getting the value for the min value from a numpy array.

        index_min = np.where(y == min_y)[0][0]



        plt.annotate('Max value is {}'.format(np.max(y)),

                     xy=(index_max, max_y),

                     xycoords='data',

                     xytext=(25, -25), 

                     textcoords='offset points',

                     arrowprops=dict(arrowstyle="->"))



        plt.annotate('Min value is {}'.format(np.min(y)),

                     xy=(index_min, min_y),

                     xycoords='data',

                     xytext=(-25, 25), 

                     textcoords='offset points',

                     arrowprops=dict(arrowstyle="->"));

    



# More on annotations

# https://matplotlib.org/3.1.1/gallery/text_labels_and_annotations/annotation_demo.html
def preprocess_before_plot(column, weather_df, flag):

    

    # we have some null values that don't allow us to plot the data

    # let's impute the mean value for the beggining

    si = SimpleImputer(missing_values=np.nan, strategy = "mean")

    weather_df[column] = si.fit_transform((weather_df[column].values).reshape(-1, 1))



    for site_id in weather_df["site_id"].unique():

        x = weather_df[weather_df["site_id"] == site_id]["timestamp"].values

        # plotting timeseries takes a lot so we will instead create an array that represents every hour that has passed since the first register

        # in the future I plan to change and plot the dates

        x_ = [i for i in range(len(x))]

        y = weather_df[weather_df["site_id"] == site_id][column].values

        title = "Plot of {} for the Site_ID {}".format(column, site_id)

        xlabel = "Time Range"

        ylabel = column

        plot_distribution(x_, y, title, xlabel, ylabel, flag)
columns = ["air_temperature", 

           #"cloud_coverage", # gives some problems with blank plots

           "dew_temperature", 

           #"sea_level_pressure", # gives some problems with blank plots

           "wind_direction", 

           "wind_speed"]



for column in columns:

    flag = True

    if flag:

        print("Plotting the data for column {}".format(column))

        preprocess_before_plot(column, weather_df, flag);
group_by = "primary_use"

numeric_column = "square_feet"

pb = building_metadata_df[[group_by, numeric_column]].pivot_table(index = group_by, aggfunc = np.mean).sort_values(numeric_column, ascending = False)



# inverse the order

x = list(pb.index)[::-1]

y = list(pb[numeric_column].values)[::-1]



plt.figure(figsize=(20, 10))

plt.barh(x, y,  align='center')

plt.title("Mean square feet by type of building")

plt.xlabel("Mean square feet")

plt.ylabel("Building type")

plt.legend();
group_by = "year_built"

numeric_column = "square_feet"

pb = building_metadata_df[[group_by, numeric_column]].pivot_table(index = group_by, aggfunc = np.mean).sort_values(numeric_column, ascending = False)



# inverse the order

x = list(pb.index)[::-1]

y = list(pb[numeric_column].values)[::-1]



plt.figure(figsize=(20, 10))

plt.barh(x, y,  align='center')

plt.title("Mean square feet by year built")

plt.xlabel("Mean square feet")

plt.ylabel("Year built")

plt.legend();
group_by = "primary_use"

numeric_column = "floor_count"

pb = building_metadata_df[[group_by, numeric_column]].pivot_table(index = group_by, aggfunc = np.mean).sort_values(numeric_column, ascending = False)



# inverse the order

x = list(pb.index)[::-1]

y = list(pb[numeric_column].values)[::-1]



plt.figure(figsize=(20, 10))

plt.barh(x, y,  align='center')

plt.title("Number of floors by year built")

plt.xlabel("Number of floors")

plt.ylabel("Year built")

plt.legend();
group_by = "year_built"

numeric_column = "floor_count"

pb = building_metadata_df[[group_by, numeric_column]].pivot_table(index = group_by, aggfunc = np.mean).sort_values(numeric_column, ascending = False)



# inverse the order

x = list(pb.index)[::-1]

y = list(pb[numeric_column].values)[::-1]



plt.figure(figsize=(20, 10))

plt.barh(x, y,  align='center')

plt.title("Number of floors by year built")

plt.xlabel("Number of floors")

plt.ylabel("Year built")

plt.legend();
InteractiveShell.ast_node_interactivity = "last_expr"




def year_grouper(value):

    if value < 1920:

        return "[1900 - 1920)"

    elif value < 1940:

        return "[1920 - 1940)"

    elif value < 1960:

        return "[1940 - 1960)"

    elif value < 1980:

        return "[1960 - 1980)"

    elif value < 2000:

        return "[1980 - 2000)"

    elif value < 2020:

        return "[2000 - 2020)"

    else:

        return "NaN"



building_metadata_df["year_grouped"] = building_metadata_df["year_built"].apply(year_grouper)

building_metadata_df.head()



pb = building_metadata_df[["year_grouped", "primary_use", "square_feet"]]

pb = pb.pivot_table(index = ["year_grouped", "primary_use"], values = "square_feet", aggfunc = np.mean, fill_value = 0)[::-1]



pb = pd.DataFrame(pb.unstack(level = 1)).fillna(0)



# huge thanks to pratapvardhan for his code

# https://github.com/pratapvardhan/notebooks/blob/master/barchart-race-matplotlib.ipynb



groups = pb.index

# extract then n-th element from the multiindex

y_names = [name[1] for name in list(pb.columns)]



# list of colors

# beautiful color pallets

# http://everyknightshoulddesign.blogspot.com/2013/08/beautiful-color-palettes-their-hex-codes.html

list_of_colors = "#588C73 #F2E394 #F2AE72 #D96459 #8C4646 #BCCF02 #5BB12F #73C5FF #9B539C #EB65A0 #260126 #59323C #F2EEB3 #BFAF80 #8C6954 #3B3A35".split()



# dictionary comprehension to assing each building type a color

name_color_dict ={name:color for (name, color) in zip(y_names, list_of_colors)}



for group in groups:

    y_values = pb[pb.index == group].values[0]

    # plot the data

    fig, ax = plt.subplots(figsize=(20, 10))

    

    ax.barh(y_names, y_values, color = list_of_colors)

    

    for i, (value, name) in enumerate(zip(y_values, y_names)):

        #ax.text(value, i,     name,            ha='right')

        #ax.text(value, i-.25, name_color_dict[name],  ha='right')

        ax.text(value,  i,     "{:,}".format(round(value, 2)).replace(",", "."), ha='left')

    

    ax.text(1, 0.4, group, transform=ax.transAxes, size=20, ha='right')

    plt.title("The mean square feet of the building for the ´{} period".format(group));
def year_grouper(value):

    if value < 1920:

        return "[1900 - 1920)"

    elif value < 1940:

        return "[1920 - 1940)"

    elif value < 1960:

        return "[1940 - 1960)"

    elif value < 1980:

        return "[1960 - 1980)"

    elif value < 2000:

        return "[1980 - 2000)"

    elif value < 2020:

        return "[2000 - 2020)"

    else:

        return "NaN"



building_metadata_df["year_grouped"] = building_metadata_df["year_built"].apply(year_grouper)

building_metadata_df.head()



pb = building_metadata_df[["year_grouped", "primary_use", "floor_count"]]

pb = pb.pivot_table(index = ["year_grouped", "primary_use"], values = "floor_count", aggfunc = np.mean, fill_value = 0)[::-1]



pb = pd.DataFrame(pb.unstack(level = 1)).fillna(0)



# huge thanks to pratapvardhan for his code

# https://github.com/pratapvardhan/notebooks/blob/master/barchart-race-matplotlib.ipynb



groups = pb.index

# extract then n-th element from the multiindex

y_names = [name[1] for name in list(pb.columns)]



# list of colors

# beautiful color pallets

# http://everyknightshoulddesign.blogspot.com/2013/08/beautiful-color-palettes-their-hex-codes.html

list_of_colors = "#588C73 #F2E394 #F2AE72 #D96459 #8C4646 #BCCF02 #5BB12F #73C5FF #9B539C #EB65A0 #260126 #59323C #F2EEB3 #BFAF80 #8C6954 #3B3A35".split()



# dictionary comprehension to assing each building type a color

name_color_dict ={name:color for (name, color) in zip(y_names, list_of_colors)}



for group in groups:

    y_values = pb[pb.index == group].values[0]

    # plot the data

    fig, ax = plt.subplots(figsize=(20, 10))

    

    ax.barh(y_names, y_values, color = list_of_colors)

    

    for i, (value, name) in enumerate(zip(y_values, y_names)):

        #ax.text(value, i,     name,            ha='right')

        #ax.text(value, i-.25, name_color_dict[name],  ha='right')

        ax.text(value,  i,     "{:,}".format(round(value, 2)).replace(",", "."), ha='left')

    

    ax.text(1, 0.4, group, transform=ax.transAxes, size=20, ha='right')

    plt.title("Average floors of the building for the ´{} period".format(group));
weather_df.head()
# several prints in one cell

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
weather_df["YYYYMMDD"] = weather_df["timestamp"].apply(lambda str_time: datetime.datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d'))

weather_df["YYYYMM"] = weather_df["timestamp"].apply(lambda str_time: datetime.datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m'))
weather_df.sort_values(["site_id", "timestamp"], ascending = True, inplace = True)
##########################################################################################################

plt.figure(figsize=(20, 10))

sns.heatmap(weather_df.isnull(), cbar = False, cmap = "inferno")

plt.title("Viz of null value for the weather DataFrame.");

columns = report_nulls(weather_df)



# First we will forward and back propagate the the missing values for our null columns by site_id and 

# timestamp by YYYYMMDD



grouper_list = ["site_id", "YYYYMMDD"]

weather_df[columns] = weather_df.groupby(grouper_list)[columns].transform(lambda x: x.ffill().bfill())



##########################################################################################################



##########################################################################################################



plt.figure(figsize=(20, 10))

sns.heatmap(weather_df.isnull(), cbar = False, cmap = "inferno")

plt.title("Viz of null value for the weather DataFrame.");

columns = report_nulls(weather_df)



# If null values remain, we will forward and back propagate the the missing values for our null columns by site_id and 

# timestamp by YYYYMM



grouper_list = ["site_id", "YYYYMM"]

weather_df[columns] = weather_df.groupby(grouper_list)[columns].transform(lambda x: x.ffill().bfill())

# weather_df[columns] = weather_df.groupby(grouper_list)[columns].transform(lambda x: x.mean())



##########################################################################################################



##########################################################################################################



plt.figure(figsize=(20, 10))

sns.heatmap(weather_df.isnull(), cbar = False, cmap = "inferno")

plt.title("Viz of null value for the weather DataFrame.");

columns = report_nulls(weather_df)



# If null values remain, we will forward and back propagate the the missing values for our null columns by site_id 



columns = report_nulls(weather_df)

grouper_list = ["site_id"]

weather_df[columns] = weather_df.groupby(grouper_list)[columns].transform(lambda x: x.ffill().bfill())



##########################################################################################################
plt.figure(figsize=(20, 10))

sns.heatmap(weather_df.isnull(), cbar = False, cmap = "inferno")

plt.title("Viz of null value for the weather DataFrame.");

columns = report_nulls(weather_df)
building_metadata_df


##########################################################################################################



plt.figure(figsize=(20, 10))

sns.heatmap(building_metadata_df.isnull(), cbar = False, cmap = "inferno")

plt.title("Viz of null value for the weather DataFrame.");

columns = report_nulls(building_metadata_df)



# If null values remain, we will forward and back propagate the the missing values for our null columns by site_id and 

# timestamp by YYYYMM



grouper_list = ["site_id", "primary_use", "year_grouped"]

building_metadata_df[columns] = building_metadata_df.groupby(grouper_list)[columns].transform(lambda x: x.ffill().bfill())



##########################################################################################################



##########################################################################################################



plt.figure(figsize=(20, 10))

sns.heatmap(building_metadata_df.isnull(), cbar = False, cmap = "inferno")

plt.title("Viz of null value for the weather DataFrame.");

columns = report_nulls(building_metadata_df)



# If null values remain, we will forward and back propagate the the missing values for our null columns by site_id and 

# timestamp by YYYYMM



grouper_list = ["primary_use"]

building_metadata_df[columns] = building_metadata_df.groupby(grouper_list)[columns].transform(lambda x: x.ffill().bfill())



##########################################################################################################



##########################################################################################################



plt.figure(figsize=(20, 10))

sns.heatmap(building_metadata_df.isnull(), cbar = False, cmap = "inferno")

plt.title("Viz of null value for the weather DataFrame.");

columns = report_nulls(building_metadata_df)



# If null values remain, we will forward and back propagate the the missing values for our null columns by site_id and 

# timestamp by YYYYMM



grouper_list = ["primary_use"]

building_metadata_df[columns] = building_metadata_df.groupby(grouper_list)[columns].transform(lambda x: x.ffill().bfill())



##########################################################################################################
