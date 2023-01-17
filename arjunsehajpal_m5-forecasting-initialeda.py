# importing the libraries



import os

import math

import numpy as np                                 # linear algebra

import pandas as pd                                # dataframes

import matplotlib.pyplot as plt                    # visualizations

import seaborn as sns

import ipywidgets as widgets                       # interative jupyter

from IPython.display import clear_output



from scipy import stats                            # statistics

from datetime import datetime, date, timedelta     # time

from dateutil.relativedelta import relativedelta

from statsmodels.tsa.seasonal import STL           # time-series decomposition



from matplotlib.patches import Polygon
# setting up the notebook parameters



root_dir = "/kaggle/input/m5-forecasting-accuracy"

print("root directory = {}".format(root_dir))



plt.rcParams["figure.figsize"] = (16, 8)

sns.set_style("darkgrid")

pd.set_option("display.max_rows", 20, "display.max_columns", None)
######################################

######## Helper Functions ############

######################################





def info_df(df):

    """

    returns the dataframe describing nulls and unique counts

    inp: dataframe

    returns: dataframe with unique and null counts

    """

    return pd.DataFrame({

        "uniques": df.nunique(),

        "nulls": df.isnull().sum(),

        "nulls (%)": df.isnull().sum() / len(df)

    }).T





def reduce_mem_usage(df, verbose=True):

    """

    reduces the mem usage by performing certain coercion operations

    inp: dataframe,

         verbose (whether to print the info regarding mem reduction or Not)

    returns: dataframe

    """

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
# defining the datasets

path_calendar_df = os.path.join(root_dir, "calendar.csv")

path_sales_train_validation_df = os.path.join(root_dir, "sales_train_validation.csv")

path_sell_prices_df = os.path.join(root_dir, "sell_prices.csv")

path_sample_submission_df = os.path.join(root_dir, "sample_submission.csv")



# importing the dataset

calendar_df = pd.read_csv(path_calendar_df, parse_dates = ["date"])

sales_sample_df = pd.read_csv(path_sales_train_validation_df)

sell_prices_df = pd.read_csv(path_sell_prices_df)

sample_submission_df = pd.read_csv(path_sample_submission_df)



# optimising the mem usage

calendar_df = reduce_mem_usage(calendar_df, verbose = True)

sales_sample_df = reduce_mem_usage(sales_sample_df, verbose = True)

sell_prices_df = reduce_mem_usage(sell_prices_df, verbose = True)

sample_submission_df = reduce_mem_usage(sample_submission_df, verbose = True)
calendar_df.head(5)
info_df(calendar_df)
sales_sample_df.head()
info_df(sales_sample_df)
sell_prices_df.head()
info_df(sell_prices_df)
sample_submission_df.head()
# # create a smaller sample of sales_train_validation_df, to comply with memory demands

# sales_sample_df = sales_sample_df.sample(100)    
print("Final len of the Dataset after unpivoting would be = {}".format(sales_sample_df.shape[0] * 1913))



# for unpivoting, we need to define the variables into two sets, id_vars and value_vars

value_variables = [col for col in sales_sample_df.columns if col.startswith("d_")]

identifier_variables = [col for col in sales_sample_df.columns if col not in value_variables] 



# converting the df from wide to long

sales_sample_df = pd.melt(sales_sample_df, 

                          id_vars = identifier_variables, 

                          value_vars = value_variables)



print("Actual Shape after unpivoting = {}".format(sales_sample_df.shape))



# changing the variable name to apt names

sales_sample_df = sales_sample_df.rename(columns = {"variable": "day_number", "value": "units_sold"})
# creating a date column

earliest_date = date(2011, 1, 29)

date_dict = {}                    # a dictionary to map the day_number values to real dates

for i in list(sales_sample_df["day_number"].unique()):

    dn_int = int(i[2:]) - 1                                   # indexing the string value to delete "d_" from the day_number and converting it to int

                                                              # subtracting 1 because "d_1" would be our zeroth day. 

    date_ = earliest_date + timedelta(days = dn_int)

    date_dict[i] = date_



# mapping the dictionary to dataframe

sales_sample_df["date"] = sales_sample_df["day_number"].map(date_dict)

sales_sample_df["date"] = pd.to_datetime(sales_sample_df["date"])
sales_sample_df.head()
ALL = "ALL"

def unique_sorted_value_fn(array):

    """

    returns unique sorted values

    inp: array

    return array with unique values

    """

    unique_arr = array.unique().tolist()

    unique_arr.sort()

    unique_arr.insert(0, ALL)   # if all values are to be selected

    return unique_arr



# initialize the dropdown

dropdown_item_id = widgets.Dropdown(options = unique_sorted_value_fn(sales_sample_df["item_id"]))



item_id_plot = widgets.Output()



def dropdown_item_id_eventhandler(change):

    item_id_plot.clear_output()

    with item_id_plot:

        if (change.new == ALL):

            display(sns.lineplot(x = "date", y = "units_sold", hue = "item_id", data = sales_sample_df))

        else:

            display(sns.lineplot(x = "date", y = "units_sold", hue = "item_id", 

                                 data = sales_sample_df[sales_sample_df["item_id"] == change.new]))

            plt.show()

            

dropdown_item_id.observe(dropdown_item_id_eventhandler, names='value')
display(dropdown_item_id)
display(item_id_plot)
sns.lineplot(x = "date", y = "units_sold", data = sales_sample_df)

plt.title("M5 - aggregated data")

plt.show()
sns.lineplot(x = "date", y = "units_sold", hue = "state_id", data = sales_sample_df)

plt.title("State wise aggregated data")

plt.show()
# creating a new dataframe for statewise aggregation

statewise_df = sales_sample_df.groupby(["state_id", "date"]).agg({

    "units_sold": "sum"

}).reset_index()



# extracting month and year from date for group by purposes

statewise_df["day"] = statewise_df["date"].dt.day

statewise_df["month"] = statewise_df["date"].dt.month

statewise_df["year"] = statewise_df["date"].dt.year



# aggregating on month level for each state

statewise_df = statewise_df.groupby(["month", "year", "state_id"]).agg({

    "units_sold": "sum", 

    "day": "first"

}).reset_index()



statewise_df["date"] = pd.to_datetime(statewise_df["year"].astype("str") + "-" + \

                                      statewise_df["month"].astype("str") + "-" + \

                                      statewise_df["day"].astype("str"))
sns.lineplot(x = "date", y = "units_sold", hue = "state_id", data = statewise_df)

plt.title("Statewise sales trend")

plt.show()
del statewise_df
sales_sample_df.groupby("state_id").agg({"store_id": "nunique"})
# creating a new dataframe for storewise aggregation

storewise_df = sales_sample_df.groupby(["state_id", "store_id", "date"]).agg({

    "units_sold": "sum"

}).reset_index()



# extracting month and year from date for group by purposes

storewise_df["day"] = storewise_df["date"].dt.day

storewise_df["month"] = storewise_df["date"].dt.month

storewise_df["year"] = storewise_df["date"].dt.year



# aggregating on month level for each state and store

storewise_df = storewise_df.groupby(["month", "year", "state_id", "store_id"]).agg({

    "units_sold": "sum", 

    "day": "first"

}).reset_index()



storewise_df["date"] = pd.to_datetime(storewise_df["year"].astype("str") + "-" + \

                                      storewise_df["month"].astype("str") + "-" + \

                                      storewise_df["day"].astype("str"))
state_list = list(storewise_df["state_id"].unique())

for i in range(1, 4):

    plt.subplot(3, 1, i)

    sns.lineplot(x = "date", 

                 y = "units_sold", 

                 hue = "store_id", 

                 data = storewise_df[storewise_df["state_id"] == state_list[i - 1]])

    plt.title("Store wise trend in {}".format(state_list[i - 1]))

    plt.show()
del storewise_df
# creating a new dataframe for storewise aggregation

catwise_df = sales_sample_df.groupby(["state_id", "cat_id", "date"]).agg({

    "units_sold": "sum"

}).reset_index()



# extracting month and year from date for group by purposes

catwise_df["day"] = catwise_df["date"].dt.day

catwise_df["month"] = catwise_df["date"].dt.month

catwise_df["year"] = catwise_df["date"].dt.year



# aggregating on month level for each state and store

catwise_df = catwise_df.groupby(["month", "year", "state_id", "cat_id"]).agg({

    "units_sold": "sum", 

    "day": "first"

}).reset_index()



catwise_df["date"] = pd.to_datetime(catwise_df["year"].astype("str") + "-" + \

                                    catwise_df["month"].astype("str") + "-" + \

                                    catwise_df["day"].astype("str"))
sns.lineplot(x = "date", y = "units_sold", hue = "cat_id", data = catwise_df)

plt.title("Catgory-wise Sales")

plt.show()
for i in range(1, 4):

    plt.subplot(3, 1, i)

    sns.lineplot(x = "date", 

                 y = "units_sold", 

                 hue = "cat_id", 

                 data = catwise_df[catwise_df["state_id"] == state_list[i - 1]])

    plt.title("Category wise trend in {}".format(state_list[i - 1]))

    plt.show()
sales_sample_df.groupby("cat_id").agg({"dept_id": "nunique"})
# creating a new dataframe for storewise aggregation

deptwise_df = sales_sample_df.groupby(["cat_id", "dept_id", "date"]).agg({

    "units_sold": "sum"

}).reset_index()



# extracting month and year from date for group by purposes

deptwise_df["day"] = deptwise_df["date"].dt.day

deptwise_df["month"] = deptwise_df["date"].dt.month

deptwise_df["year"] = deptwise_df["date"].dt.year



# aggregating on month level for each state and store

deptwise_df = deptwise_df.groupby(["month", "year", "cat_id", "dept_id"]).agg({

    "units_sold": "sum", 

    "day": "first"

}).reset_index()



deptwise_df["date"] = pd.to_datetime(deptwise_df["year"].astype("str") + "-" + \

                                     deptwise_df["month"].astype("str") + "-" + \

                                     deptwise_df["day"].astype("str"))
cat_list = list(sales_sample_df["cat_id"].unique())

for i in range(1, 4):

    plt.subplot(3, 1, i)

    sns.lineplot(x = "date", 

                 y = "units_sold", 

                 hue = "dept_id", 

                 data = deptwise_df[deptwise_df["cat_id"] == cat_list[i - 1]])

    plt.title("Category wise trend in {}".format(cat_list[i - 1]))

    plt.show()
# creating a new dataframe with aggregated sales.

stl_df = sales_sample_df[["date", "units_sold"]].set_index("date")

stl_df = stl_df.resample("D").sum()

stl_df.head()
stl = STL(stl_df, seasonal = 7)

res = stl.fit()

fig = res.plot()
del stl_df
sales_sample_df["day_of_week"] = sales_sample_df["date"].dt.weekday

sales_sample_df["month"] = sales_sample_df["date"].dt.month

sales_sample_df["year"] = sales_sample_df["date"].dt.year
week_month_pivot = sales_sample_df.pivot_table(index = "day_of_week", 

                                               columns = "month", 

                                               values = "units_sold", 

                                               aggfunc = "sum")



week_month_pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

week_month_pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]



sns.heatmap(week_month_pivot, linewidth = 0.2, cmap="YlGnBu")

plt.title("Performance of sales for Day of Week aggregated on monthly basis")

plt.show()
year_month_pivot = sales_sample_df.pivot_table(index = "month", 

                                               columns = "year", 

                                               values = "units_sold", 

                                               aggfunc = "sum")

year_month_pivot.index = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]



sns.heatmap(year_month_pivot, linewidth = 0.2, cmap="YlGnBu")

plt.title("Performance of sales for Month aggregated on yearly basis")

plt.show()
del week_month_pivot, year_month_pivot
temp_list = ["day_of_week", "month"]

for i in range(0, 2):

    plt.subplot(2, 1, i + 1)

    tempdf = sales_sample_df.groupby([temp_list[i], "state_id"]).agg({

        "units_sold": "mean"

    }).reset_index()

    sns.lineplot(x = temp_list[i], y = "units_sold", hue = "state_id", data = tempdf)

    plt.title("units sold trend - {}".format(temp_list[i]))

    plt.show()
calendar_df.head()
event1_bool = []   # boolean list. Captures whether an event exist or not

for i in range(0, len(calendar_df)):

    if calendar_df["event_name_1"].iloc[i] == calendar_df["event_name_1"].iloc[i]:

        event1_bool.append("True")

    else:

        event1_bool.append("False")

        

# inserting the above list in calendar_df

calendar_df.insert(loc = 9, column = "event_bool_1", value = event1_bool)
# plot distribution of event days

plt.subplot(1, 2, 1)

sns.countplot(calendar_df["event_bool_1"], palette = "Set2")

plt.title("Frequency of events and non-events")

plt.xlabel("Whether event is there or Not")



plt.subplot(1, 2, 2)

sns.countplot(y = calendar_df["event_type_1"], palette = "Set2")

plt.title("Frequency of the types of events")

plt.ylabel("Event Type")



plt.show()
# plot distribution of snap days across three states

plt.subplot(1, 3, 1)

sns.countplot(x = calendar_df["snap_CA"], palette = "RdBu")

plt.title("Frequency plot - Snap days across California")



plt.subplot(1, 3, 2)

sns.countplot(x = calendar_df["snap_TX"], palette = "RdBu")

plt.title("Frequency plot - Snap days across Texas")



plt.subplot(1, 3, 3)

sns.countplot(x = calendar_df["snap_WI"], palette = "RdBu")

plt.title("Frequency plot - Snap days across Wisconsin")



plt.show()
def generate_data(df, date_col, data_col):

    """

    converts the pd dataframe into numpy arrays of data and respective dates

    

    inp: df (dataframe)

         date_col (column which contains the dates)

         data_col (data to be mapped)

    returns: data_arr (array of data)

             dates (array of dates)

    """

    data_arr = np.array(df[data_col])

    data_len = len(data_arr)

    start_date = df[date_col].iloc[0]

    dates = [start_date + timedelta(days = i) for i in range(data_len)]

    return data_arr, dates





def calendar_array_fn(date_arr, data_arr):

    """

    returns an array of shape (-1, 7)

    

    inp: date_arr (array of dates)

         data_arr (array of data)

    returns: i, j (indices)

             calendar (array of data of shape (-1, 7))

    """

    

    # return the date as an ISO calendar (year, week, day)

    i, j = zip(*[date.isocalendar()[1:] for date in date_arr])

    i = np.array(i) - min(i) 

    j = np.array(j) - 1

    max_i = max(i) + 1

    calendar = np.nan * np.zeros((max_i, 7))  # creating empty arrays

    calendar[i, j] = data_arr                 # creating a data matrix

    

    return i, j, calendar





def label_days(ax, date_arr, i, j, calendar):

    """

    creates label for days

    

    inp: ax,

         date_arr (array of dates),

         i, j (indices),

         calendar (calendar array, returned by calendar_arr_fn())

    returns: nothing

    """

    ni, nj = calendar.shape                          # len and width of the matrix

    day_of_month_arr = np.nan * np.zeros((ni, 7))    # initializing day_of_month array

    day_of_month_arr[i, j] = [date.day for date in date_arr]

    

    # ndenuerate - multi index iterator

    for (i, j), day in np.ndenumerate(day_of_month_arr):

        # following condition checks if the thing is not NaN

        if np.isfinite(day):

            ax.text(j, i, 

                    int(day), 

                    ha = "center", 

                    va = "center")

    # defining x-axis labels

    ax.set(xticks = np.arange(7), 

           xticklabels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

    ax.xaxis.tick_top()

    

    

def label_months(ax, date_arr, i, j, calendar):

    """

    creates label for days

    

    inp: ax,

         date_arr (array of dates),

         i, j (indices),

         calendar (calendar array, returned by calendar_arr_fn())

    returns: nothing

    """

    months_labels = np.array([

        "Jan", "Feb", "Mar", "Apr", "May", "Jun", 

        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"

    ])                                                        # month names

    months_arr = np.array([date.month for date in date_arr])     # extracting months from dates

    unique_months = sorted(set(months_arr))                   # get unique months                  

    yticks = [i[months_arr == m].mean() for m in unique_months]   

    labels = [months_labels[m - 1] for m in unique_months]

    ax.set(yticks = yticks)

    ax.set_yticklabels(labels, rotation = 90)

    

    

def calendar_heatmap(ax, date_arr, data_arr):

    i, j, calendar = calendar_array_fn(date_arr, data_arr)

    im = ax.imshow(calendar, interpolation = "none", cmap = "summer")

    label_days(ax, date_arr, i, j, calendar)

    label_months(ax, date_arr, i, j, calendar)

    # uncomment following line if you want colorbars

    #ax.figure.colorbar(im)         

    

def plot_calmap(ax, year, data_col):

    """

    main function for ploting calendar heatmaps

    

    inp: year (which year to be plotted)

         data_col (data column)

    returns nothing

    """

    data_arr, date_arr = generate_data(df = calendar_df[calendar_df["year"] == year], 

                                       date_col = "date", 

                                       data_col = data_col)

    calendar_heatmap(ax, date_arr, data_arr)

    ax.set_title("{} distribution in the year {}".format(data_col, year))
fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (19, 20))

plot_calmap(ax[0], year = 2011, data_col = "snap_CA")

plot_calmap(ax[1], year = 2011, data_col = "snap_TX")

plot_calmap(ax[2], year = 2011, data_col = "snap_WI")
sell_prices_df.head()
# creating a few additional columns to aid in analysis below



sell_prices_df["state"] = sell_prices_df["store_id"].str[:2]

sell_prices_df["cat_id"] = sell_prices_df["item_id"].str[:-4]
# plotting the distribution of various stores in a state



plt.figure(figsize = (20, 16))

plt.subplots_adjust(hspace = 0.5)



plt.subplot(4, 3, 1)

for i in ["CA_1", "CA_2", "CA_3", "CA_4"]:

    sns.distplot(sell_prices_df[sell_prices_df["store_id"] == i]["sell_price"], label = i)

    plt.legend()

    plt.xlabel("Sell Price")

    plt.title("Store-wise price distribution - California")



plt.subplot(4, 3, 2)

for j in ["TX_1", "TX_2", "TX_3"]:

    sns.distplot(sell_prices_df[sell_prices_df["store_id"] == j]["sell_price"], label = j)

    plt.legend()

    plt.xlabel("Sell Price")

    plt.title("Store-wise price distribution - Texas")



plt.subplot(4, 3, 3)

for k in ["WI_1", "WI_2", "WI_3"]:

    sns.distplot(sell_prices_df[sell_prices_df["store_id"] == k]["sell_price"], label = k)

    plt.legend()

    plt.xlabel("Sell Price")

    plt.title("Store-wise price distribution - Wisconsin")

    

plt.subplot(4, 3, 4)

for i in ["HOBBIES_1", "HOBBIES_2"]:

    sns.distplot(sell_prices_df[(sell_prices_df["cat_id"] == i) & (sell_prices_df["state"] == "CA")]["sell_price"], label = i)

    plt.legend()

    plt.xlabel("Sell Price")

    plt.title("Category-wise price distribution in California- Hobbies")

    

plt.subplot(4, 3, 5)

for i in ["HOBBIES_1", "HOBBIES_2"]:

    sns.distplot(sell_prices_df[(sell_prices_df["cat_id"] == i) & (sell_prices_df["state"] == "TX")]["sell_price"], label = i)

    plt.legend()

    plt.xlabel("Sell Price")

    plt.title("Category-wise price distribution in Texas- Hobbies")

    

plt.subplot(4, 3, 6)

for i in ["HOBBIES_1", "HOBBIES_2"]:

    sns.distplot(sell_prices_df[(sell_prices_df["cat_id"] == i) & (sell_prices_df["state"] == "WI")]["sell_price"], label = i)

    plt.legend()

    plt.xlabel("Sell Price")

    plt.title("Category-wise price distribution in Wisconsin- Hobbies")



plt.subplot(4, 3, 7)

for j in ["HOUSEHOLD_1", "HOUSEHOLD_2"]:

    sns.distplot(sell_prices_df[(sell_prices_df["cat_id"] == j) & (sell_prices_df["state"] == "CA")]["sell_price"], label = j)

    plt.legend()

    plt.xlabel("Sell Price")

    plt.title("Category-wise price distribution in California - Households")



plt.subplot(4, 3, 8)

for j in ["HOUSEHOLD_1", "HOUSEHOLD_2"]:

    sns.distplot(sell_prices_df[(sell_prices_df["cat_id"] == j) & (sell_prices_df["state"] == "TX")]["sell_price"], label = j)

    plt.legend()

    plt.xlabel("Sell Price")

    plt.title("Category-wise price distribution in Texas - Households")  

    

plt.subplot(4, 3, 9)

for j in ["HOUSEHOLD_1", "HOUSEHOLD_2"]:

    sns.distplot(sell_prices_df[(sell_prices_df["cat_id"] == j) & (sell_prices_df["state"] == "WI")]["sell_price"], label = j)

    plt.legend()

    plt.xlabel("Sell Price")

    plt.title("Category-wise price distribution in Wisconsin - Households")



plt.subplot(4, 3, 10)

for k in ["FOODS_1", "FOODS_2", "FOODS_3"]:

    sns.distplot(sell_prices_df[(sell_prices_df["cat_id"] == k) & (sell_prices_df["state"] == "CA")]["sell_price"], label = k)

    plt.legend()

    plt.xlabel("Sell Price")

    plt.title("Category-wise price distribution in California - Foods")

    

plt.subplot(4, 3, 11)

for k in ["FOODS_1", "FOODS_2", "FOODS_3"]:

    sns.distplot(sell_prices_df[(sell_prices_df["cat_id"] == k) & (sell_prices_df["state"] == "TX")]["sell_price"], label = k)

    plt.legend()

    plt.xlabel("Sell Price")

    plt.title("Category-wise price distribution in Texas - Foods")

    

plt.subplot(4, 3, 12)

for k in ["FOODS_1", "FOODS_2", "FOODS_3"]:

    sns.distplot(sell_prices_df[(sell_prices_df["cat_id"] == k) & (sell_prices_df["state"] == "WI")]["sell_price"], label = k)

    plt.legend()

    plt.xlabel("Sell Price")

    plt.title("Category-wise price distribution in Wisconsin - Foods")