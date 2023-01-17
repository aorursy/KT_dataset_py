import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as  plt
from matplotlib.widgets import TextBox
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
import ipywidgets as widgets
from datetime import datetime, timedelta

%matplotlib notebook
sns.set_palette("deep")
def validate_csv(path):
    df = pd.read_csv(path, dtype={"LATITUDE": float, "LONGITUDE": float}, parse_dates=["DATE", "TIME"], low_memory=False)
    
    assert df["DATE"].dtype == np.dtype('datetime64[ns]'), "invalid date. Expect 'datetime64[ns]' got {}".format(df["DATE"].dtype)
    assert df["TIME"].dtype == np.dtype('datetime64[ns]'), "invalid time. Expect 'datetime64[ns]' got {}".format(df["TIME"].dtype)
    assert df["LATITUDE"].dtype == np.dtype(float), "invalid latitude. Expect 'float' got {}".format(df["LATITUDE"].dtype)
    assert df["LONGITUDE"].dtype == np.dtype(float), "invalid longitude. Expect 'float' got {}".format(df["LONGITUDE"].dtype)
#from io import StringIO
#
## good data
#good_data = StringIO("""DATE,TIME,LATITUDE,LONGITUDE,IGNORED
#2019-1-25T01:59:10,02:01,40,70,""
#2018-11-02T1:1:10,23:55,-51,0,IGNORED1""
#1960-3-2T00:55:1,00:00,50.0,-60.5,IGNORED2""
#""")
#
#validate_csv(good_data)
## bad data
#bad_data = StringIO("""DATE,TIME,LATITUDE,LONGITUDE,IGNORED
#2019-1-25,2:1,40,70,""
#2018-11-02T00:00:00,23:55.5,,0,IGNORED1""
#1960-3-2T00:-1:54,-1,50.0,lol,IGNORED2""
#""")
#
#validate_csv(bad_data)
# Validate csv types and then load it into a data frame
path = "../input/nypd-motor-vehicle-collisions.csv"
validate_csv(path)
df = pd.read_csv(path, low_memory=False)
def preprocessing(df):
    # Drop rows missing the latitude or longitude
    dataframe = df.dropna(subset=["DATE", "TIME","LONGITUDE", "LATITUDE"]).copy()
    dataframe = dataframe[(dataframe["LATITUDE"] != 0) & (dataframe["LONGITUDE"] != 0)]
    
    return dataframe

# test preprocessing
test_data = StringIO("""DATE,TIME,LATITUDE,LONGITUDE
2019-1-25T01:59:10,02:01,40,70
,23:55,-51,0
1960-3-2T00:55:1,NA,50.0,-60.5
1960-3-2T00:55:1,00:00,0,-60.5
1960-3-2T00:55:1,00:00,50.0,0
""")
test_df = pd.read_csv(test_data)

expect_data = StringIO("""DATE,TIME,LATITUDE,LONGITUDE
2019-1-25T01:59:10,02:01,40.0,70.0
""")
expect_df = pd.read_csv(expect_data)

# check to see if the test dataframe & the
# dataframe our function prouduces are the same
# (this will produce no output)
pd.testing.assert_frame_equal(expect_df, preprocessing(test_df))
df = preprocessing(df)

# Convert date column to a proper date format
df["DATE"] = pd.to_datetime(df["DATE"], format="%Y-%m-%d")
df["TIME"] = pd.to_datetime(df["TIME"], format="%H:%M")
# The better way: using ipywidgets...but they don't work in HTML mode
def collisions_distplot(df, start_date, end_date):
    """Plots the number of collisions throughout the day given a start and end date"""
    
    start_stamp = pd.Timestamp(start_date)
    end_stamp = pd.Timestamp(end_date)
    interval_df = df.loc[(df["DATE"] >= start_stamp) & (df["DATE"] <= end_stamp)]

    plt.figure(figsize=(10.5, 4))
    ax = sns.distplot(interval_df["TIME"].dt.hour, bins=np.arange(25), kde=False, hist_kws={"rwidth":0.999,'edgecolor':'white', 'alpha':1.0})
    
    # Add number of collisions on top of bars
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width() / 2, p.get_height(), str(int(p.get_height())), horizontalalignment="center")
    
    ax.set_xlim([0, 24])
    ax.set_xticks(np.arange(25))
    ax.set_xlabel("Hour")
    ax.set_ylabel("Number of collisions")
    ax.set_title("Collisions throughout the day from {} to {}".format(start_date.strftime("%d-%m-%Y"), end_date.strftime("%d-%m-%Y")))
    plt.show()

# Widgets for interactivity
today = datetime.combine(datetime.today().date(), datetime.min.time())
start_picker = widgets.DatePicker(description='Start date', value=today - timedelta(days=60))
end_picker = widgets.DatePicker(description='End date', value=today)
widgets.interact(collisions_distplot, df=widgets.fixed(df), start_date=start_picker, end_date=end_picker)
# The matplotlib way...shrug
f, ax = plt.subplots(figsize=(10.5, 5))
plt.subplots_adjust(bottom=0.35)

def collisions_distplot(ax, df, start_date, end_date):
    """Plots the number of collisions throughout the day given a start and end date"""
    # Threashold the dataframe using the start and end dates
    start_stamp = pd.Timestamp(start_date)
    end_stamp = pd.Timestamp(end_date)
    interval_df = df.loc[(df["DATE"] >= start_stamp) & (df["DATE"] <= end_stamp)].copy()
    
    # Clear and plot updated distribution
    ax.cla()
    sns.distplot(interval_df["TIME"].dt.hour, bins=np.arange(25), kde=False, ax=ax, hist_kws={"rwidth":0.999,'edgecolor':'white', 'alpha':1.0})
    
    # Add number of collisions on top of bars
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width() / 2, p.get_height(), str(int(p.get_height())), horizontalalignment="center")

    ax.set_xlim([0, 24])
    ax.set_xticks(np.arange(25))
    ax.set_xlabel("Hour")
    ax.set_ylabel("Number of collisions")
    ax.set_title("Collisions throughout the day from {} to {}".format(start_date.strftime("%d-%m-%Y"), end_date.strftime("%d-%m-%Y")))

# Initial date text
init_start_date = (datetime.today() - timedelta(days=60)).strftime("%d-%m-%Y")
init_end_date = datetime.today().strftime("%d-%m-%Y")

# Matplotlib widgets for interactivity
# [left, bottom, width, height] (origin is the bottom left corner of the figure)
start_ax = plt.axes([0.15, 0.15, 0.4, 0.075])
end_ax = plt.axes([0.15, 0.05, 0.4, 0.075])
start_tbox = TextBox(start_ax, 'Start date', initial=init_start_date)
end_tbox = TextBox(end_ax, 'End date', initial=init_end_date)

# Callback for text boxes
def submit(_):
    start_text = start_tbox.text
    start_date = datetime.strptime(start_text, "%d-%m-%Y")
    end_text = end_tbox.text
    end_date = datetime.strptime(end_text, "%d-%m-%Y")
    collisions_distplot(ax, df, start_date, end_date)

start_tbox.on_submit(submit)
end_tbox.on_submit(submit)

# Plot with initial dates
submit(0)
# Large amounts of markers do not render in Jupyter notebooks in Chrome. Te variable below sets the maximum number of accidents that will be plotted
num_accidents = 1000

# Choose accidents to plot randomly
df = df.sample(num_accidents)

# Zoom default is 10
m = folium.Map(location=[df["LATITUDE"].mean(), df["LONGITUDE"].mean()], zoom_start=11)
marker_cluster = MarkerCluster()
for index, row in df.iterrows():
    location = (row["LATITUDE"], row["LONGITUDE"])
    marker = folium.Marker(location=location, icon=folium.Icon(color="red", prefix="fa", icon="car"))
    marker_cluster.add_child(marker)

marker_cluster.add_to(m)
display(m)