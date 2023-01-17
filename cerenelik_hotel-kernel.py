# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings
warnings.filterwarnings("ignore")


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
hotel_bookings_df=pd.read_csv("/kaggle/input/hotel-booking-demand/hotel_bookings.csv")
hotel_bookings_df.columns
hotel_bookings_df.head()
hotel_bookings_df.describe()
hotel_bookings_df.info()
def bar_plot(variable):
    """
    input :variable ex:"hotels"
    output: bar plot & value count
    """
    var=hotel_bookings_df[variable]
    varValue=var.value_counts()
    plt.figure(figsize=(9,3))
    plt.bar(varValue.index,varValue)
    plt.xticks(varValue.index,varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}:/n {}".format(variable,varValue))
    
category1=["hotel","is_canceled","days_in_waiting_list","required_car_parking_spaces","stays_in_weekend_nights"]
for c in category1:
    bar_plot(c)
category2=["company","arrival_date_day_of_month","arrival_date_year","arrival_date_day_of_month","meal","country","market_segment","distribution_channel","is_repeated_guest","previous_cancellations","previous_bookings_not_canceled","reserved_room_type","assigned_room_type","deposit_type"]
for c in category2:
    print("{} /n".format(hotel_bookings_df[c].value_counts()))

def plot_hist(variable):
    var=hotel_bookings_df[variable]
    varValue=var.value_counts()
    plt.figure(figsize=(9,3))
    plt.bar(varValue.index,varValue)
    plt.xticks(varValue.index,varValue.index.values)
    plt.hist(hotel_bookings_df[variable])
    plt.xlabel(variable)
    plt.ylabel("Frenquency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()
    print("{} /n {}".format(variable,varValue))
    
numericVar=["agent","adults","children","babies","adr"]
for n in numericVar:
    plot_hist(n)
hotel_bookings_df[["agent","children","adults","hotel"]].groupby(["hotel"],as_index=False).mean()
sns.set(style="whitegrid")
pd.set_option("display.max_columns", 36)


file_path = "../input/hotel-booking-demand/hotel_bookings.csv"
full_data = pd.read_csv(file_path)
full_data.head()
hotel_bookings_df.tail()
hotel_bookings_df.shape
hotel_bookings_df.isnull().sum()
hotel_bookings_df.describe(include="all").T
rh = hotel_bookings_df.loc[( hotel_bookings_df["hotel"] == "Resort Hotel") & ( hotel_bookings_df["is_canceled"] == 0)]
ch =  hotel_bookings_df.loc[( hotel_bookings_df["hotel"] == "City Hotel") & ( hotel_bookings_df["is_canceled"] == 0)]
hoteltype=hotel_bookings_df.groupby("hotel").is_canceled.count().reset_index()
hoteltype.columns=["hotel","count"]
sns.set(style="whitegrid")
ax = sns.barplot(x="hotel", y="count", data=hoteltype)
cancelp=hotel_bookings_df.groupby(["hotel","is_canceled"]).lead_time.count().reset_index()
cancelp.columns=["hotel","is_canceled","count"]
ax = sns.barplot(x="hotel", y="count", hue="is_canceled", data=cancelp)
ax = sns.distplot(hotel_bookings_df.is_canceled,kde=False,color='r')
for p in ax.patches:
    if p.get_height() > 0: ax.text(p.get_x()+p.get_width()/2,p.get_height(),f"{int(p.get_height())}",fontsize=16) 
bookings_by_year =  hotel_bookings_df[ hotel_bookings_df.is_canceled == 0].groupby('arrival_date_year').arrival_date_year.count()
ax = sns.barplot(bookings_by_year.index,bookings_by_year.values)
for p in ax.patches:
    ax.text(p.get_x()+p.get_width()/4,p.get_height(),f"{int(p.get_height())}",fontsize=16)
booking_by_year_month = hotel_bookings_df[hotel_bookings_df.is_canceled == 0].groupby(['arrival_date_year','arrival_date_month']).arrival_date_month.count().sort_values(ascending=False)
plt.figure(figsize=(10,30))
plt.title('Number of Bookings by Year-Month')
sns.set(font_scale=1.1)
ax = sns.barplot(booking_by_year_month.values,booking_by_year_month.index)
for p in ax.patches:
    ax.text(p.get_width(),p.get_y()+p.get_height()/2,f"{int(p.get_width())}",fontsize=16)
