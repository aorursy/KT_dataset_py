# Load common libraries:

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import folium





# set some display options:

sns.set(style="whitegrid")

pd.set_option("display.max_columns", 36)
# load data:

file_path = "../input/hotel-booking-demand/hotel_bookings.csv"

full_data = pd.read_csv(file_path)
full_data.shape
full_data.head()

#full_data.tail()
full_data.info()
full_data.hotel.unique()
t = pd.DataFrame([[i,full_data[i].unique()] for i in full_data.columns])

t.columns = ['name','unique']

t   
full_data.describe(include='all')
# check for missing values

full_data.isnull().sum()
# Replace missing values:

nan_replacements = {"children": 0.0, "country": "Unknown", "agent": 0, "company": 0}

full_data_cln = full_data.fillna(nan_replacements)



# "meal" contains values "Undefined", which is equal to SC.

full_data_cln["meal"].replace("Undefined", "SC", inplace=True)
# check for missing values

print('Remaining Missing Values = ',full_data_cln.isna().sum().sum())
zero_guests = list(full_data_cln.loc[full_data_cln["adults"]

                   + full_data_cln["children"]

                   + full_data_cln["babies"]==0].index)

zero_guests
full_data_cln.drop(full_data_cln.index[zero_guests], inplace=True)
# How much data is left?

full_data_cln.shape
t = pd.DataFrame([[i,full_data[i].unique()] for i in full_data_cln.columns])

t.columns = ['name','unique']

t   
full_data_cln[full_data_cln['babies'] > 8]
full_data_cln[full_data_cln['children'] > 8]
full_data_cln[full_data_cln['required_car_parking_spaces'] > 7]
ax = sns.boxplot(x=full_data_cln['adr'])
# Deleting a record with ADR greater than 5000

full_data_cln = full_data_cln[full_data_cln['adr'] < 5000]

ax = sns.boxplot(x=full_data_cln['adr'])
full_data_cln['hotel'].value_counts()
rh = full_data_cln.loc[(full_data_cln["hotel"] == "Resort Hotel") & (full_data_cln["is_canceled"] == 0)]

ch = full_data_cln.loc[(full_data_cln["hotel"] == "City Hotel") & (full_data_cln["is_canceled"] == 0)]
# get number of acutal guests by country

country_data = pd.DataFrame(full_data_cln.loc[full_data_cln["is_canceled"] == 0]["country"].value_counts())

country_data.index.name = "country"

country_data.rename(columns={"country": "Number of Guests"}, inplace=True)

total_guests = country_data["Number of Guests"].sum()

country_data["Guests in %"] = round(country_data["Number of Guests"] / total_guests * 100, 2)

country_data.head()
# show on map

guest_map = px.choropleth(country_data,

                    locations=country_data.index,

                    color=country_data["Guests in %"], 

                    hover_name=country_data.index, 

                    color_continuous_scale=px.colors.sequential.Plasma,

                    title="Home country of guests")

guest_map.show()
plt.rcParams['figure.figsize'] = 15,6

plt.hist(full_data_cln['lead_time'], bins=50)



plt.ylabel('Count')

plt.xlabel('Time (days)')

plt.title("Lead time distribution ", fontdict=None, position= [0.48,1.05])

plt.show()
rh["adr"].describe()
ch["adr"].describe()
full_data_guests = full_data_cln.loc[full_data_cln["is_canceled"] == 0] # only actual gusts

room_prices = full_data_guests[["hotel", "reserved_room_type", "adr"]].sort_values("reserved_room_type")



# barplot with standard deviation:

plt.figure(figsize=(12, 8))

sns.barplot(x = "reserved_room_type", y="adr", hue="hotel", data=room_prices, 

            hue_order = ["City Hotel", "Resort Hotel"], ci="sd", errwidth=1, capsize=0.2)

plt.title("Price of room types per night", fontsize=16)

plt.xlabel("Room type", fontsize=16)

plt.ylabel("Price [EUR]", fontsize=16)

plt.legend(loc="upper right")

plt.show()
full_data_guests['total_guests'] = full_data_guests['adults']+ full_data_guests['children']+ full_data_guests['babies']

plt.figure(figsize=(12,8))

ax = sns.countplot(x="total_guests", data = full_data_guests)

plt.title('Number of Guests')

plt.xlabel('total_guests')

plt.ylabel('Count')

for p in ax.patches:

    ax.annotate((p.get_height()),(p.get_x()+0.1 , p.get_height()+100)) 
# normalize price per night (adr):

full_data_cln["adr_pp"] = full_data_cln["adr"] / (full_data_cln["adults"] + full_data_cln["children"])

full_data_guests = full_data_cln.loc[full_data_cln["is_canceled"] == 0] # only actual gusts

room_prices = full_data_guests[["hotel", "reserved_room_type", "adr_pp"]].sort_values("reserved_room_type")



# barplot with standard deviation:

plt.figure(figsize=(12, 8))

sns.barplot(x = "reserved_room_type", y="adr_pp", hue="hotel", data=room_prices, 

            hue_order = ["City Hotel", "Resort Hotel"], ci="sd", errwidth=1, capsize=0.2)

plt.title("Price of room types per night and person", fontsize=16)

plt.xlabel("Room type", fontsize=16)

plt.ylabel("Price [EUR]", fontsize=16)

plt.legend(loc="upper right")

plt.show()
# grab data:

room_prices_mothly = full_data_guests[["hotel", "arrival_date_month", "adr"]].sort_values("arrival_date_month")



# order by month:

ordered_months = ["January", "February", "March", "April", "May", "June", 

          "July", "August", "September", "October", "November", "December"]

room_prices_mothly["arrival_date_month"] = pd.Categorical(room_prices_mothly["arrival_date_month"], categories=ordered_months, ordered=True)



# barplot with standard deviation:

plt.figure(figsize=(12, 6))

sns.lineplot(x = "arrival_date_month", y="adr", hue="hotel", data=room_prices_mothly, 

            hue_order = ["City Hotel", "Resort Hotel"], ci="sd", size="hotel", sizes=(2.5, 2.5))

plt.title("Room price per night over the year", fontsize=16)

plt.xlabel("Month", fontsize=16)

plt.xticks(rotation=45)

plt.ylabel("Price [EUR]", fontsize=16)

plt.show()
# Create a DateFrame with the relevant data:

resort_guests_monthly = rh.groupby("arrival_date_month")["hotel"].count()

city_guests_monthly = ch.groupby("arrival_date_month")["hotel"].count()



resort_guest_data = pd.DataFrame({"month": list(resort_guests_monthly.index),

                    "hotel": "Resort hotel", 

                    "guests": list(resort_guests_monthly.values)})



city_guest_data = pd.DataFrame({"month": list(city_guests_monthly.index),

                    "hotel": "City hotel", 

                    "guests": list(city_guests_monthly.values)})

full_guest_data = pd.concat([resort_guest_data,city_guest_data], ignore_index=True)



# order by month:

ordered_months = ["January", "February", "March", "April", "May", "June", 

          "July", "August", "September", "October", "November", "December"]

full_guest_data["month"] = pd.Categorical(full_guest_data["month"], categories=ordered_months, ordered=True)



# Dataset contains July and August date from 3 years, the other months from 2 years. Normalize data:

full_guest_data.loc[(full_guest_data["month"] == "July") | (full_guest_data["month"] == "August"),

                    "guests"] /= 3

full_guest_data.loc[~((full_guest_data["month"] == "July") | (full_guest_data["month"] == "August")),

                    "guests"] /= 2



# show figure:

plt.figure(figsize=(12, 6))

sns.lineplot(x = "month", y="guests", hue="hotel", data=full_guest_data, 

             hue_order = ["City hotel", "Resort hotel"], size="hotel", sizes=(2.5, 2.5))

plt.title("Average number of hotel guests per month", fontsize=16)

plt.xlabel("Month", fontsize=16)

plt.xticks(rotation=45)

plt.ylabel("Number of guests", fontsize=16)

plt.show()
from datetime import datetime



def month_converter(month):

    months = ['January', 'February', 'March', 'April', 'May', 'June','July', 'August', 'September', 'October', 'November', 'December']

    return months.index(month) + 1



rh_arr = rh

rh_arr['arrival_month'] = rh_arr['arrival_date_month'].apply(month_converter)

rh_arr['arrival_year_month'] = rh_arr['arrival_date_year'].astype(str) + " _ " + rh_arr['arrival_month'].astype(str)

rh_arr['Arrrival Date'] = rh_arr.apply(lambda row: datetime.strptime(f"{int(row.arrival_date_year)}-{int(row.arrival_month)}-{int(row.arrival_date_day_of_month)}", '%Y-%m-%d'), axis=1)

rh_arr['arrival_day_of_week'] = rh_arr['Arrrival Date'].dt.day_name()

weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

rh_arr['arrival_day_of_week'] = pd.Categorical(rh_arr['arrival_day_of_week'],categories = weekdays)

arrivals = pd.pivot_table(rh_arr,columns = 'arrival_day_of_week',index = 'arrival_month',values = 'reservation_status',aggfunc = 'count')
fig, ax = plt.subplots(figsize = (16,11))

ax = sns.heatmap(arrivals, annot=True, fmt="d", cmap = 'rocket_r')
# Create a DateFrame with the relevant data:

rh["total_nights"] = rh["stays_in_weekend_nights"] + rh["stays_in_week_nights"]

ch["total_nights"] = ch["stays_in_weekend_nights"] + ch["stays_in_week_nights"]



num_nights_res = list(rh["total_nights"].value_counts().index)

num_bookings_res = list(rh["total_nights"].value_counts())

rel_bookings_res = rh["total_nights"].value_counts() / sum(num_bookings_res) * 100 # convert to percent



num_nights_cty = list(ch["total_nights"].value_counts().index)

num_bookings_cty = list(ch["total_nights"].value_counts())

rel_bookings_cty = ch["total_nights"].value_counts() / sum(num_bookings_cty) * 100 # convert to percent



res_nights = pd.DataFrame({"hotel": "Resort hotel",

                           "num_nights": num_nights_res,

                           "rel_num_bookings": rel_bookings_res})



cty_nights = pd.DataFrame({"hotel": "City hotel",

                           "num_nights": num_nights_cty,

                           "rel_num_bookings": rel_bookings_cty})



nights_data = pd.concat([res_nights, cty_nights], ignore_index=True)
# show figure:

plt.figure(figsize=(16, 6))

sns.barplot(x = "num_nights", y = "rel_num_bookings", hue="hotel", data=nights_data,

            hue_order = ["City hotel", "Resort hotel"])

plt.title("Length of stay", fontsize=16)

plt.xlabel("Number of nights", fontsize=16)

plt.ylabel("Guests [%]", fontsize=16)

plt.legend(loc="upper right")

plt.show()
avg_nights_res = sum(list((res_nights["num_nights"] * (res_nights["rel_num_bookings"]/100)).values))

avg_nights_cty = sum(list((cty_nights["num_nights"] * (cty_nights["rel_num_bookings"]/100)).values))

print(f"On average, guests of the City hotel stay {avg_nights_cty:.2f} nights, and {cty_nights['num_nights'].max()} at maximum.")

print(f"On average, guests of the Resort hotel stay {avg_nights_res:.2f} nights, and {res_nights['num_nights'].max()} at maximum.")
plt.figure(figsize=(12,6))

ax = sns.countplot(x="market_segment", data=full_data_cln, order = full_data_cln['market_segment'].value_counts().index)

plt.title('Market Segment')

plt.xlabel('market_segment')

plt.ylabel('Count')

for p in ax.patches:

    ax.annotate((p.get_height()),(p.get_x()+0.2 , p.get_height()+100)) 
plt.figure(figsize=(12,6))

ax = sns.countplot(x="distribution_channel", data=full_data_cln, order = full_data_cln['distribution_channel'].value_counts().index)

plt.title('Distribution Channel')

plt.xlabel('distribution_channel')

plt.ylabel('Count')
plt.figure(figsize=(12,6))

ax = sns.countplot(x="is_repeated_guest", data = full_data_cln)

plt.title('Is Repeated Guest?')

plt.xlabel('is_repeated_guest')

plt.ylabel('Total Count')
# absolute cancelations:

total_cancelations = full_data_cln["is_canceled"].sum()

rh_cancelations = full_data_cln.loc[full_data_cln["hotel"] == "Resort Hotel"]["is_canceled"].sum()

ch_cancelations = full_data_cln.loc[full_data_cln["hotel"] == "City Hotel"]["is_canceled"].sum()



# as percent:

rel_cancel = total_cancelations / full_data_cln.shape[0] * 100

rh_rel_cancel = rh_cancelations / full_data_cln.loc[full_data_cln["hotel"] == "Resort Hotel"].shape[0] * 100

ch_rel_cancel = ch_cancelations / full_data_cln.loc[full_data_cln["hotel"] == "City Hotel"].shape[0] * 100



print(f"Total bookings canceled: {total_cancelations:,} ({rel_cancel:.0f} %)")

print(f"Resort hotel bookings canceled: {rh_cancelations:,} ({rh_rel_cancel:.0f} %)")

print(f"City hotel bookings canceled: {ch_cancelations:,} ({ch_rel_cancel:.0f} %)")
# Create a DateFrame with the relevant data:

res_book_per_month = full_data_cln.loc[(full_data_cln["hotel"] == "Resort Hotel")].groupby("arrival_date_month")["hotel"].count()

res_cancel_per_month = full_data_cln.loc[(full_data_cln["hotel"] == "Resort Hotel")].groupby("arrival_date_month")["is_canceled"].sum()



cty_book_per_month = full_data_cln.loc[(full_data_cln["hotel"] == "City Hotel")].groupby("arrival_date_month")["hotel"].count()

cty_cancel_per_month = full_data_cln.loc[(full_data_cln["hotel"] == "City Hotel")].groupby("arrival_date_month")["is_canceled"].sum()



res_cancel_data = pd.DataFrame({"Hotel": "Resort Hotel",

                                "Month": list(res_book_per_month.index),

                                "Bookings": list(res_book_per_month.values),

                                "Cancelations": list(res_cancel_per_month.values)})

cty_cancel_data = pd.DataFrame({"Hotel": "City Hotel",

                                "Month": list(cty_book_per_month.index),

                                "Bookings": list(cty_book_per_month.values),

                                "Cancelations": list(cty_cancel_per_month.values)})



full_cancel_data = pd.concat([res_cancel_data, cty_cancel_data], ignore_index=True)

full_cancel_data["cancel_percent"] = full_cancel_data["Cancelations"] / full_cancel_data["Bookings"] * 100



# order by month:

ordered_months = ["January", "February", "March", "April", "May", "June", 

          "July", "August", "September", "October", "November", "December"]

full_cancel_data["Month"] = pd.Categorical(full_cancel_data["Month"], categories=ordered_months, ordered=True)



# show figure:

plt.figure(figsize=(12, 8))

sns.barplot(x = "Month", y = "cancel_percent" , hue="Hotel",

            hue_order = ["City Hotel", "Resort Hotel"], data=full_cancel_data)

plt.title("Cancelations per month", fontsize=16)

plt.xlabel("Month", fontsize=16)

plt.ylabel("Cancelations [%]", fontsize=16)

plt.legend(loc="upper right")

plt.show()