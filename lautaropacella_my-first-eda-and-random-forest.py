# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Visualization packages

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots



#Machine Learning packages

from sklearn.model_selection import KFold, cross_validate, cross_val_score

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import RandomizedSearchCV

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df= pd.read_csv("/kaggle/input/hotel-booking-demand/hotel_bookings.csv")
df.head()
df.info()
df.describe()
df.nunique()
df.rename(columns = {"arrival_date_year":"year", "arrival_date_month":"month", "arrival_date_week_number": "week_number", "arrival_date_day_of_month":"day_of_month", "stays_in_weekend_nights":"weekend_nights", "stays_in_week_nights":"week_nights"}, inplace=True)

df.info()
df.isnull().sum()
nan_replace = {"country": "Unknown", "agent": 0, "company": 0}

df = df.fillna(nan_replace)

df.dropna(subset = ["children"], inplace = True)
df.drop(df[(df["adults"] == 0) & (df["children"] == 0) & (df["babies"] == 0)].index, inplace = True)

df.drop(df[(df["weekend_nights"] == 0) & (df["week_nights"] == 0) & (df["adr"] == 0)].index, inplace = True)

df.shape
cancel = pd.DataFrame(df["is_canceled"].value_counts())

cancel.rename(columns={"is_canceled": "Cancellations"}, index =({0: "Not Canceled", 1:"Canceled"}), inplace=True)

cancel["Status"] = cancel.index



fig = px.pie(cancel, values = "Cancellations", names = "Status")

fig.update_traces(textposition='inside', textinfo='percent+label+value')

fig.update_layout(title_text = "Cancellations")

fig.show()



print(f"""There were {str(cancel[cancel["Status"]=="Canceled"]["Cancellations"].sum())} cancellations in total""")
hotel_res = pd.DataFrame(df["hotel"].value_counts())





hotel_res.rename(columns={"hotel":"Bookings"}, inplace=True)

hotel_res["Type"] = hotel_res.index

hotel_res.head()





fig = px.pie(hotel_res, values = "Bookings", names = "Type")

fig.update_traces(textposition='inside', textinfo='percent+label+value')

fig.update_layout(title_text = "Reservations per Hotel Type")

fig.show()
resort = df[df["hotel"] == "Resort Hotel"]

city = df[df["hotel"] == "City Hotel"]



cancel_res = pd.DataFrame(resort["is_canceled"].value_counts())

cancel_res.rename(columns={"is_canceled": "Cancellations"}, index =({0: "Not Canceled", 1:"Canceled"}), inplace=True)

cancel_res["Status"] = cancel_res.index



cancel_cit = pd.DataFrame(city["is_canceled"].value_counts())

cancel_cit.rename(columns={"is_canceled": "Cancellations"}, index =({0: "Not Canceled", 1:"Canceled"}), inplace=True)

cancel_cit["Status"] = cancel_cit.index





fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]], subplot_titles = ["City Hotel", "Resort Hotel"])

fig.add_trace(go.Pie(values = cancel_cit["Cancellations"], labels = cancel_cit["Status"]),1,1)

fig.add_trace(go.Pie(values = cancel_res["Cancellations"], labels = cancel_res["Status"]),1,2)

fig.update_traces(textposition='inside', textinfo='value+percent+label')

fig.update_layout(title_text = "Cancellations per Hotel Type")

fig.show()



print(f"The difference in cancellations between City Hotel and Resort Hotel was: {cancel_cit.iloc[1,0] - cancel_res.iloc[1,0]}")
country_data = pd.DataFrame(df["country"].value_counts())

country_data.rename(columns={"country": "Number of Guests"}, inplace=True)

country_data["Guests in %"] = round(country_data["Number of Guests"].value_counts(normalize=True))

country_data["country"] = country_data.index



fig = px.pie(country_data, values = "Number of Guests", names = "country")

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.update_layout(title_text = "Country per Bookings")

fig.show()
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

df.month=pd.Categorical(df.month, categories = months, ordered=True)



d2015 = df[df["year"] == 2015]

d2016 = df[df["year"] == 2016]

d2017 = df[df["year"] == 2017]



b2015 = pd.DataFrame(d2015["month"].value_counts(sort=False))

b2016 = pd.DataFrame(d2016["month"].value_counts(sort=False))

b2017 = pd.DataFrame(d2017["month"].value_counts(sort=False))



b2015.rename(columns={"month":"Bookings"}, inplace=True)

b2016.rename(columns={"month":"Bookings"}, inplace=True)

b2017.rename(columns={"month":"Bookings"}, inplace=True)





b2015["Month"]=b2015.index

b2016["Month"]=b2016.index

b2017["Month"]=b2017.index



fig = make_subplots(rows=3, cols=1)

fig.add_trace(go.Scatter(x= b2015.Month, y=b2015.Bookings, name="2015"), 1,1)

fig.add_trace(go.Scatter(x= b2016.Month, y= b2016.Bookings, name="2016"), 2,1)

fig.add_trace(go.Scatter(x= b2017.Month, y= b2017.Bookings, name="2017"), 3,1)

fig.update_layout(title="Bookings per Month",height=600)

fig.show()



print(f"""Mean bookings per month (where we have the data) were:

In 2015: {b2015[b2015["Bookings"]!=0]["Bookings"].mean()}

In 2016: {b2016[b2016["Bookings"]!=0]["Bookings"].mean()}

In 2017: {b2017[b2017["Bookings"]!=0]["Bookings"].mean()}""")
c2015 = df[(df["year"] == 2015) & (df["is_canceled"] == 1)]

c2016 = df[(df["year"] == 2016) & (df["is_canceled"] == 1)]

c2017 = df[(df["year"] == 2017) & (df["is_canceled"] == 1)]



c2015 = pd.DataFrame(c2015["month"].value_counts(sort=False))

c2016 = pd.DataFrame(c2016["month"].value_counts(sort=False))

c2017 = pd.DataFrame(c2017["month"].value_counts(sort=False))



c2015.rename(columns={"month":"Cancellations"}, inplace=True)

c2016.rename(columns={"month":"Cancellations"}, inplace=True)

c2017.rename(columns={"month":"Cancellations"}, inplace=True)





c2015["Month"]=c2015.index

c2016["Month"]=c2016.index

c2017["Month"]=c2017.index



fig = make_subplots(rows=3, cols=1)

fig.add_trace(go.Bar(x= c2015.Month, y=c2015.Cancellations, name="2015"), 1,1)

fig.add_trace(go.Bar(x= c2016.Month, y= c2016.Cancellations, name="2016"), 2,1)

fig.add_trace(go.Bar(x= c2017.Month, y= c2017.Cancellations, name="2017"), 3,1)

fig.update_layout(title="Cancellations per Month",height=800)

fig.show()



print(f"""Mean cancellations per month (where we have the data) were:

In 2015: {round(c2015[c2015["Cancellations"]!=0]["Cancellations"].mean(),2)}

In 2016: {round(c2016[c2016["Cancellations"]!=0]["Cancellations"].mean(),2)}

In 2017: {round(c2017[c2017["Cancellations"]!=0]["Cancellations"].mean(),2)}""")
df["adr_pp"] = df["adr"] / (df["adults"] + df["children"])

df_guest = df[df["is_canceled"]==0]

room_prices = df_guest[["hotel", "adr_pp", "reserved_room_type"]].sort_values("reserved_room_type")



plt.figure(figsize= (13,9))

sns.boxplot(x="reserved_room_type", y= "adr_pp", hue="hotel", data= room_prices, fliersize=0)

plt.ylim(0,170)
city_guest = df[(df["hotel"] == "City Hotel") & (df["is_canceled"] == 0)].copy()

resort_guest = df[(df["hotel"] == "Resort Hotel") & (df["is_canceled"] == 0)].copy()



city_guest["adr_pp"] = city_guest["adr"] / (city_guest["adults"] + city_guest["children"])

resort_guest["adr_pp"] = resort_guest["adr"] / (resort_guest["adults"] + resort_guest["children"])

print(f"""For every guest that didn't cancel, the average daily rate for each hotel type was: 

City Hotel: {round(city_guest["adr_pp"].mean(),2)}

Resort Hotel: {round(resort_guest["adr_pp"].mean(),2)}""")
#remind that: city_guest and resort_guest already have values



city_guest["total_nights"] = city_guest["weekend_nights"] + city_guest["week_nights"]

resort_guest["total_nights"] = resort_guest["weekend_nights"] + resort_guest["week_nights"]



print(f"""Mean nights that guest stays in each hotel type were:

City Hotel: {round(city_guest["total_nights"].mean(),2)}

Resort Hotel: {round(resort_guest["total_nights"].mean(),2)}""")
market = pd.DataFrame(df["market_segment"].value_counts(sort=True))

market.rename(columns={"market_segment":"Counts"}, inplace=True)

market["Segment"]=market.index





fig = px.bar(market, x="Counts", y= "Segment", orientation="h")

fig.update_layout(yaxis=dict(autorange="reversed"))

fig.show()
channel = pd.DataFrame(df["distribution_channel"].value_counts(sort=True))

channel.rename(columns={"distribution_channel":"Counts"}, inplace=True)

channel["Channel"]=channel.index





fig = px.bar(channel, x="Counts", y= "Channel", orientation="h")

fig.update_layout(yaxis=dict(autorange="reversed"))

fig.show()
deposit = pd.DataFrame(df["deposit_type"].value_counts(sort=True))

deposit.rename(columns={"deposit_type":"Counts"}, inplace=True)

deposit["Deposit Type"]=deposit.index





fig = px.bar(deposit, y="Counts", x= "Deposit Type")

fig.show()
num_var = ["lead_time","week_number","day_of_month","weekend_nights","week_nights","adults","children","babies","is_repeated_guest", "previous_cancellations","previous_bookings_not_canceled","agent","company","required_car_parking_spaces", "total_of_special_requests", "adr"]



cat_var = ["hotel","month","meal","market_segment","distribution_channel","reserved_room_type","deposit_type","customer_type"]
features = num_var + cat_var

X= df.drop(["is_canceled"], axis=1)[features]

y = df["is_canceled"]
num_transformer = SimpleImputer(strategy="constant")



cat_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),("onehot", OneHotEncoder(handle_unknown='ignore'))])



preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_var),("cat", cat_transformer, cat_var)])
rf = RandomForestClassifier(n_jobs=-1,random_state=42)

kf = KFold(n_splits=10, shuffle=True, random_state=42)



model_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', rf)])



cv_results = cross_val_score(model_pipe, X, y, cv=kf, scoring="accuracy",n_jobs=-1)



min_score = round(min(cv_results), 10)

max_score = round(max(cv_results), 10)

mean_score = round(np.mean(cv_results), 10)

std_dev = round(np.std(cv_results), 10)

print(f"Random Forest model cross validation accuarcy score: {mean_score} +/- {std_dev} (std) min: {min_score}, max: {max_score}")