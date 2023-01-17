# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



#warnings

import warnings

warnings.filterwarnings('ignore')



#folium

import folium

#plotly

import plotly.express as px

import plotly.figure_factory as ff











# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



#WEEK2 DATA

tw2=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")

test_week2=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")



#Population 2020

population=pd.read_csv("/kaggle/input/population-data/Population_2020.csv")



#WEATHER

weather=pd.read_csv("/kaggle/input/weather-data/weather.csv")





#TW2.............................

tw2=tw2.rename(columns={"Province_State":"Province","Country_Region":"Country","ConfirmedCases":"Confirmed"})

tw2["Province"]=tw2["Province"].fillna('')



#WEATHER DATA...........................

weather=weather.drop(columns=["Unnamed: 0","Confirmed","Fatalities","capital","Province","Id"])

weather["humidity"]=weather["humidity"].fillna(0)

weather["sunHour"]=weather["sunHour"].fillna(0)

weather["tempC"]=weather["tempC"].fillna(0)

weather["windspeedKmph"]=weather["windspeedKmph"].fillna(0)







#POPULATION

population=population.drop('Unnamed: 0',axis=1)

population["fertility"]=population["fertility"].fillna(0)

population["age"]=population["age"].fillna(0)

population["urban_percentage"]=population["urban_percentage"].fillna(0)





data=pd.merge(tw2,weather,on=["Country","Date"],how="inner")
data.info()
data=pd.merge(data,population,on=["Country"],how="inner")
data.info()
#CHANGING FLOAT TO INT

data["Confirmed"]=data["Confirmed"].astype(int)

data["Fatalities"]=data["Fatalities"].astype(int)



#CONVERTING DATE OBJECT TO DATETIME FORMAT



data["Date"]=pd.to_datetime(data["Date"])
data.info()
data.shape
data["Date"].min(),data["Date"].max()
data.head()
data_latest=data[data["Date"]==max(data["Date"])].reset_index()

data_latest=data_latest.drop(['index','Id'],axis=1)



data_grouped=data.groupby(["Province","Country","Date"])["Confirmed",

                                                         "Fatalities",

                                                        "Lat","Long","humidity","sunHour","tempC",

                                                        "windspeedKmph","Population",

                                                        "density","fertility","age",

                                                        "urban_percentage"].agg({"Confirmed":"sum",

                                                                                "Fatalities":"sum",

                                                                                "Lat":"mean",

                                                                                "Long":"mean",

                                                                                "humidity":"mean",

                                                                                "sunHour":"mean",

                                                                                "tempC":"mean",

                                                                                "windspeedKmph":"mean",

                                                                                "Population":"mean",

                                                                                "density":"mean",

                                                                                "fertility":"mean",

                                                                                "age":"mean",

                                                                                "urban_percentage":"mean"}).reset_index()



data_grouped["Date"]=data_grouped["Date"].dt.strftime("%m/%d/%Y")



data_latest_grouped=data_latest.groupby(["Country"])["Confirmed","Fatalities","Population","Lat","Long"].agg({"Confirmed":"sum",

                                                                                                 "Fatalities":"sum",

                                                                                                 "Population":"mean",

                                                                                                 "Lat":"mean","Long":"mean"}).reset_index()
gdf=data.groupby(["Country"])["Confirmed",

                             "Fatalities",

                            "Lat","Long","humidity","sunHour","tempC",

                            "windspeedKmph","Population",

                            "density","fertility","age",

                            "urban_percentage"].agg({"Confirmed":"sum",

                                                    "Fatalities":"sum",

                                                    "Lat":"mean",

                                                    "Long":"mean",

                                                    "humidity":"mean",

                                                    "sunHour":"mean",

                                                    "tempC":"mean",

                                                    "windspeedKmph":"mean",

                                                    "Population":"mean",

                                                    "density":"mean",

                                                    "fertility":"mean",

                                                    "age":"mean",

                                                    "urban_percentage":"mean"}).reset_index()
data_latest.head()
data_grouped.head()
#SPREAD OVER TIME...........................................



fig=px.scatter_geo(data_grouped,locations="Country",

                  locationmode="country names",

                   color=np.log(data_grouped["Confirmed"]),

                  animation_frame="Date",

                   size=data_grouped["Confirmed"].pow(0.3),

                  projection="natural earth",

                  hover_name="Country",

                  title="Spread of Coronavirus Over time")



fig.show()
#TOTAL CONFIRMED CASES AROUND THE WORLD



fig=px.choropleth(gdf,locations="Country",

                 locationmode="country names",

                 color=np.log(gdf["Confirmed"]),

                  hover_name="Country",

                  hover_data=["Confirmed","Population","tempC","windspeedKmph","humidity","sunHour"],

                  title="Total Confirmed cases around the world")



fig.show()
most_con=data_latest_grouped.sort_values(by="Confirmed",ascending=False)[0:10].reset_index(drop=True).style.background_gradient(cmap="Reds")

most_con

fig=px.bar(data_latest_grouped.sort_values(by="Confirmed")[-10:],x="Confirmed",y="Country",

          color_discrete_sequence=['dark cyan'],orientation='h',

           text='Confirmed',height=800,title="Confirmed Cases")

fig.show()
temp_con=data.groupby(["Country","Date"])["Confirmed","Fatalities"].sum().reset_index()

temp_con["Date"]=temp_con["Date"].dt.strftime("%m/%d/%Y")
temp_con.head()
#LINE CHART OF CONFIRMED CASES OF EACH COUNTRY



fig=px.line(temp_con,x="Date",y="Confirmed",color="Country",title="Confirmed cases in each country over time")



fig.show()
#TOTAL FATALITY CASES AROUND THE WORLD



fig=px.choropleth(gdf,locations="Country",

                 locationmode="country names",

                 color=np.log(gdf["Fatalities"]),

                  hover_name="Country",

                  hover_data=["Fatalities","Population","tempC","windspeedKmph","humidity","sunHour"],

                  title="Total Fatality cases around the world")



fig.show()
most_fat=data_latest_grouped.sort_values(by="Fatalities",ascending=False)[0:10].reset_index(drop=True).style.background_gradient(cmap="Reds")

most_fat
#For a better visualization



fig=px.bar(data_latest_grouped.sort_values(by="Fatalities")[-10:],x="Fatalities",y="Country",

          color_discrete_sequence=['dark cyan'],orientation='h',

           text='Fatalities',height=800,title="Fatality Cases")

fig.show()
#LINE CHART OF CONFIRMED CASES OF EACH COUNTRY



fig=px.line(temp_con,x="Date",y="Fatalities",color="Country",title="Fatality cases in each country over time")



fig.show()
#LATEST CONFIRMED CASES AROUND THE WORLD



fig=px.choropleth(data_latest_grouped,locations="Country",

                 locationmode="country names",

                 color=np.log(data_latest_grouped["Confirmed"]),

                  hover_name="Country",

                  hover_data=["Confirmed","Population"],

                  title="Total Confirmed cases around the world")



fig.show()
#LATEST FATALITY CASES AROUND THE WORLD



fig=px.choropleth(data_latest_grouped,locations="Country",

                 locationmode="country names",

                 color=np.log(data_latest_grouped["Fatalities"]),

                  hover_name="Country",

                  hover_data=["Fatalities","Population"],

                  title="Latest Fatality cases around the world")



fig.show()
fig=px.bar(gdf.sort_values(by='humidity')[-10:],

           x="Confirmed",y="Country",

          color_discrete_sequence=['dark cyan'],orientation='h',

           text="Confirmed",hover_data=["Confirmed","Fatalities","Population","humidity"],

                                         height=500,title="Highest humidity countries")

fig.show()
fig=px.bar(gdf.sort_values(by='Population')[-10:],

           x="Confirmed",y="Country",

          color_discrete_sequence=['dark cyan'],orientation='h',

           text="Confirmed",hover_data=["Confirmed","Fatalities","Population"],

                                         height=500,title="Most populated countries")

fig.show()
fig=px.bar(gdf.sort_values(by='Population')[0:10][::-1],

           x="Confirmed",y="Country",

          color_discrete_sequence=['dark cyan'],orientation='h',

           text="Confirmed",hover_data=["Confirmed","Fatalities","Population"],

                                         height=500,title="Most populated countries")

fig.show()
fig=px.bar(gdf.sort_values(by='windspeedKmph')[-10:],

           x="Confirmed",y="Country",

          color_discrete_sequence=['dark cyan'],orientation='h',

           text="Confirmed",hover_data=["Confirmed","Fatalities","windspeedKmph"],

                                         height=500,title="Countries with high wind speed")

fig.show()
fig=px.bar(gdf.sort_values(by='windspeedKmph')[0:10][::-1],

           x="Confirmed",y="Country",

          color_discrete_sequence=['dark cyan'],orientation='h',

           text="Confirmed",hover_data=["Confirmed","Fatalities","windspeedKmph"],

                                         height=500,title="Countries with less wind speed")

fig.show()
fig=px.bar(gdf.sort_values(by='tempC')[-10:],

           x="Confirmed",y="Country",

          color_discrete_sequence=['dark cyan'],orientation='h',

           text="Confirmed",hover_data=["Confirmed","Fatalities","tempC","Population","windspeedKmph"],

                                         height=500,title="Countries with high temperature")

fig.show()
fig=px.bar(gdf.sort_values(by='tempC')[0:20][::-1],

           x="Confirmed",y="Country",

          color_discrete_sequence=['dark cyan'],orientation='h',

           text="Confirmed",hover_data=["Confirmed","Fatalities","tempC","Population","windspeedKmph"],

                                         height=800,title="Countries with low temperature")

fig.show()
fig=px.bar(gdf.sort_values(by='sunHour')[-20:],

           x="Confirmed",y="Country",

          color_discrete_sequence=['dark cyan'],orientation='h',

           text="Confirmed",hover_data=["Confirmed","Fatalities","tempC","Population","windspeedKmph","sunHour"],

                                         height=800,title="Countries with more hours of sunlight")

fig.show()
fig=px.bar(gdf.sort_values(by='sunHour')[0:20][::-1],

           x="Confirmed",y="Country",

          color_discrete_sequence=['dark cyan'],orientation='h',

           text="Confirmed",hover_data=["Confirmed","Fatalities","tempC","Population","windspeedKmph","sunHour"],

                                         height=800,title="Countries with more hours of sunlight")

fig.show()
#We have training data as (data) dataset

#We'll split this data into training and evaluating datasets



#For that we need to drop some columns and create a train_data dataset



train_data=data.drop(columns=["Id","Province","Country","Date","Lat","Long"],axis=1)
train_data.head()
#We'll create X and y 

#X will have all dependent features

#y will have target variables



y=train_data[["Confirmed","Fatalities"]]

X=train_data.drop(columns=["Confirmed","Fatalities"],axis=1)



X.head()
y.head()
from sklearn.model_selection import train_test_split as tts



X_train,X_val,y_train,y_val=tts(X,y,test_size=0.2,random_state=42)
#training and testing data are ready

#We'll be using Random Forest Classifier



from sklearn.ensemble import RandomForestRegressor



#Model for predicting Confirmed cases

rf_confirmed=RandomForestRegressor(n_estimators=1000, random_state = 42)

#Model for predicting Fatality cases

rf_fatality=RandomForestRegressor(n_estimators=1000,random_state=42)
#FITTING CONFIRMED MODEL TO TRAINING DATA

rf_confirmed.fit(X_train,y_train["Confirmed"])
#PREDICTING ON EVALUATING DATA

result_confirmed=rf_confirmed.predict(X_val)
#Error

from sklearn.metrics import mean_squared_log_error
error_confirmed=np.sqrt(mean_squared_log_error(y_val["Confirmed"],result_confirmed))

print(error_confirmed)
rf_fatality.fit(X_train,y_train["Fatalities"])
result_fatality=rf_fatality.predict(X_val)
#Error

error_fatality=np.sqrt(mean_squared_log_error(y_val["Fatalities"],result_fatality))

print(error_fatality)
print("Final Validatio score: {}".format(np.mean([error_confirmed,error_fatality])))
model_confirmed=rf_confirmed.fit(X,y["Confirmed"])

model_fatalities=rf_fatality.fit(X,y["Fatalities"])
# Extract feature importances for confirmed

fi_con = pd.DataFrame({'feature': list(X.columns),

                   'importance': model_confirmed.feature_importances_})
fi_con.sort_values(by="importance",ascending=False).reset_index(drop=True)
# Get list of important variables for predicting confirmed cases

importances_confirmed = list(model_confirmed.feature_importances_)
features_list=list(X.columns)
#With data visualization for important variables for confirmed cases



# Set the style

plt.style.use('ggplot')

# list of x locations for plotting

x_values = list(range(len(importances_confirmed)))

# Make a bar chart

plt.bar(x_values, importances_confirmed, orientation = 'vertical')

# Tick labels for x axis

plt.xticks(x_values, features_list, rotation='vertical')

# Axis labels and title

plt.ylabel('Importance')

plt.xlabel('Variable')

plt.title('Variable Importances for Confirmed cases')
# Extract feature importances for fatalities

fi_fatalities = pd.DataFrame({'feature': list(X.columns),

                   'importance': model_fatalities.feature_importances_})
fi_fatalities.sort_values(by="importance",ascending=False).reset_index(drop=True)
# Get a list of important variables for predicting fatality cases

importances_fatalities = list(model_fatalities.feature_importances_)
features_list=list(X.columns)
#With data visualization for fatality cases



# Set the style

plt.style.use('ggplot')

# list of x locations for plotting

x_values = list(range(len(importances_fatalities)))

# Make a bar chart

plt.bar(x_values, importances_fatalities, orientation = 'vertical')

# Tick labels for x axis

plt.xticks(x_values, features_list, rotation='vertical')

# Axis labels and title

plt.ylabel('Importance')

plt.xlabel('Variable')

plt.title('Variable Importances for fatality cases')
test_week2.head()
test_week2=test_week2.rename(columns={"ForecastId":"Id","Province_State":"Province","Country_Region":"Country"})

test_week2["Province"]=test_week2["Province"].fillna('')
weather.head()
test_week2.head()
test_df=test_week2.merge(weather,on=["Country","Date"],how='left')

test_df.head()
test_df=test_df.merge(population,on=["Country"],how="left")

test_df.head()
test_df.info()
X_test = test_df.set_index("Id").drop(["Lat", "Long", "Date", "Province", "Country"], axis=1).fillna(0)

X_test.head()
X_test.info()
y_pred_confirmed = model_confirmed.predict(X_test)

y_pred_fatalities = model_fatalities.predict(X_test)
len(y_pred_confirmed)
submission = pd.DataFrame()

submission["ForecastId"]= pd.to_numeric(test_df["Id"], errors= 'coerce')

submission["ConfirmedCases"] = y_pred_confirmed

submission["Fatalities"] = y_pred_fatalities

submission["ConfirmedCases"]=submission["ConfirmedCases"].astype(int)

submission["Fatalities"]=submission["Fatalities"].astype(int)

submission = submission.drop_duplicates(subset= ['ForecastId'])

submission = submission.set_index(['ForecastId'])

submission.head()



submission.to_csv("submission.csv")