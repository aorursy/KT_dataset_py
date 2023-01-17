#importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

import tensorflow.keras.backend as keb

from scipy.optimize import curve_fit

from datetime import datetime as dtime

from datetime import timedelta



import warnings

warnings.filterwarnings("ignore")
#loading data

path = "../input/covid19-global-forecasting-week-4/"

#path = ""

data = pd.read_csv(path+"train.csv")

data= data.drop("Id", axis=1)

data.head()
data["isCountry"] = data.Province_State.isna()

data.Province_State[data.Province_State.isna()] = ""
data_non_countries = data[data.isCountry==False]

data_non_countries = data_non_countries[["Country_Region", "Date","ConfirmedCases", "Fatalities"]].groupby(["Country_Region", "Date"]).sum()

data_non_countries.reset_index(inplace=True)

data_non_countries["Province_State"] = np.nan

data_non_countries = data_non_countries[["Province_State", "Country_Region", "Date", "ConfirmedCases", "Fatalities"]]

data_non_countries["isCountry"] = True



data = pd.concat([data, data_non_countries])

data.reset_index(inplace=True)

data["Id"] = data.index

data = data.drop("index", axis=1)

data["Region"]= data.Country_Region 

data["Region"][~data.isCountry]= data.Country_Region + "-" + data.Province_State

number_of_rows = data.shape[0]

registers_count_by_region = data[["Id", "Region"]].groupby("Region").count()

registers_count_by_region.columns = ["Count"]

registers_count_by_region.reset_index(inplace=True)#

days = np.unique(data.Date)

n_days = len(days)

regions = np.unique(data.Region)



#new confirmed column

data["Id_by_Region"] = data[["Region", "Id"]].groupby("Region").cumcount()

data["PreviousConfirmed"] = data["ConfirmedCases"].shift()

data.loc[ data.Id_by_Region==0, "PreviousConfirmed"] = 0

data["NewConfirmed"]= data.ConfirmedCases - data.PreviousConfirmed



#new fatalities column

data["Id_by_Region"] = data[["Region", "Id"]].groupby("Region").cumcount()

data["PreviousFatalities"] = data["Fatalities"].shift()

data.loc[ data.Id_by_Region==0,"PreviousFatalities"] = 0

data["NewFatalities"]= data.Fatalities - data.PreviousFatalities



data["FirstConfirmed"] = data.apply(lambda x: x["ConfirmedCases"]>0, axis=1)

data["FirstFatality"] = data.apply(lambda x: x["Fatalities"]>0, axis=1)

data["DayFromFirstConfirmed"] = data[["FirstConfirmed", "Region"]].groupby("Region").cumsum()

data["DayFromFirstFatality"] = data[["FirstFatality", "Region"]].groupby("Region").cumsum()



data["TenConfirmed"] = data.apply(lambda x: x["ConfirmedCases"]>10, axis=1)

data["TenFatality"] = data.apply(lambda x: x["Fatalities"]>10, axis=1)

data["DayFromTenConfirmed"] = data[["TenConfirmed", "Region"]].groupby("Region").cumsum()

data["DayFromTenFatality"] = data[["TenFatality", "Region"]].groupby("Region").cumsum()



data["LogNewConfirmed"] = np.log(data["NewConfirmed"]+1) 

data["LogNewFatalities"] = np.log(data["NewFatalities"]+1)



print("Days:", days)

print("No. regions:", len(regions))

print("Regions:", regions)
N = 5

data ["NewConfirmedSmoothed"] = np.convolve(data.NewConfirmed, np.ones((N,))/N, mode='same')

data ["NewFatalitiesSmoothed"] = np.convolve(data.NewFatalities, np.ones((N,))/N, mode='same')
data ["NewConfirmedSmoothed"] = 0

data ["NewFatalitiesSmoothed"] = 0



for region in regions:

    data ["NewConfirmedSmoothed"][data.Region==region] = np.convolve(data[data.Region==region].NewConfirmed, np.ones((N,))/N, mode='same')

    data ["NewFatalitiesSmoothed"][data.Region==region] = np.convolve(data[data.Region==region].NewFatalities, np.ones((N,))/N, mode='same')

selected_countries = ["Spain", "Italy", "Germany", "Singapore", 'Korea, South']



fig, ax = plt.subplots(len(selected_countries),2, figsize=(15,10))



for i, country in enumerate(selected_countries):

    

    confirmed_country = data[data.Region==country].NewConfirmed

    confirmed_country_smoothed = data[data.Region==country].NewConfirmedSmoothed

    ax[i][0].plot(days, confirmed_country)

    ax[i][0].plot(days, confirmed_country_smoothed)

    ax[i][0].set_xticks(np.arange(0, n_days,20 ))



    ax[i][0].set_title("Confirmed cases")

    ax[i][0].grid()



for i, country in enumerate(selected_countries):

    

    fatalities_country = data[data.Region==country].NewFatalities

    fatalities_country_smoothed = data[data.Region==country].NewFatalitiesSmoothed

    ax[i][1].plot(days, fatalities_country)

    ax[i][1].plot(days, fatalities_country_smoothed)

    ax[i][1].set_xticks(np.arange(0, n_days,20 ))

    

    ax[i][1].set_title("Fatalities")

    ax[i][1].grid()
max_confirmed = data[["Region", "NewConfirmed"]].groupby("Region").max()

max_confirmed.reset_index(inplace=True)

max_confirmed.columns = ["Region", "MaxConfirmed"]



day_max_confirmed = data[["Region", "NewConfirmed"]].groupby("Region").idxmax()

day_max_confirmed2 = data[["DayFromTenConfirmed"]].iloc[day_max_confirmed.NewConfirmed]



day_max_confirmed.NewConfirmed = np.array(day_max_confirmed2.DayFromTenConfirmed)

day_max_confirmed.reset_index(inplace=True)

day_max_confirmed.columns = ["Region", "DayMaxConfirmed"]



current_day_count = data[["Region", "DayFromTenConfirmed"]].groupby("Region").max()

current_day_count .reset_index(inplace=True)

current_day_count.columns = ["Region", "CurrentDayCount"]



data_merged = max_confirmed.merge(day_max_confirmed, on = "Region").merge(current_day_count, on = "Region")

data_merged.head()
max_fatalities = data[["Region", "NewFatalities"]].groupby("Region").max()#smoothed?

max_fatalities.reset_index(inplace=True)

max_fatalities.columns = ["Region", "MaxFatalities"]



day_max_fatalities = data[["Region", "NewFatalities"]].groupby("Region").idxmax()#smoothed

day_max_fatalities2 = data[["DayFromTenFatality"]].iloc[day_max_fatalities.NewFatalities]#smoothed



day_max_fatalities.NewFatalities = np.array(day_max_fatalities2.DayFromTenFatality)

day_max_fatalities.reset_index(inplace=True)

day_max_fatalities.columns = ["Region", "DayMaxFatalities"]



data_merged = data_merged.merge(max_fatalities, on="Region").merge(day_max_fatalities, on = "Region")
data_merged["LastConfirmed"] = 0

data_merged["LastFatalities"] = 0

data_merged["LastLastConfirmed"] = 0

data_merged["LastLastFatalities"] = 0



for region in data_merged.Region:

    data_merged["LastConfirmed"][data_merged.Region==region] = data[data.Region==region]["NewConfirmed"].iloc[-1]

    data_merged["LastFatalities"][data_merged.Region==region] = data[data.Region==region]["NewFatalities"].iloc[-1]

    data_merged["LastLastConfirmed"][data_merged.Region==region] = data[data.Region==region]["NewConfirmed"].iloc[-2]

    data_merged["LastLastFatalities"][data_merged.Region==region] = data[data.Region==region]["NewFatalities"].iloc[-2]

data_merged
count = []

for threshold in range(20):

    

    country_type = data_merged[["DayMaxConfirmed", "CurrentDayCount"]].apply(lambda x: (x[1]-x[0])>threshold, 1)

    count.append(np.sum(country_type))

    

plt.plot(range(20), count)

plt.grid()
data_merged["CountryType"] = data_merged[["DayMaxConfirmed", "CurrentDayCount"]].apply(lambda x: int((x[1]-x[0])>7), 1)
list(data_merged[data_merged.CountryType == 1].Region)
color = ["red", "green"]

color_assigned = [color[i] for i in data_merged.CountryType]

plt.scatter(data_merged.DayMaxConfirmed, np.log(data_merged.MaxConfirmed), c=color_assigned)

plt.grid()
temp_data = data[data.Region=="Germany"][data.TenConfirmed==True]

day_max_confirmed = int(data_merged[data_merged.Region=="Germany"].DayMaxConfirmed)

max_confirmed = int(data_merged[data_merged.Region=="Germany"].MaxConfirmed)

plt.plot(temp_data.DayFromTenConfirmed,temp_data.NewConfirmed)

plt.axvline(x=day_max_confirmed, c="red")

plt.grid()
from scipy.optimize import curve_fit



def linear(x, a):

    

    return a*x



pos_data = temp_data[temp_data.DayFromTenConfirmed>int(day_max_confirmed)].NewConfirmed

x_data = np.arange(pos_data.shape[0])

log_data = np.log(pos_data)-np.log(max_confirmed)



popt, pcov = curve_fit(linear, x_data, log_data)

pred = linear(x_data, *popt)

plt.plot(x_data, np.exp(log_data+np.log(max_confirmed)))

plt.plot(x_data, np.exp(pred+np.log(max_confirmed)))

plt.grid()

print("Desacelerating rate:", popt)
data_merged["DesacceleratingRateConfirmed"] = 0.0

data_merged["DesacceleratingRateFatalities"] = 0.0



for region in regions:

    

    temp_data = data[data.Region==region][data.TenConfirmed==True]

    

    if(temp_data.shape[0]>0):

        

        temp_data_merged = data_merged[data_merged.Region==region]



        country_type = int(temp_data_merged.CountryType)

        

        if(country_type==1):

            



            day_max_confirmed = int(temp_data_merged.DayMaxConfirmed)

            max_confirmed = int(temp_data_merged.MaxConfirmed)

            day_max_fatalities = int(temp_data_merged.DayMaxFatalities)

            max_fatalities = int(temp_data_merged.MaxFatalities)

            isCountry = temp_data.isCountry.iloc[0]

            

            pos_data = temp_data[temp_data.DayFromTenConfirmed>int(day_max_confirmed)].NewConfirmed

            x_data = np.arange(pos_data.shape[0])

            log_data = np.log(pos_data.clip(0)+1)-np.log(max_confirmed+1)

            popt, pcov = curve_fit(linear, x_data, log_data)        

            data_merged.loc[data_merged.Region==region, "DesacceleratingRateConfirmed"]= popt[0]

            

            pos_data = temp_data[temp_data.DayFromTenConfirmed>int(day_max_fatalities)].NewFatalities

            x_data = np.arange(pos_data.shape[0])

            log_data = np.log(pos_data.clip(0)+1)-np.log(max_fatalities+1)

            popt, pcov = curve_fit(linear, x_data, log_data)        

            data_merged.loc[data_merged.Region==region, "DesacceleratingRateFatalities"]= popt[0]
#obtain the country name

path2 = "../input/demographic/"

country_region = data[["Region", "Country_Region", "isCountry"]].drop_duplicates()

data_merged = data_merged.merge(country_region, on="Region", how="left")

data_merged.head()
gini = pd.read_csv(path2+"gini.csv")

gini_2019 = gini[["country","2019"]]

gini_2019.columns = ["Country_Region", "Gini"]

gini_2019.head()
population = pd.read_csv(path2+"population_total.csv")

population_2019 = population[["country", "2019"]]

population_2019.columns = ["Country_Region", "Population"]

population_2019.Population = population_2019.Population/1000000

population_2019.head()
health_system = pd.read_csv(path2+"government_health_spending_of_total_gov_spending_percent.csv")

health_system_2010 = health_system [["country", "2010"]]

health_system_2010.columns = ["Country_Region", "HealthSystem"]

health_system_2010.head()
gdp = pd.read_csv(path2+"gdp_total_yearly_growth.csv")

gdp_2013 = gdp[["country", "2013"]]

gdp_2013.columns = ["Country_Region", "GDP"]

gdp_2013.head()
life_expectancy = pd.read_csv(path2+"life_expectancy_years.csv")

life_expectancy_2019 = life_expectancy[["country", "2019"]]

life_expectancy_2019.columns = ["Country_Region", "LifeExpectancy"]

life_expectancy_2019.head()
smokers = pd.read_csv(path2+"smoking_adults_percent_of_population_over_age_15.csv")

smokers_2005 = smokers[["country", "2005"]]

smokers_2005.columns = ["Country_Region", "Smokers"]

smokers_2005.head()
demographic_data = gini_2019.merge(population_2019, on="Country_Region", how="outer")

demographic_data = demographic_data.merge(health_system_2010, on="Country_Region", how="outer")

demographic_data = demographic_data.merge(gdp_2013, on = "Country_Region", how="outer")

demographic_data = demographic_data.merge(life_expectancy_2019, on = "Country_Region", how="outer")

demographic_data = demographic_data.merge(smokers_2005, on = "Country_Region", how="outer")

demographic_data
country_names_dict ={"Congo, Dem. Rep.": "Congo (Kinshasa)",

                     "Congo, Rep.": "Congo (Brazzaville)",

                     "Czech Republic": "Czechia",

                     "Kyrgyz Republic": "Kyrgyzstan",

                     "South Korea": "Korea, South",

                     "Lao": "Laos",

                     "St. Kitts and Nevis":"Saint Kitts and Nevis",

                     "St. Lucia":"Saint Lucia",

                     "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",

                     "Slovak Republic": "Slovakia",

                     "United States": "US"}

demographic_data["Country_Region"] = demographic_data["Country_Region"].replace(country_names_dict)
data_merged2 = data_merged.merge(demographic_data, on="Country_Region", how="outer")

data_merged2
data_merged
list(data_merged2[np.isnan(data_merged2.MaxConfirmed) | np.isnan(data_merged2.Gini)].Country_Region)
data_merged3 = data_merged2[~np.isnan(data_merged2.MaxConfirmed)]

data_merged3
#imputing demographic variables

from sklearn.impute import SimpleImputer



features = ["CurrentDayCount", "Gini", "Population", "HealthSystem", "GDP", "LifeExpectancy", "Smokers", "LastConfirmed", "LastFatalities", "LastLastConfirmed", "LastLastFatalities"]

targets = ["MaxConfirmed", "DayMaxConfirmed", "MaxFatalities", "DayMaxFatalities", "DesacceleratingRateConfirmed", "DesacceleratingRateFatalities"]

X = data_merged3[features][data_merged3.isCountry][data_merged3.CountryType==1]

y = data_merged3[targets][data_merged3.isCountry][data_merged3.CountryType==1]



imputer = SimpleImputer(missing_values=np.nan,  strategy="mean")

X_imputed = imputer.fit_transform(X)

X_imputed.shape
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



y = np.array(y)

y[:,0] = np.log(y[:,0]+1)

y[:,2] = np.log(y[:,2]+1)





X_train, X_test, y_train, y_test = train_test_split( X_imputed, y, test_size=0.33, random_state=42)



X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)





from sklearn.datasets import make_regression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error





rgr_list = []

for i in range(len(targets)):

    rgr = RandomForestRegressor(max_depth=4, random_state=0)

    rgr.fit(X_train_scaled, y_train[:,i])

    rgr_list.append(rgr)

    

    pred_train = rgr.predict(X_train_scaled)

    pred = rgr.predict(X_test_scaled)

    score = mean_squared_error(pred, y_test[:,i])

    print("Score for "+targets[i]+" :", score)

    

    fig = plt.figure()

    plt.scatter(pred, y_test[:,i])

    plt.grid()

    plt.title(targets[i])







X_scaled = scaler.fit_transform(X_imputed)



rgr_list = []

for i in range(len(targets)):

    rgr = RandomForestRegressor(max_depth=3, random_state=0)

    rgr.fit(X_scaled, y[:,i])

    rgr_list.append(rgr)

X = data_merged3[features][data_merged3.isCountry]

X_imputed = imputer.transform(X)

X_scaled = scaler.transform(X_imputed)



pred= []

for i in range(len(targets)):

    

    pred.append(rgr_list[i].predict(X_scaled))

    

pred = np.array(pred).T

pred[:,0] = np.exp(pred[:,0]).astype(int)

pred[:,1] = pred[:,1].astype(int)

pred[:,2] = np.exp(pred[:,2]).astype(int)

pred[:,3] = pred[:,3].astype(int)

pred_df = pd.DataFrame(pred)

pred_df.columns = ["Pred"+target for target in targets]
pred_df
data_pred = data_merged3[data_merged3.isCountry]

data_pred.reset_index(inplace=True)

data_pred= data_pred.assign(**pred_df)

data_pred


data_pred["PredMaxConfirmed"] = data_pred[["PredMaxConfirmed", "MaxConfirmed"]].apply(lambda x: np.max((x[0], x[1])),1)

data_pred["PredMaxFatalities"] = data_pred[["PredMaxFatalities", "MaxFatalities"]].apply(lambda x: max(x[0], x[1]),1)

data_pred["PredDayMaxConfirmed"] = data_pred[["PredDayMaxConfirmed", "DayMaxConfirmed"]].apply(lambda x: max(x[0], x[1]),1)

data_pred["PredDayMaxFatalities"] = data_pred[["PredDayMaxFatalities", "DayMaxFatalities"]].apply(lambda x: max(x[0], x[1]),1)





data_pred["PredMaxConfirmed"] = data_pred[["PredMaxConfirmed", "MaxConfirmed", "CountryType"]].apply(lambda x: x[1] if x[2]==1 else x[0],1)

data_pred["PredMaxFatalities"] = data_pred[["PredMaxFatalities", "MaxFatalities", "CountryType"]].apply(lambda x: x[1] if x[2]==1 else x[0],1)

data_pred["PredDayMaxConfirmed"] = data_pred[["PredDayMaxConfirmed", "DayMaxConfirmed", "CountryType"]].apply(lambda x: x[1] if x[2]==1 else x[0],1)

data_pred["PredDayMaxFatalities"] = data_pred[["PredDayMaxFatalities", "DayMaxFatalities", "CountryType"]].apply(lambda x: x[1] if x[2]==1 else x[0],1)

data_pred["PredDesacceleratingRateConfirmed"] = data_pred[["PredDesacceleratingRateConfirmed", "DesacceleratingRateConfirmed", "CountryType"]].apply(lambda x: x[1] if x[2]==1 else x[0],1)

data_pred["PredDesacceleratingRateFatalities"] = data_pred[["PredDesacceleratingRateFatalities", "DesacceleratingRateFatalities", "CountryType"]].apply(lambda x: x[1] if x[2]==1 else x[0],1)



data_countries = data[data.isCountry==False][["Country_Region", "ConfirmedCases"]].groupby(["Country_Region"]).max()

data_non_countries = data[data.isCountry==False][["Province_State", "Country_Region", "ConfirmedCases"]].groupby(["Province_State", "Country_Region"]).max()

data_non_countries.reset_index(inplace=True)

data_non_countries = data_non_countries.merge(data_countries, on="Country_Region", how="left")

data_non_countries["RegionFraction"] = data_non_countries[["ConfirmedCases_x", "ConfirmedCases_y"]].apply(lambda x: x[0]/x[1], axis=1)

data_non_countries
def pred_type1 (max_value, desaccelerating_rate, time_horizon):

    

    x = np.arange(time_horizon)

    pred = np.exp(x*desaccelerating_rate+np.log(max_value+1))

    return pred





def pred_type0 (current_day, day_max_value, current_value, max_value, time_horizon):

    

    x = np.arange(current_day, current_day+time_horizon)

    x0, x1, y0, y1 = current_day, day_max_value, current_value, max_value

    

    a = (1/(x1-x0))*(np.log(y1+1)-np.log(y0+1))

    b = (1/(x1-x0))*(-x0*np.log(y1+1)+x1*np.log(y0+1))



    pred = np.exp(a*x+b)

    return pred

test_data = pd.read_csv(path+"test.csv")

test_data.head()



test_data["isCountry"] = test_data.Province_State.isna()

test_data.Province_State[test_data.Province_State.isna()] = ""

test_data["Region"]= test_data.Country_Region 

test_data["Region"][~test_data.isCountry]= test_data.Country_Region + "-" + test_data.Province_State



number_of_rows = test_data.shape[0]

registers_count_by_region_test = test_data[["ForecastId", "Region"]].groupby("Region").count()

registers_count_by_region_test.columns = ["Count"]

registers_count_by_region_test.reset_index(inplace=True)#

days_test = np.unique(test_data.Date)

n_days_test = len(days_test)



test_data
np.unique(data_merged.Region)
days_to_predict = [dtime.strftime(dtime.strptime(days[-1],  "%Y-%m-%d")+timedelta(i), "%Y-%m-%d") for i in range(1,34)]

n_days_to_predict = len(days_to_predict)

concat_df = pd.DataFrame()

print(days_to_predict)
concat_df = pd.DataFrame()



for i, region in enumerate(regions):

    data_region = data[data.Region == region]

    data_merged_region = data_merged[data_merged.Region == region]

    data_pred_region = data_pred[data_pred.Region==region]

    isCountry = int(data_region.isCountry.iloc[-1])

    

    temp_df = pd.DataFrame({"Date":days_to_predict, "Country_Region": [data_region.Country_Region.iloc[0]]*n_days_to_predict, 

                            "Province_State": [data_region.Province_State.iloc[0]]*n_days_to_predict,

                            "isCountry": [data_region.isCountry.iloc[0]]*n_days_to_predict})

      

    

    if(isCountry==0):



        province = data_region.Province_State.iloc[-1]

        temp_data_non_country = data_non_countries[data_non_countries.Province_State==province]

        country = data_region.Country_Region.iloc[-1]

        data_region = data[data.Region==country]

        data_pred_region = data_pred[data_pred.Region==country]

        data_merged_region = data_merged[data_merged.Region == country]





    current_day = int(data_merged_region["CurrentDayCount"])

    day_max_confirmed = int(data_pred_region["PredDayMaxConfirmed"])

    day_max_fatalities = int(data_pred_region["PredDayMaxFatalities"])

    max_confirmed = int(data_pred_region["PredMaxConfirmed"])

    max_fatalities = int(data_pred_region["PredMaxFatalities"])

    des_confirmed = float(data_pred_region["PredDesacceleratingRateConfirmed"])

    des_fatalities = float(data_pred_region["PredDesacceleratingRateFatalities"])

    current_new_confirmed = int(data_region.NewConfirmed.iloc[-1])

    current_new_fatalities = int(data_region.NewFatalities.iloc[-1])



    current_confirmed = data_region.ConfirmedCases.iloc[-1]

    current_fatalities = data_region.Fatalities.iloc[-1]



    if(current_day < day_max_confirmed):

        print(region)



        pred_0_confirmed = pred_type0 (current_day, day_max_confirmed, current_new_confirmed, max_confirmed, day_max_confirmed-current_day)            

        pred_1_confirmed = pred_type1 (current_new_confirmed+1, des_confirmed, n_days_to_predict-day_max_confirmed+current_day)            

        predicted_confirmed = list(pred_0_confirmed) + list(pred_1_confirmed)



    else:



        pred_1_confirmed = pred_type1 (current_new_confirmed+1, des_confirmed, n_days_to_predict)                  

        predicted_confirmed = pred_1_confirmed



    if(current_day< day_max_fatalities):

        pred_0_fatalities = pred_type0 (current_day, day_max_fatalities, current_new_fatalities, max_fatalities, day_max_fatalities-current_day)                   

        pred_1_fatalities = pred_type1 (current_new_fatalities+1, des_fatalities, n_days_to_predict-day_max_fatalities+current_day)            

        predicted_fatalities = list(pred_0_fatalities) + list(pred_1_fatalities)        



    else:

        pred_1_fatalities = pred_type1 (current_new_fatalities+1, des_fatalities, n_days_to_predict)            

        predicted_fatalities = pred_1_fatalities   

        

    if(isCountry==0):

        fraction = float(temp_data_non_country.RegionFraction)

        predicted_confirmed = predicted_confirmed*fraction

        predicted_fatalities = predicted_fatalities*fraction

        

    predicted_confirmed[0] = predicted_confirmed[0] + current_confirmed

    predicted_confirmed = np.cumsum(predicted_confirmed)



    predicted_fatalities[0] = predicted_fatalities[0] + current_fatalities

    predicted_fatalities = np.cumsum(predicted_fatalities)

        

    temp_df["Region"]= temp_df.Country_Region 

    temp_df["Region"][~temp_df.isCountry] = temp_df.Country_Region + "-" + temp_df.Province_State

    

    

    

    temp_df["ConfirmedCases"] = predicted_confirmed

    temp_df["Fatalities"] = predicted_fatalities

    concat_df = pd.concat([concat_df, temp_df])
current_data = data[["Date", "Country_Region", "Province_State", "ConfirmedCases", "Fatalities", "Region"]]

current_data.Province_State[current_data.Province_State.isna()] = ""

current_data = current_data.groupby(["Date", "Country_Region", "Province_State", "Region"]).max()

submission1 = test_data.merge(current_data, on=["Region", "Date"] ,  how='left')

submission2 = submission1.merge(concat_df, on= ["Region", "Date"], how="left")

submission2["ConfirmedCases"] = submission2[["ConfirmedCases_x", "ConfirmedCases_y"]].apply(lambda x: x[0] if ~np.isnan(x[0]) else x[1], axis=1)

submission2["Fatalities"] = submission2[["Fatalities_x", "Fatalities_y"]].apply(lambda x: x[0] if ~np.isnan(x[0]) else x[1], axis=1)

submission_data = pd.read_csv(path+"submission.csv")#

submission3 = submission_data[["ForecastId"]].merge(submission2[["ForecastId", "ConfirmedCases", "Fatalities" ]], on= "ForecastId", how="left")



submission3 = submission3.astype("int32")

submission3.reset_index(inplace=True)

submission3 = submission3.drop(["index"], axis=1)

submission3.to_csv("submission.csv", index=False)