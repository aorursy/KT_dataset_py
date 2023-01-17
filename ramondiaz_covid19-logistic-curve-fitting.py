import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import datetime

from sklearn import preprocessing

from scipy.optimize import curve_fit

from scipy.optimize import fsolve

import sklearn.metrics as metrics

import plotly.express as px

from pandas.plotting import table

from statsmodels.graphics.gofplots import qqplot

from scipy.stats import shapiro

import warnings

warnings.filterwarnings('ignore')

import os

print(os.listdir("../input"))

np.random.seed(100)
train = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv", na_values=['']) 

test = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv") 
train.info()

train.shape
test.info()

test.shape
train.ConfirmedCases = train.ConfirmedCases.astype(int)

train.Fatalities = train.Fatalities.astype(int)
train["Date"] = train["Date"].apply(pd.to_datetime, "%m/%d/%Y")

test["Date"] = test["Date"].apply(pd.to_datetime, "%m/%d/%Y")
test.shape
desc = train.describe()

#create a subplot without frame

plot = plt.subplot(111, frame_on=False)



#remove axis

plot.xaxis.set_visible(False) 

plot.yaxis.set_visible(False) 



#create the table plot and position it in the upper left corner

table(plot, desc,loc='upper right')



#save the plot as a png file

plt.savefig('mytable.png')
train.groupby("Date").agg({"ConfirmedCases":["sum"],"Fatalities":["sum"]}).plot(kind="line", figsize=(12,7),marker ="s",linewidth=2)

plt.xlabel("Date", fontsize=20)

plt.ylabel("Number of cases", fontsize=20)

plt.title("COVID19 Confirmed Cases and Fatalities since Jan 22,2020",fontsize=25)

plt.bar(train.groupby("Date").agg({"ConfirmedCases":["sum"],"Fatalities":["sum"]}).index, train.groupby("Date").sum()["ConfirmedCases"],alpha=0.1,color="b")

plt.savefig("Worldwide")

plt.show()

US_confirmed = train[train["Country_Region"]=="US"].groupby(["Date"]).agg({"ConfirmedCases":["sum"]})

US_fatalities = train[train["Country_Region"]=="US"].groupby(["Date"]).agg({"Fatalities":["sum"]})

US_cases = US_confirmed.join(US_fatalities)



plt.figure(figsize=(15,15))

plt.subplot(3, 2, 1)

US_cases.plot(ax=plt.gca(),marker ="s",linewidth=1, title='US Cases')

plt.bar(train[train["Country_Region"]=="US"].groupby("Date").sum()[["ConfirmedCases","Fatalities"]].index, train[train["Country_Region"]=="US"].groupby("Date").sum()["ConfirmedCases"],alpha=0.1,color="b")

plt.ylabel("Number of cases", size=10)

plt.xlim(datetime.date(2020, 3, 7))



MX_confirmed = train[train["Country_Region"]=="Mexico"].groupby(["Date"]).agg({"ConfirmedCases":["sum"]})

MX_fatalities = train[train["Country_Region"]=="Mexico"].groupby(["Date"]).agg({"Fatalities":["sum"]})

MX_cases = MX_confirmed.join(MX_fatalities)



plt.subplot(3, 2, 2)

MX_cases.plot(ax=plt.gca(),marker ="s",linewidth=1, title='Mexico Cases')

plt.bar(train[train["Country_Region"]=="Mexico"].groupby("Date").sum()[["ConfirmedCases","Fatalities"]].index, train[train["Country_Region"]=="Mexico"].groupby("Date").sum()["ConfirmedCases"],alpha=0.1,color="brown")

plt.xlim(datetime.date(2020, 2, 24))



CN_confirmed = train[train["Country_Region"]=="Canada"].groupby(["Date"]).agg({"ConfirmedCases":["sum"]})

CN_fatalities = train[train["Country_Region"]=="Canada"].groupby(["Date"]).agg({"Fatalities":["sum"]})

CN_cases = CN_confirmed.join(CN_fatalities)



plt.subplot(3, 2, 3)

CN_cases.plot(ax=plt.gca(),marker ="s",linewidth=1, title='Canada Cases')

plt.bar(train[train["Country_Region"]=="Canada"].groupby("Date").sum()[["ConfirmedCases","Fatalities"]].index, train[train["Country_Region"]=="Canada"].groupby("Date").sum()["ConfirmedCases"],alpha=0.1,color="orange")

plt.xlim(datetime.date(2020, 2, 24))

plt.xlabel("Date", fontsize=10)

plt.ylabel("Number of cases", size=10)



CH_confirmed = train[train["Country_Region"]=="China"].groupby(["Date"]).agg({"ConfirmedCases":["sum"]})

CH_fatalities = train[train["Country_Region"]=="China"].groupby(["Date"]).agg({"Fatalities":["sum"]})

CH_cases = CH_confirmed.join(CH_fatalities)



plt.subplot(3, 2, 4)

CH_cases.plot(ax=plt.gca(),marker ="s",linewidth=1, title='China Cases')

plt.bar(train[train["Country_Region"]=="China"].groupby("Date").sum()[["ConfirmedCases","Fatalities"]].index, train[train["Country_Region"]=="China"].groupby("Date").sum()["ConfirmedCases"],alpha=0.1,color="green")

plt.xlabel("Date", fontsize=10)



IT_confirmed = train[train["Country_Region"]=="Italy"].groupby(["Date"]).agg({"ConfirmedCases":["sum"]})

IT_fatalities = train[train["Country_Region"]=="Italy"].groupby(["Date"]).agg({"Fatalities":["sum"]})

IT_cases = IT_confirmed.join(IT_fatalities)



plt.subplot(3, 2, 5)

IT_cases.plot(ax=plt.gca(),marker ="s",linewidth=1, title='Italy Cases')

plt.bar(train[train["Country_Region"]=="Italy"].groupby("Date").sum()[["ConfirmedCases","Fatalities"]].index, train[train["Country_Region"]=="Italy"].groupby("Date").sum()["ConfirmedCases"],alpha=0.1,color="magenta")

plt.xlabel("Date", fontsize=10)

plt.ylabel("Number of cases", size=10)



IN_confirmed = train[train["Country_Region"]=="Iran"].groupby(["Date"]).agg({"ConfirmedCases":["sum"]})

IN_fatalities = train[train["Country_Region"]=="Iran"].groupby(["Date"]).agg({"Fatalities":["sum"]})

IN_cases = IN_confirmed.join(IN_fatalities)



plt.subplot(3, 2, 6)

IN_cases.plot(ax=plt.gca(),marker ="s",linewidth=1, title='Iran Cases')

plt.bar(train[train["Country_Region"]=="Iran"].groupby("Date").sum()[["ConfirmedCases","Fatalities"]].index, train[train["Country_Region"]=="Iran"].groupby("Date").sum()["ConfirmedCases"],alpha=0.1,color="pink")

plt.xlabel("Date", fontsize=10)

plt.tight_layout(3.0)

plt.savefig('myfigCountry')
US_confirmed_dates = train[(train["Country_Region"]=="US")].groupby(["Date"]).agg({"ConfirmedCases":["sum"]})

US_fatalities_dates = train[(train["Country_Region"]=="US")].groupby(["Date"]).agg({"Fatalities":["sum"]})

US_cases_dates = US_confirmed_dates.join(US_fatalities_dates)



MX_confirmed_dates = train[(train["Country_Region"]=="Mexico") & train["ConfirmedCases"]>0].groupby(["Date"]).agg({"ConfirmedCases":["sum"]})

MX_fatalities_dates = train[(train["Country_Region"]=="Mexico") & train["ConfirmedCases"]>0].groupby(["Date"]).agg({"Fatalities":["sum"]})

MX_cases_dates = MX_confirmed_dates.join(MX_fatalities_dates)



CN_confirmed_dates = train[(train["Country_Region"]=="Canada") & train["ConfirmedCases"]>0].groupby(["Date"]).agg({"ConfirmedCases":["sum"]})

CN_fatalities_dates = train[(train["Country_Region"]=="Canada") & train["ConfirmedCases"]>0].groupby(["Date"]).agg({"Fatalities":["sum"]})

CN_cases_dates = CN_confirmed_dates.join(CN_fatalities_dates)



CH_confirmed_dates = train[(train["Country_Region"]=="China")].groupby(["Date"]).agg({"ConfirmedCases":["sum"]})

CH_fatalities_dates = train[(train["Country_Region"]=="China")].groupby(["Date"]).agg({"Fatalities":["sum"]})

CH_cases_dates = CH_confirmed_dates.join(CH_fatalities_dates)



IT_confirmed_dates = train[(train["Country_Region"]=="Italy") & train["ConfirmedCases"]>0].groupby(["Date"]).agg({"ConfirmedCases":["sum"]})

IT_fatalities_dates = train[(train["Country_Region"]=="Italy") & train["ConfirmedCases"]>0].groupby(["Date"]).agg({"Fatalities":["sum"]})

IT_cases_dates = IT_confirmed_dates.join(IT_fatalities_dates)



IN_confirmed_dates = train[(train["Country_Region"]=="Iran") & train["ConfirmedCases"]>0].groupby(["Date"]).agg({"ConfirmedCases":["sum"]})

IN_fatalities_dates = train[(train["Country_Region"]=="Iran") & train["ConfirmedCases"]>0].groupby(["Date"]).agg({"Fatalities":["sum"]})

IN_cases_dates = IN_confirmed_dates.join(IN_fatalities_dates)



US = [e0 for e0 in US_cases_dates.ConfirmedCases['sum'].values]

US = US[49:]

MX = [e1 for e1 in MX_cases_dates.ConfirmedCases['sum'].values]

CN = [e2 for e2 in CN_cases_dates.ConfirmedCases['sum'].values]

CH = [e3 for e3 in CH_cases_dates.ConfirmedCases['sum'].values]

IT = [e2 for e2 in IT_cases_dates.ConfirmedCases['sum'].values]

IN = [e2 for e2 in IN_cases_dates.ConfirmedCases['sum'].values]



# Plots

plt.figure(figsize=(12,6))

plt.plot(US)

plt.plot(MX)

plt.plot(CN)

plt.plot(CH)

plt.plot(IT)

plt.plot(IN)

plt.legend(["US", "Mexico", "Canada", "China","Italy","Iran"], loc='upper left')

plt.title("Confirmed cases of COVID19", size=15)

plt.xlabel("Date", size=13)

plt.ylabel("Number of cases", size=13)

plt.ylim(0, 100000)

plt.xlim(0, 19)

plt.show()
US_fatalities = [e0 for e0 in US_cases_dates.Fatalities['sum'].values]

MX_fatalities = [e1 for e1 in MX_cases_dates.Fatalities['sum'].values]

CN_fatalities = [e2 for e2 in CN_cases_dates.Fatalities['sum'].values]

CH_fatalities = [e3 for e3 in CH_cases_dates.Fatalities['sum'].values]

IT_fatalities = [e3 for e3 in IT_cases_dates.Fatalities['sum'].values]

IN_fatalities = [e3 for e3 in IN_cases_dates.Fatalities['sum'].values]





plt.figure(figsize=(12,6))

plt.plot(US_fatalities)

plt.plot(MX_fatalities)

plt.plot(CN_fatalities)

plt.plot(CH_fatalities)

plt.plot(IT_fatalities)

plt.plot(IN_fatalities)



plt.legend(["US", "Mexico", "Canada","China","Italy","Iran"], loc='upper left')

plt.title("Fatalities of COVID19", size=15)

plt.xlabel("Date", size=13)

plt.ylabel("Number of cases", size=13)

plt.show()
US_cases_ratio = US_cases[datetime.date(2020, 3, 10):]

MX_cases_ratio = MX_cases[datetime.date(2020, 3, 19):]

CN_cases_ratio = CN_cases[datetime.date(2020, 3, 9):]

CH_cases_ratio = CH_cases[:datetime.date(2020, 3, 22)]

IN_cases_ratio = IN_cases[datetime.date(2020, 2, 19):]

IT_cases_ratio = IT_cases[datetime.date(2020, 2, 22):]

US_cases_ratio.columns = US_cases_ratio.columns.droplevel(level=1)

MX_cases_ratio.columns = MX_cases_ratio.columns.droplevel(level=1)

CN_cases_ratio.columns = CN_cases_ratio.columns.droplevel(level=1)

CH_cases_ratio.columns = CH_cases_ratio.columns.droplevel(level=1)

IN_cases_ratio.columns = IN_cases_ratio.columns.droplevel(level=1)

IT_cases_ratio.columns = IT_cases_ratio.columns.droplevel(level=1)

US_cases_ratio = US_cases_ratio.reset_index()

MX_cases_ratio = MX_cases_ratio.reset_index()

CN_cases_ratio = CN_cases_ratio.reset_index()

CH_cases_ratio = CH_cases_ratio.reset_index()

IN_cases_ratio = IN_cases_ratio.reset_index()

IT_cases_ratio = IT_cases_ratio.reset_index()



US_cases_ratio["Fatalities_difference"] = 0

US_cases_ratio["Fatalities_trend"] = 0.0

MX_cases_ratio["Fatalities_difference"] = 0

MX_cases_ratio["Fatalities_trend"] = 0.0

CN_cases_ratio["Fatalities_difference"] = 0

CN_cases_ratio["Fatalities_trend"] = 0.0

CH_cases_ratio["Fatalities_difference"] = 0

CH_cases_ratio["Fatalities_trend"] = 0.0

IT_cases_ratio["Fatalities_difference"] = 0

IN_cases_ratio["Fatalities_trend"] = 0.0

IT_cases_ratio["Fatalities_difference"] = 0

IT_cases_ratio["Fatalities_trend"] = 0.0
for element in range(1, US_cases_ratio.shape[0]):

    US_cases_ratio.at[US_cases_ratio.index[element], "Fatalities_difference"] = US_cases_ratio.iloc[element]["Fatalities"]-US_cases_ratio.iloc[element-1]["Fatalities"]

    US_cases_ratio.at[US_cases_ratio.index[element], "Fatalities_trend"] = US_cases_ratio.iloc[element]["Fatalities_difference"]/US_cases_ratio.iloc[element-1]["Fatalities"]



for element in range(1, MX_cases_ratio.shape[0]):

    MX_cases_ratio.at[MX_cases_ratio.index[element], "Fatalities_difference"] = MX_cases_ratio.iloc[element]["Fatalities"]-MX_cases_ratio.iloc[element-1]["Fatalities"]

    MX_cases_ratio.at[MX_cases_ratio.index[element], "Fatalities_trend"] = MX_cases_ratio.iloc[element]["Fatalities_difference"]/MX_cases_ratio.iloc[element-1]["Fatalities"]



for element in range(1, CN_cases_ratio.shape[0]):

    CN_cases_ratio.at[CN_cases_ratio.index[element], "Fatalities_difference"] = CN_cases_ratio.iloc[element]["Fatalities"]-CN_cases_ratio.iloc[element-1]["Fatalities"]

    CN_cases_ratio.at[CN_cases_ratio.index[element], "Fatalities_trend"] = CN_cases_ratio.iloc[element]["Fatalities_difference"]/CN_cases_ratio.iloc[element-1]["Fatalities"]



for element in range(1, CH_cases_ratio.shape[0]):

    CH_cases_ratio.at[CH_cases_ratio.index[element], "Fatalities_difference"] = CH_cases_ratio.iloc[element]["Fatalities"]-CH_cases_ratio.iloc[element-1]["Fatalities"]

    CH_cases_ratio.at[CH_cases_ratio.index[element], "Fatalities_trend"] = CH_cases_ratio.iloc[element]["Fatalities_difference"]/CH_cases_ratio.iloc[element-1]["Fatalities"]



for element in range(1, IN_cases_ratio.shape[0]):

    IN_cases_ratio.at[IN_cases_ratio.index[element], "Fatalities_difference"] = IN_cases_ratio.iloc[element]["Fatalities"]-IN_cases_ratio.iloc[element-1]["Fatalities"]

    IN_cases_ratio.at[IN_cases_ratio.index[element], "Fatalities_trend"] = IN_cases_ratio.iloc[element]["Fatalities_difference"]/IN_cases_ratio.iloc[element-1]["Fatalities"]



for element in range(1, IT_cases_ratio.shape[0]):

    IT_cases_ratio.at[IT_cases_ratio.index[element], "Fatalities_difference"] = IT_cases_ratio.iloc[element]["Fatalities"]-IT_cases_ratio.iloc[element-1]["Fatalities"]

    IT_cases_ratio.at[IT_cases_ratio.index[element], "Fatalities_trend"] = IT_cases_ratio.iloc[element]["Fatalities_difference"]/IT_cases_ratio.iloc[element-1]["Fatalities"]

print("US fatalities average growth everyday: "+str(round(US_cases_ratio[1:]["Fatalities_trend"].mean(axis=0)*100,2))+"%")

print("Mexico fatalities average growth everyday: "+str(round(MX_cases_ratio[1:]["Fatalities_trend"].mean(axis=0)*100,2))+"%")

print("Canada fatalities average growth everyday: "+str(round(CN_cases_ratio[1:]["Fatalities_trend"].mean(axis=0)*100,2))+"%")

print("China fatalities average growth everyday: "+str(round(CH_cases_ratio[1:]["Fatalities_trend"].mean(axis=0)*100,2))+"%")

print("Iran fatalities average growth everyday: "+str(round(IN_cases_ratio[1:]["Fatalities_trend"].mean(axis=0)*100,2))+"%")

print("Italy fatalities average growth everyday: "+str(round(IT_cases_ratio[1:]["Fatalities_trend"].mean(axis=0)*100,2))+"%")
US_cases_ratio1 = US_cases[datetime.date(2020, 3, 10):]

MX_cases_ratio1 = MX_cases[datetime.date(2020, 2, 28):]

CN_cases_ratio1 = CN_cases[datetime.date(2020, 1, 26):]

CH_cases_ratio1 = CH_cases[:datetime.date(2020, 3, 22)]

IT_cases_ratio1 = IT_cases[datetime.date(2020, 1, 31):]

IN_cases_ratio1 = IN_cases[datetime.date(2020, 2, 19):]

US_cases_ratio1.columns = US_cases_ratio1.columns.droplevel(level=1)

MX_cases_ratio1.columns = MX_cases_ratio1.columns.droplevel(level=1)

CN_cases_ratio1.columns = CN_cases_ratio1.columns.droplevel(level=1)

CH_cases_ratio1.columns = CH_cases_ratio1.columns.droplevel(level=1)

IT_cases_ratio1.columns = IT_cases_ratio1.columns.droplevel(level=1)

IN_cases_ratio1.columns = IN_cases_ratio1.columns.droplevel(level=1)

US_cases_ratio1 = US_cases_ratio1.reset_index()

MX_cases_ratio1 = MX_cases_ratio1.reset_index()

CN_cases_ratio1 = CN_cases_ratio1.reset_index()

CH_cases_ratio1 = CH_cases_ratio1.reset_index()

IT_cases_ratio1 = IT_cases_ratio1.reset_index()

IN_cases_ratio1 = IN_cases_ratio1.reset_index()



US_cases_ratio1["Confirmed_difference"] = 0

US_cases_ratio1["Confirmed_trend"] = 0.0

MX_cases_ratio1["Confirmed_difference"] = 0

MX_cases_ratio1["Confirmed_trend"] = 0.0

CN_cases_ratio1["Confirmed_difference"] = 0

CN_cases_ratio1["Confirmed_trend"] = 0.0

CH_cases_ratio1["Confirmed_difference"] = 0

CH_cases_ratio1["Confirmed_trend"] = 0.0

IT_cases_ratio1["Confirmed_difference"] = 0

IT_cases_ratio1["Confirmed_trend"] = 0.0

IN_cases_ratio1["Confirmed_difference"] = 0

IN_cases_ratio1["Confirmed_trend"] = 0.0
for element in range(1, US_cases_ratio1.shape[0]):

    US_cases_ratio1.at[US_cases_ratio1.index[element], "Confirmed_difference"] = US_cases_ratio1.iloc[element]["ConfirmedCases"]-US_cases_ratio1.iloc[element-1]["ConfirmedCases"]

    US_cases_ratio1.at[US_cases_ratio1.index[element], "Confirmed_trend"] = US_cases_ratio1.iloc[element]["Confirmed_difference"]/US_cases_ratio1.iloc[element-1]["ConfirmedCases"]



for element in range(1, MX_cases_ratio1.shape[0]):

    MX_cases_ratio1.at[MX_cases_ratio1.index[element], "Confirmed_difference"] = MX_cases_ratio1.iloc[element]["ConfirmedCases"]-MX_cases_ratio1.iloc[element-1]["ConfirmedCases"]

    MX_cases_ratio1.at[MX_cases_ratio1.index[element], "Confirmed_trend"] = MX_cases_ratio1.iloc[element]["Confirmed_difference"]/MX_cases_ratio1.iloc[element-1]["ConfirmedCases"]



for element in range(1, CN_cases_ratio1.shape[0]):

    CN_cases_ratio1.at[CN_cases_ratio1.index[element], "Confirmed_difference"] = CN_cases_ratio1.iloc[element]["ConfirmedCases"]-CN_cases_ratio1.iloc[element-1]["ConfirmedCases"]

    CN_cases_ratio1.at[CN_cases_ratio1.index[element], "Confirmed_trend"] = CN_cases_ratio1.iloc[element]["Confirmed_difference"]/CN_cases_ratio1.iloc[element-1]["ConfirmedCases"]



for element in range(1, CH_cases_ratio1.shape[0]):

    CH_cases_ratio1.at[CH_cases_ratio1.index[element], "Confirmed_difference"] = CH_cases_ratio1.iloc[element]["ConfirmedCases"]-CH_cases_ratio1.iloc[element-1]["ConfirmedCases"]

    CH_cases_ratio1.at[CH_cases_ratio1.index[element], "Confirmed_trend"] = CH_cases_ratio1.iloc[element]["Confirmed_difference"]/CH_cases_ratio1.iloc[element-1]["ConfirmedCases"]

    

for element in range(1, IT_cases_ratio1.shape[0]):

    IT_cases_ratio1.at[IT_cases_ratio1.index[element], "Confirmed_difference"] = IT_cases_ratio1.iloc[element]["ConfirmedCases"]-IT_cases_ratio1.iloc[element-1]["ConfirmedCases"]

    IT_cases_ratio1.at[IT_cases_ratio1.index[element], "Confirmed_trend"] = IT_cases_ratio1.iloc[element]["Confirmed_difference"]/IT_cases_ratio1.iloc[element-1]["ConfirmedCases"]

for element in range(1, IN_cases_ratio1.shape[0]):

    IN_cases_ratio1.at[IN_cases_ratio1.index[element], "Confirmed_difference"] = IN_cases_ratio1.iloc[element]["ConfirmedCases"]-IN_cases_ratio1.iloc[element-1]["ConfirmedCases"]

    IN_cases_ratio1.at[IN_cases_ratio1.index[element], "Confirmed_trend"] = IN_cases_ratio1.iloc[element]["Confirmed_difference"]/IN_cases_ratio1.iloc[element-1]["ConfirmedCases"]

   
print("US average infected growth everyday: "+str(round(US_cases_ratio1[1:]["Confirmed_trend"].mean(axis=0)*100,2))+"%")

print("Mexico average infected growth everyday: "+str(round(MX_cases_ratio1[1:]["Confirmed_trend"].mean(axis=0)*100,2))+"%")

print("Canada average infected growth everyday: "+str(round(CN_cases_ratio1[1:]["Confirmed_trend"].mean(axis=0)*100,2))+"%")

print("China average infected growth everyday: "+str(round(CH_cases_ratio1[1:]["Confirmed_trend"].mean(axis=0)*100,2))+"%")

print("Iran average infected growth everyday: "+str(round(IN_cases_ratio1[1:]["Confirmed_trend"].mean(axis=0)*100,2))+"%")

print("Italy average infected growth everyday: "+str(round(IT_cases_ratio1[1:]["Confirmed_trend"].mean(axis=0)*100,2))+"%")
US_cases_ratio["Mortality_rate"] = US_cases_ratio["Fatalities"]/US_cases_ratio["ConfirmedCases"]

MX_cases_ratio["Mortality_rate"] = MX_cases_ratio["Fatalities"]/MX_cases_ratio["ConfirmedCases"]

CN_cases_ratio["Mortality_rate"] = CN_cases_ratio["Fatalities"]/CN_cases_ratio["ConfirmedCases"]

CH_cases_ratio["Mortality_rate"] = CH_cases_ratio["Fatalities"]/CH_cases_ratio["ConfirmedCases"]

IN_cases_ratio["Mortality_rate"] = IN_cases_ratio["Fatalities"]/IN_cases_ratio["ConfirmedCases"]

IT_cases_ratio["Mortality_rate"] = IT_cases_ratio["Fatalities"]/IT_cases_ratio["ConfirmedCases"]
print("US average mortality rate: "+str(round(US_cases_ratio[:]["Mortality_rate"].mean(axis=0)*100,2))+"%")

print("Mexico average mortality rate: "+str(round(MX_cases_ratio[:]["Mortality_rate"].mean(axis=0)*100,2))+"%")

print("Canada average mortality rate: "+str(round(CN_cases_ratio[:]["Mortality_rate"].mean(axis=0)*100,2))+"%")

print("China average mortality rate: "+str(round(CH_cases_ratio[:]["Mortality_rate"].mean(axis=0)*100,2))+"%")

print("Iran average mortality rate: "+str(round(IN_cases_ratio[:]["Mortality_rate"].mean(axis=0)*100,2))+"%")

print("Italy average mortality rate: "+str(round(IT_cases_ratio[:]["Mortality_rate"].mean(axis=0)*100,2))+"%")
population = pd.read_csv("/kaggle/input/population-by-country-2020/population_by_country_2020.csv", usecols=["Country (or dependency)","Population (2020)", "Density (P/Km²)","Migrants (net)","Med. Age","Urban Pop %"], na_values=["","N.A."])
population.info()

population.shape
# Change columns names

population.columns = ["Country_Region","Population", "Density","Migrants","Med_Age","Urban_Pop"]

# Convert string type percentages to floats

population["Urban_Pop"] = population["Urban_Pop"].str.replace(r"%",r".0").astype("float")/100.0

# Replace missing values with mean

population = population.fillna(population.mean())

population.info()
population = population.sort_values("Country_Region",ascending=True).reset_index(drop=True)

population.at[population.index[224],"Country_Region"] = "US"

population.at[population.index[55],"Country_Region"] = "Czechia"

population.at[population.index[46],"Country_Region"] = "Congo (Brazzaville)"

population.at[population.index[194],"Country_Region"] = "Korea, South"

population.at[population.index[175],"Country_Region"] = "Saint Kitts and Nevis"

population.at[population.index[198],"Country_Region"] = "Saint Vincent and the Grenadines"

population.at[population.index[205],"Country_Region"] = "Taiwan*"
train = pd.merge(train, population, on="Country_Region")
print ("\nMissing values in train:  ", train.isnull().sum().values.sum())

print ("\nMissing values in test:  ", test.isnull().sum().values.sum())
train_df = train.groupby(["Date","Country_Region"], as_index=False).agg({"ConfirmedCases":["sum"],"Fatalities":["sum"]})

train_df.columns = list(map("".join, train_df.columns.values))

train_df.columns = ["Date","Country_Region","ConfirmedCases","Fatalities"]

train_df = pd.merge(train_df, population, on="Country_Region")

train_df
print ("\nMissing values :  ", train_df.isnull().sum().values.sum())

print ("\nMissing values :  ", test.isnull().sum().values.sum())
encoder = preprocessing.LabelEncoder()

train_df['Day_No'] = (encoder.fit_transform(train_df.Date))

test['Day_No'] = (encoder.fit_transform(test.Date))
a = list(train_df["ConfirmedCases"])

a.sort()

b = list(train_df["Fatalities"])

stat_a, p = shapiro(a)

stat_b, p_b = shapiro(b)

print('Confirmed Cases Statistics=%.3f, p=%.3f' % (stat_a, p))

if p > 0.05:

    print('Sample looks Gaussian (fail to reject H0)')

else:

    print('Sample does not look Gaussian (reject H0)')

print('Fatalities Statistics=%.3f, p=%.3f' % (stat_b, p_b))

if p > 0.05:

    print('Sample looks Gaussian (fail to reject H0)')

else:

    print('Sample does not look Gaussian (reject H0)')
qqplot(train_df["ConfirmedCases"], line='s')

plt.title("Confirmed Cases")

qqplot(train_df["Fatalities"], line="s")

plt.title("Fatalities")

plt.show()
corr = train_df.corr(method="spearman")

corr.style.background_gradient(cmap='coolwarm')
CH_df = train_df[train_df.Country_Region.isin(["China"])]

IT_df = train_df[train_df.Country_Region.isin(["Italy"])]

US_df = train_df[train_df.Country_Region.isin(["US"])]

IN_df = train_df[train_df.Country_Region.isin(["Iran"])]

MX_df = train_df[train_df.Country_Region.isin(["Mexico"])]

CN_df = train_df[train_df.Country_Region.isin(["Canada"])]
plt.plot(CH_df["Date"],CH_df["ConfirmedCases"])

plt.plot(IT_df["Date"],IT_df["ConfirmedCases"])

plt.plot(US_df["Date"],US_df["ConfirmedCases"])

plt.plot(IN_df["Date"],IN_df["ConfirmedCases"])

plt.plot(MX_df["Date"],MX_df["ConfirmedCases"])

plt.plot(CN_df["Date"],CN_df["ConfirmedCases"])

plt.legend(["China", "Italy", "US","Iran","Mexico","Canada"], loc='upper left')

plt.title("Confirmed cases of COVID19", size=15)

plt.xlabel("Date", size=13)

plt.ylabel("Number of cases", size=13)

plt.show()
train_df = train.groupby(["Date"], as_index=False).agg({"ConfirmedCases":["sum"],"Fatalities":["sum"]})

train_df.columns = list(map("".join, train_df.columns.values))

train_df.columns = ["Date","ConfirmedCases","Fatalities"]

encoder = preprocessing.LabelEncoder()

train_df['Day_No'] = (encoder.fit_transform(train_df.Date))
test_df = test[["Day_No"]]
# Train x,y

x_train_wd = list(train_df["Day_No"])

y_train_wd = list(train_df["ConfirmedCases"])

# Test x

x_test_wd = list(test_df["Day_No"])
# Logistic Curve Equation

def log_curve(x, k, x_0, L):

    return L / (1 + np.exp(-k*(x-x_0)))
# Fit the model

fit_par, fit_cov = curve_fit(log_curve, x_train_wd, y_train_wd, bounds=([0,0,0],np.inf), maxfev=100000)

estimated_k_wd, estimated_x0_wd, L_wd = fit_par

# Create the predictions

predictions_con = log_curve(x_test_wd, estimated_k_wd, estimated_x0_wd, L_wd)
# Train with Fatalities

y_train_wd = list(train_df["Fatalities"])
# Fit the model

fit_par, fit_cov = curve_fit(log_curve, x_train_wd, y_train_wd, bounds=([0,0,0],np.inf), maxfev=100000)

estimated_k_wd, estimated_x0_wd, L_wd = fit_par

# Create the predictions

predictions_fat = log_curve(x_test_wd, estimated_k_wd, estimated_x0_wd, L_wd)
submission = pd.DataFrame(test["ForecastId"])
submission['ConfirmedCases'] = predictions_con

submission['Fatalities'] = predictions_fat

submission
submission.to_csv('submission.csv', index = False)