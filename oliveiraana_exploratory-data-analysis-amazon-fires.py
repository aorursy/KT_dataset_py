import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import MaxNLocator, FuncFormatter

import calendar

import folium

from folium import plugins
filepath = "../input/forest-fires-in-brazil/amazon.csv"



amazon_dataframe = pd.read_csv(filepath, encoding = "latin1")



columns = amazon_dataframe.columns



shape = amazon_dataframe.shape



print(amazon_dataframe.head(10), "\n")

print("Number of rows: ", shape[0], "\n")

print("Number of columns: ",  shape[1], "\n")

print("Columns of the dataframe: ", columns)
amazon_dataframe.info()
months_renaming = {"Janeiro": "January",

                  "Fevereiro": "February",

                  "Março": "March",

                  "Abril": "April",

                  "Maio": "May",

                  "Junho": "June",

                  "Julho": "July",

                  "Agosto": "August",

                  "Setembro": "September",

                  "Outubro": "October",

                  "Novembro": "November",

                  "Dezembro": "December"}



amazon_dataframe = amazon_dataframe.replace(months_renaming)



amazon_dataframe.head()
amazon_dataframe["date"] = pd.to_datetime(amazon_dataframe["date"])



amazon_dataframe.date.dtype

number_duplicated_rows = amazon_dataframe.duplicated().sum()



print("Number of duplicated rows: ", number_duplicated_rows)



duplicated_rows = amazon_dataframe[amazon_dataframe.duplicated()]



print("Duplicated rows:\n ", duplicated_rows)

amazon_dataframe.drop_duplicates(inplace=True)



print("New dimensions of the dataset: ", amazon_dataframe.shape)
number_null_values = amazon_dataframe.isnull().sum()



print("Number of null values:\n", number_null_values)
print("Total of fires registed: ", amazon_dataframe.shape[0])
amazon_dataframe.describe()
plt.hist(amazon_dataframe["number"], bins=100, edgecolor ="k")

plt.xlabel("Number of fires")

plt.ylabel("Frequency")

plt.title("Fires Distribution")

fires_year = amazon_dataframe.groupby(amazon_dataframe["year"]).count().number

print(fires_year)
plt.figure(figsize=(12,7))

plot = sns.lineplot(data=amazon_dataframe, x="year", y="number", markers=True)

plot.xaxis.set_major_locator(plt.MaxNLocator(19))

plot.set_xlim(1998, 2017)
month_fires = amazon_dataframe.groupby(amazon_dataframe["month"]).number.count().reset_index()

month_fires.sort_values("number", ascending=False)

print(month_fires)

plt.style.use("ggplot")



month_fires.plot(x="month", y="number", kind="bar", figsize=(12,7), color="orange", alpha = 0.5)



plt.title("Distribution of fires by month")

plt.xlabel("Month", fontsize=16)

plt.ylabel("Number of fires", fontsize=16)
fires_weekday = amazon_dataframe.groupby(amazon_dataframe["date"].dt.dayofweek).count().date



fires_weekday.index = [calendar.day_name[x] for x in range(0,7)]

print(fires_weekday)
plt.style.use("ggplot")



fires_weekday.plot(kind="bar", figsize=(12,7), color="orange", alpha = 0.5)



plt.title("Distribution of fires by day of the week")

plt.xlabel("Day of the week", fontsize=16)

plt.ylabel("Number of fires", fontsize=16)
fires_state = amazon_dataframe.groupby(amazon_dataframe["state"]).count().number

print(fires_state)
plt.style.use("ggplot")



fires_state.plot(kind="bar", figsize=(12,7), color="orange", alpha=0.5)



plt.title("Distribution of fires by state")

plt.xlabel("State", fontsize=16)

plt.ylabel("Number of fires", fontsize=16)
mato_grosso_dataframe = amazon_dataframe[amazon_dataframe["state"] == "Mato Grosso"]

print(mato_grosso_dataframe.head())



number_fires_mato_grosso = mato_grosso_dataframe.shape[0]

print("\nNumber of fires in Mato Grosso State: ", number_fires_mato_grosso)

year_fires_mato_grosso = mato_grosso_dataframe.groupby(mato_grosso_dataframe["year"]).count().number

print(year_fires_mato_grosso)
plt.style.use("ggplot")



year_fires_mato_grosso.plot(kind="bar", figsize=(12,7), color="black", alpha=0.5)



plt.title("Distribution of fires in Mato Grosso State by Year")

plt.xlabel("Year", fontsize=16)

plt.ylabel("Number of fires", fontsize=16)
month_fires_mato_grosso = mato_grosso_dataframe.groupby(mato_grosso_dataframe["month"]).count().number

print(month_fires_mato_grosso)
plt.style.use("ggplot")



month_fires_mato_grosso.plot(kind="bar", figsize=(12,7), color="pink", alpha=0.5)



plt.title("Distribution of fires in Mato Grosso State by Month")

plt.xlabel("Month", fontsize=16)

plt.ylabel("Number of fires", fontsize=16)
day_week_fires_mato_grosso = mato_grosso_dataframe.groupby(mato_grosso_dataframe["date"].dt.dayofweek).count().date

day_week_fires_mato_grosso.index = [calendar.day_name[x] for x in range(0,7)]

print(day_week_fires_mato_grosso)
plt.style.use("ggplot")



day_week_fires_mato_grosso.plot(kind="bar", figsize=(12,7), color="yellow", alpha=0.5)



plt.title("Distribution of fires in Mato Grosso State by Day of Week")

plt.xlabel("Day of Week", fontsize=16)

plt.ylabel("Number of fires", fontsize=16)
paraiba_fires_dataframe = amazon_dataframe[amazon_dataframe["state"] == "Paraiba"]

print(paraiba_fires_dataframe.head())



number_fires_paraiba = paraiba_fires_dataframe.shape[0]

print("Number of fires in Paraiba: ", number_fires_paraiba)
year_fires_paraiba = paraiba_fires_dataframe.groupby(paraiba_fires_dataframe["year"]).count().number

print(year_fires_paraiba)
plt.style.use("ggplot")



year_fires_paraiba.plot(kind="bar", figsize=(12,7), color="blue", alpha=0.5)



plt.title("Distribution of fires in Paraiba State by Year")

plt.xlabel("Year", fontsize=16)

plt.ylabel("Number of fires", fontsize=16)
month_fires_paraiba = paraiba_fires_dataframe.groupby(paraiba_fires_dataframe["month"]).count().number

print(month_fires_paraiba)
plt.style.use("ggplot")



month_fires_paraiba.plot(kind="bar", figsize=(12,7), color="red", alpha=0.5)



plt.title("Distribution of fires in Paraiba State by Month")

plt.xlabel("Month", fontsize=16)

plt.ylabel("Number of fires", fontsize=16)
day_week_fires_paraiba = paraiba_fires_dataframe.groupby(paraiba_fires_dataframe["date"].dt.dayofweek).count().date

day_week_fires_paraiba.index = [calendar.day_name[x] for x in range(0,7)]

print(day_week_fires_paraiba)

plt.style.use("ggplot")



day_week_fires_paraiba.plot(kind="bar", figsize=(12,7), color="gray", alpha=0.5)



plt.title("Distribution of fires in Paraiba State by Day of Week")

plt.xlabel("Day of Week", fontsize=16)

plt.ylabel("Number of fires", fontsize=16)
rio_fires_dataframe = amazon_dataframe[amazon_dataframe["state"] == "Rio"]

print(rio_fires_dataframe.head())



number_fires_rio = rio_fires_dataframe.shape[0]

print("Number of fires in Rio: ", number_fires_rio)
year_fires_rio = rio_fires_dataframe.groupby(rio_fires_dataframe["year"]).count().number

print(year_fires_rio)
plt.style.use("ggplot")



year_fires_rio.plot(kind="bar", figsize=(12,7), color="brown", alpha=0.5)



plt.title("Distribution of fires in Rio State by Year")

plt.xlabel("Year", fontsize=16)

plt.ylabel("Number of fires", fontsize=16)
month_fires_rio = rio_fires_dataframe.groupby(rio_fires_dataframe["month"]).count().number

print(month_fires_rio)
plt.style.use("ggplot")



month_fires_rio.plot(kind="bar", figsize=(12,7), color="blue", alpha=0.5)



plt.title("Distribution of fires in Rio State by Month")

plt.xlabel("Month", fontsize=16)

plt.ylabel("Number of fires", fontsize=16)
day_week_fires_rio = rio_fires_dataframe.groupby(rio_fires_dataframe["date"].dt.dayofweek).count().date

day_week_fires_rio.index = [calendar.day_name[x] for x in range(0,7)]

print(day_week_fires_rio)
plt.style.use("ggplot")



day_week_fires_rio.plot(kind="bar", figsize=(12,7), color="green", alpha=0.5)



plt.title("Distribution of fires in Rio State by Day of Week")

plt.xlabel("Day of Week", fontsize=16)

plt.ylabel("Number of fires", fontsize=16)
latitude={

    'Acre':-9.02,'Alagoas':-9.57,'Amapa':02.05,'Amazonas':-5.00,'Bahia':-12.00,'Ceara':-5.00,

          

    'Distrito Federal':-15.45,'Espirito Santo':-20.00,'Goias':-15.55,'Maranhao':-5.00,'Mato Grosso':-14.00

      

    ,'Minas Gerais':-18.50,'Pará':-3.20,'Paraiba':-7.00,'Pernambuco':-8.00,'Piau':-7.00,'Rio':-22.90,

          

    'Rondonia':-11.00,'Roraima':-2.00,'Santa Catarina':-27.25,'Sao Paulo':-23.32,'Sergipe':-10.30,

         

    'Tocantins':-10.00

    }





longitude={

    'Acre':-70.8120,'Alagoas':-36.7820,'Amapa':-50.50,'Amazonas':-65.00,'Bahia':-42.00,'Ceara':-40.00,

    

    'Distrito Federal':-47.45,'Espirito Santo':-40.45,'Goias':-50.10,'Maranhao':-46.00,'Mato Grosso':-55.00,

    

    'Minas Gerais':-46.00,'Pará':-52.00,'Paraiba':-36.00,'Pernambuco':-37.00,'Piau':-73.00, 'Rio':-43.17,

    

    'Rondonia':-63.00,'Roraima':-61.30,'Santa Catarina':-48.30,'Sao Paulo':-46.37,'Sergipe':-37.30,

    

    'Tocantins':-48.00

    }



amazon_dataframe["latitude"] = amazon_dataframe["state"].map(latitude)

amazon_dataframe["longitude"] = amazon_dataframe["state"].map(longitude)

amazon_dataframe.head()


brasil_map = folium.Map(location=[-16.1237611, -59.9219642], zoom_start=3.5, tiles='Stamen Terrain')

brasil_map



fires = plugins.MarkerCluster().add_to(brasil_map) 



for latitude, longitude in zip(amazon_dataframe.latitude, amazon_dataframe.longitude):

    folium.Marker(

        location=[latitude, longitude],

        icon=None,

    ).add_to(fires)

    

brasil_map