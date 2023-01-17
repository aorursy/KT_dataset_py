import numpy as np

import pandas as pd

import plotly.graph_objects as go

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import os

import plotly_express as px

from collections import Counter

import json

from datetime import timedelta, datetime

import time

from tqdm import tqdm

import pycountry

from geopy.geocoders import Nominatim
directory = "../input"



def load(file):

    return pd.read_csv(os.path.join(directory, file))



df_containment = load("covid19-containment-and-mitigation-measures/COVID 19 Containment measures data.csv")

df_containment.head()



class CountryDict:

    def __init__(self, countries):

        self.countryToCountrycode = {country: pycountry.countries.get(name=country).alpha_3 for country in countries if pycountry.countries.get(name=country) is not None}

        self.countrycodeToCountry = {pycountry.countries.get(name=country).alpha_3: country for country in countries if pycountry.countries.get(name=country) is not None}

        self.countryToidx = {pycountry.countries.get(name=country).alpha_3: idx for idx, country in enumerate(countries) if pycountry.countries.get(name=country) is not None}

        self.idxToCountry = {idx: pycountry.countries.get(name=country).alpha_3 for idx, country in enumerate(countries) if pycountry.countries.get(name=country) is not None}



    def getIdx(self, country):

        if len(country) == 3:

            country = self.countryToCountrycode[country]

        return self.countryToidx[country]



    def getCountry(self, index):

        return self.idxToCountry[index]



class DataExtractor():

    def __init__(self, directory, files):

        self.dir = directory

        self.files = files

        

    def load_file(self, case_type):

        file = [x for x in self.files if case_type.lower() in x and not "US" in x][0]

        return pd.read_csv(os.path.join(self.dir, file))



    def sameCoulumns(self, df1, df2, df3):

        first_and_second = set(df1) == set(df2)

        second_and_third = set(df2) == set(df3)



        if not first_and_second or not second_and_third:

            raise Exception("Columns not the same")

        else:

            return True



    def clean(self, x):

        return 1



    def create_data(self):

        df_conf = self.load_file("Confirmed")

        df_death = self.load_file("Deaths")

        df_recov = self.load_file("Recovered")

        

        df_conf["Province/State"] = df_conf["Province/State"].fillna(value="None")

        df_death["Province/State"] = df_death["Province/State"].fillna(value="None")

        df_recov["Province/State"] = df_recov["Province/State"].fillna(value="None")

        print(df_conf.head())





        self.sameCoulumns(df_conf.columns, df_death.columns, df_recov.columns)



        dates = df_conf.columns[4:]

        all_data = {}



        for n in range(len(df_conf)):

            conf_row = df_conf.loc[n]

            country, province = conf_row["Country/Region"], conf_row["Province/State"]



            death_row = df_death.loc[(df_death["Country/Region"] == country) & (df_death["Province/State"] == province)]

            recov_row = df_recov.loc[(df_recov["Country/Region"] == country) & (df_recov["Province/State"] == province)]



            for date in dates:

                confs = conf_row[date].item()

                deaths = death_row[date].item()

                try:

                    recovs = recov_row[date].item()

                except:

                    recovs = 0



                unix = time.mktime(datetime.strptime(date, "%m/%d/%y").timetuple())

                date = datetime.fromtimestamp(unix).strftime("%Y/%m/%d")



                if not country in all_data.keys():

                    all_data[country] = {}



                if not province in all_data[country].keys():

                    all_data[country][province] = {}



                if not date in all_data[country][province].keys():

                    all_data[country][province][date] = {

                        "Confirmed": confs,

                        'Deaths': deaths,

                        'Recovered': recovs,

                    }

        json.dump(all_data, open("../data/data.json","w"), indent=2)

        return data





# Extract information from dataset

#de = DataExtractor("../input/novel-corona-virus-2019-dataset", [x for x in os.listdir("../input/novel-corona-virus-2019-dataset") if "time_series" in x])

#oldest_data = de.create_data()

keywords = df_containment["Keywords"]

keywords.dropna(inplace=True)



data = []



for kw in keywords:

    for k in kw.split(","):

        data.append(k.lstrip())



measures = Counter(data)

measures
def create_json(df, fp):

    data = {}

    for country in set(df["Country"].values):

        if type(country) == float:

            continue

        sub_df = df.loc[df["Country"] == country][["Date Start", "Keywords"]]

        sub_df = sub_df[(sub_df["Date Start"].notna()) & (sub_df["Keywords"].notna())]



        per_day_data = {}



        for date, measure in sub_df[["Date Start" ,"Keywords"]].values:

            measures_taken = []

            for m in measure.split(","):

                measures_taken.append(m.lstrip())



            per_day_data[date] = {"Measures Taken": measures_taken, "Total Measures Taken": len(measures_taken)}



        data[country] = per_day_data

    

    os.mkdir("../data") if not os.path.exists("../data") else None

    json.dump(data, open(fp, "w"), indent=2)



# Create json file

#create_json(df_containment, "../data/mitigation_data.json")



def load_json(fp):

    return json.load(open(fp, "r"))



data = load_json("../input/coronavirus-cases-mitigation/mitigation_data.json")
def fixUSStates(data):

    US_states = list(filter(lambda k: ":" in k, data.keys()))

    US_data = {}



    tmp = []



    for state in US_states:

        for date, measure_data in data[state].items():

            if date not in US_data.keys():

                US_data[date] = measure_data

            else:

                for measure in measure_data["Measures Taken"]:

                    US_data[date]["Measures Taken"].append(measure)



                US_data[date]["Total Measures Taken"] += measure_data["Total Measures Taken"]



        del data[state]

    

    data["USA"] = US_data

    return data



def getAllDates(data):

    dates = []

    for country in data.values():

        for k in country.keys():

            t = time.mktime(datetime.strptime(k, "%b %d, %Y").timetuple())

            uniform_date = datetime.utcfromtimestamp(t).strftime("%Y/%m/%d")

            dates.append(uniform_date)



    dates = sorted(dates)

    from_date = dates[0]



    from_date = datetime.strptime(from_date, "%Y/%m/%d")

    today = datetime.today()



    delta = today-from_date



    dates = []

    for i in range(delta.days + 1):

        day = from_date + timedelta(days=i)

        dates.append(day)

    return dates





def fix_country(country):

    if country == "Vietnam":

        country = "Viet Nam"

    if country == "Vatican City":

        country = "Holy See (Vatican City State)"

    if country == "Iran":

        country = "Iran, Islamic Republic of"

    if country == "Russia":

        country = "Russian Federation"

    if country == "Taiwan":

        country = "Taiwan, Province of China"

    if country == "Macedonia":

        country = "North Macedonia"

    if country == "Moldova":

        country = "Moldova, Republic of"

    if country == "South Korea" or country == "North Korea":

        country = "Korea, Democratic People's Republic of"

    if country == "USA":

        country = "United States"



    return country
def make_data():

    data = load_json("../input/coronavirus-cases-mitigation/mitigation_data.json")

    data = fixUSStates(data)

    dates = getAllDates(data)

    df = pd.DataFrame()

    for country in tqdm(data.keys()):

        country_name = country

        country = fix_country(country)

        country = pycountry.countries.get(name=country)



        if country == None:

            continue

        else:

            country_code = country.alpha_3



        # Removing duplicate data

        if country.name == "United States" and country_name == "United States":

            continue



        num_measures = 0

        for date in dates:

            measures_on_the_day = 'No new mitigation'



            date = date.strftime("%b %d, %Y")



            if date in data[country_name].keys():

                num_measures += data[country_name][date]["Total Measures Taken"]

                measures_on_the_day = data[country_name][date]["Measures Taken"]

                measures_on_the_day = "".join([m+" | " for m in measures_on_the_day])



            df = df.append({"Country Code": country_code, "Date": date, "Num Measures": num_measures, "Measures": measures_on_the_day}, ignore_index=True)

    df.to_csv("../input/coronavirus-cases-mitigation/mitigation_date_data.csv")



# Makes dataframe data

#make_data()
def make_animation(scope="world"):

    data = load_json("../input/coronavirus-cases-mitigation/mitigation_data.json")

    dates = getAllDates(data)

    df = pd.read_csv("../input/coronavirus-cases-mitigation/mitigation_date_data.csv")



    fig = px.choropleth(

        df,

        locations="Country Code",

        color="Num Measures",

        hover_name="Measures",

        animation_frame="Date",

        color_continuous_scale=px.colors.sequential.matter,

        range_color=(0, 50),

        scope=scope

    )

    

    fig.layout.update(

        title="Number of mitigation measure taken by different countries over time",

    )

    fig.show()



make_animation()
def getPairPerTime(dates, days):

    d = 0

    while (len(dates[d:])) % (days) != 0:

        dates.pop(0)

    weeks = []

    for n in np.arange(0, len(dates), days):

        if n+1 == len(dates):

            break

        weeks.append([dates[n], dates[(n-1)+days]])

    return weeks



def get_cases_per_unit_time(province_data, dates, days=7):

    weeks = getPairPerTime(dates, days)

    data = {"Confirmed": {}, "Deaths": {}, "Recovered": {}}

    for n, week in enumerate(weeks):

        data["Confirmed"][f"Time {n+1}"] = province_data[weeks[n][-1]]["Confirmed"]

        data["Deaths"][f"Time {n+1}"]  = province_data[weeks[n][-1]]["Deaths"]

        data["Recovered"][f"Time {n+1}"] = province_data[weeks[n][-1]]["Recovered"]

    return data



def get_new_cases_per_unit_time(province_data, dates, days=7):

    weeks = getPairPerTime(dates, days)

    data = {"Confirmed": {}, "Deaths": {}, "Recovered": {}}

    for n, week in enumerate(weeks):

        data["Confirmed"][f"Time {n+1}"] = province_data[weeks[n][1]]["Confirmed"] - province_data[weeks[n][0]]["Confirmed"]

        data["Deaths"][f"Time {n+1}"]  = province_data[weeks[n][1]]["Deaths"] - province_data[weeks[n][0]]["Deaths"]

        data["Recovered"][f"Time {n+1}"] = province_data[weeks[n][1]]["Recovered"] - province_data[weeks[n][0]]["Recovered"]

    return data




data = json.load(open("../input/coronavirus-cases-mitigation/data_countries_provinces_grouped.json", "r"))



df = pd.DataFrame(columns=["x", "y", "con", "t"])



X = []

Y = []

for country in list(data.keys()):

    country_data = data[country]



    dates = list(country_data.keys())



    x = list(get_cases_per_unit_time(country_data, dates, 7)["Confirmed"].values())

    y = list(get_new_cases_per_unit_time(country_data, dates, 7)["Confirmed"].values())



    x = np.array(x)

    x[x == 0] = 1



    y = np.array(y)

    y[y == 0] = 1





    sub_df = pd.DataFrame(columns=["x", "y", "con", "t"])

    sub_df["x"] = x

    sub_df["y"] = y

    sub_df["con"] = [country for _ in range(len(x))]

    sub_df["t"] = [t for t in range(len(x))]



    df = pd.concat([sub_df, df], ignore_index=True)



range_x = max(df["x"].values.flatten())*2

range_y = max(df["y"].values.flatten())*2





fig = px.scatter(df, x="x", y="y", hover_name="con", animation_frame="t", log_x=True, log_y=True, text="con", trendline="lowess",

                 range_x=[0.5, range_x], range_y=[0.5, range_y]

   )



fig.show()

geo = Nominatim(user_agent="Covid")

direc = "Data1"



def load_csv(fp):

    return pd.read_csv(fp)





def loadGeoDict():

    return json.load(open("../input/coronavirus-cases-mitigation/geo_dict.json", "r"))



geo_dict = loadGeoDict()



df = load_csv("../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")

df = df[["country", "city", "latitude", "longitude", "travel_history_location"]]

df.dropna(inplace=True)





df_tavel = df[["latitude", "longitude", "travel_history_location"]]



values = df_tavel.values



travel_dict = {}

count_dict = {}

for val in values:

    if val[2] not in travel_dict.keys():

        travel_dict[val[2]] = val[:2]

        count_dict[val[2]] = 5

    else:

        count_dict[val[2]] += 1



        



fig = go.Figure()





for city, value in travel_dict.items():



    if city not in geo_dict.keys():

        continue

    lives_CORDS = value

    visted_CORDS = geo_dict[city]

    

    fig.add_trace(go.Scattermapbox(

        mode="markers+lines",

        lon=[lives_CORDS[1], visted_CORDS[1]],

        lat=[lives_CORDS[0], visted_CORDS[0]],

        marker={'size': count_dict[city] if count_dict[city] < 30 else 30},

        name=city

    ))





fig.update_layout(

    margin ={'l':0,'t':0,'b':0,'r':0},

    mapbox = {

        'center': {'lon': 10, 'lat': 10},

        'style': "carto-positron",

        'center': {'lon': -20, 'lat': -20},

        'zoom': 1,

    })





fig.show()