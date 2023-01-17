## importing packages

import time



import numpy as np

import pandas as pd

import seaborn as sns



from bokeh.layouts import column, row

from bokeh.models import Panel, Tabs, LinearAxis, Range1d, BoxAnnotation, LabelSet, Span

from bokeh.models.tools import HoverTool

from bokeh.palettes import Category20, Spectral3, Spectral4, Spectral8

from bokeh.plotting import ColumnDataSource, figure, output_notebook, show

from bokeh.transform import dodge



from datetime import datetime as dt

from math import pi



output_notebook()

## defining constants

PATH_COVID = "/kaggle/input/covid19-in-india/covid_19_india.csv"

PATH_CENSUS = "/kaggle/input/covid19-in-india/population_india_census2011.csv"

PATH_TESTS = "/kaggle/input/covid19-in-india/ICMRTestingDetails.csv"

PATH_LABS = "/kaggle/input/covid19-in-india/ICMRTestingLabs.csv"

PATH_HOSPITALS = "/kaggle/input/covid19-in-india/HospitalBedsIndia.csv"

PATH_GLOBAL = "/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv"

PATH_METADATA = "/kaggle/input/covid19-forecasting-metadata/region_metadata.csv"

PATH_AQI = "/kaggle/input/air-quality-data-in-india/city_day.csv"



def read_covid_data():

    """

    Reads the main covid-19 India data and preprocesses it.

    """

    

    df = pd.read_csv(PATH_COVID)

    df.rename(columns = {

        "State/UnionTerritory": "state",

        "Confirmed": "cases",

        "Deaths": "deaths",

        "Cured": "recoveries"

    }, inplace = True)



    df.loc[df.state == "Telengana", "state"] = "Telangana"

    df["date"] = pd.to_datetime(df.Date, format = "%d/%m/%y").dt.date.astype(str)



    return df



def read_census_data():

    """

    Reads the 2011 Indian census data and preprocesses it.

    """

    

    df = pd.read_csv(PATH_CENSUS)

    df.rename(columns = {

        "State / Union Territory": "state",

        "Population": "population",

        "Urban population": "urban_population",

        "Gender Ratio": "gender_ratio"

    }, inplace = True)



    df["area"] = df.Area.str.replace(",", "").str.split("km").str[0].astype(int)



    return df



def read_test_samples_data():

    """

    Reads the ICMR test samples data and preprocesses it.

    """

    

    df = pd.read_csv(PATH_TESTS)

    df.drop(index = 0, inplace = True)

    df.rename(columns = {

        "TotalSamplesTested": "samples_tested"

    }, inplace = True)



    df["date"] = pd.to_datetime(df.DateTime, format = "%d/%m/%y %H:%S").dt.date.astype(str)

    

    return df



def read_test_labs_data():

    """

    Reads the ICMR testing labs data and preprocesses it.

    """

    

    df = pd.read_csv(PATH_LABS)

    

    return df



def read_hospitals_data():

    """

    Reads the Hospitals data and preprocesses it.

    """

    

    df = pd.read_csv(PATH_HOSPITALS)

    df.rename(columns = {

        "State/UT": "state"

    }, inplace = True)

    

    df.loc[df.state == "Andaman & Nicobar Islands", "state"] = "Andaman and Nicobar Islands"

    df.loc[df.state == "Jammu & Kashmir", "state"] = "Jammu and Kashmir"

    

    return df



def read_global_data():

    """

    Reads the global covid-19 data and preprocesses it.

    """

    

    df = pd.read_csv(PATH_GLOBAL)

    df_metadata = pd.read_csv(PATH_METADATA)

    

    df.rename(columns = {

        "Country/Region": "country",

        "Confirmed": "cases",

        "Deaths": "deaths",

        "Recovered": "recoveries"

    }, inplace = True)

    

    df_metadata.rename(columns = {

        "Country_Region": "country"

    }, inplace = True)



    df.loc[df.country == "Mainland China", "country"] = "China"

    df["date"] = pd.to_datetime(df.ObservationDate, format = "%m/%d/%Y").dt.date.astype(str)

    

    df = df.merge(df_metadata[["country", "continent"]].drop_duplicates(), on = "country", how = "left")

    

    return df



def read_aqi_data():

    """

    Reads AQI data and preprocesses it.

    """

    

    df = pd.read_csv(PATH_AQI)

    

    return df

df_aqi = read_aqi_data()



df = df_aqi[df_aqi.City.isin(["Ahmedabad", "Bengaluru", "Chennai", "Delhi", "Gurugram", "Hyderabad", "Jaipur",

                              "Kolkata", "Lucknow", "Mumbai", "Patna", "Thiruvananthapuram"])]

df = df[((df.Date >= "2019-04-01") & (df.Date < "2019-04-10")) | ((df.Date >= "2020-04-01") & (df.Date < "2020-04-10"))]

df["Year"] = pd.to_datetime(df.Date).dt.year

df = df.groupby(["City", "Year"])["AQI"].mean().reset_index()

df = df.pivot_table(index = "City", columns = "Year", values = "AQI", aggfunc = np.mean).reset_index()

df.rename(columns = {2019: "AQI_2019_April", 2020: "AQI_2020_April"}, inplace = True)

df["Improvement in AQI"] = round((df.AQI_2019_April - df.AQI_2020_April) * 100 / df.AQI_2019_April, 2)

df.AQI_2019_April = df.AQI_2019_April.astype(int).astype(str)

df.AQI_2020_April = df.AQI_2020_April.astype(int).astype(str)

df.sort_values("Improvement in AQI", ascending = False, ignore_index = True, inplace = True)

df.columns.name = None

cm = sns.light_palette("green", as_cmap = True)



df.style.background_gradient(cmap = cm)
df_covid = read_covid_data()

df_hospitals = read_hospitals_data()



df = df_covid.groupby("state")["cases", "recoveries", "deaths"].max().reset_index().merge(df_hospitals, on = "state")

df["active"] = df.cases - df.recoveries - df.deaths

df.fillna(0, inplace = True)

df.sort_values("NumPublicBeds_HMIS", ascending = False, inplace = True)

source = ColumnDataSource(data = dict(

    state = df.state.values,

    active = df.active.values,

    PublicBeds = df.NumPublicBeds_HMIS.values

))



tooltips = [

    ("State", "@state"),

    ("Active Cases", "@active"),

    ("Beds", "@PublicBeds")

]



v = figure(plot_width = 650,plot_height = 400, x_range = df.state.values, tooltips = tooltips, title = "Beds and Cases by State")



v.vbar(x = dodge("state", 0.15, range = v.x_range), top = "active", width = 0.2, source = source, color = "orange", legend_label = "Active Cases")

v.vbar(x = dodge("state", -0.15, range = v.x_range), top = "PublicBeds", width = 0.2, source = source, color = "green", legend_label = "Beds")



v.xaxis.major_label_orientation = -pi / 4



v.xaxis.axis_label = "State"

v.yaxis.axis_label = "Count"



v.legend.location = "top_right"



show(v)

df_covid = read_covid_data()

df_hospitals = read_hospitals_data()



df = df_covid.groupby("state")["cases", "recoveries", "deaths"].max().reset_index().merge(df_hospitals, on = "state")

df.fillna(0, inplace = True)

df["hospitals"] = df.NumSubDistrictHospitals_HMIS + df.NumDistrictHospitals_HMIS

df["healthcare_facilities"] = df.TotalPublicHealthFacilities_HMIS

df["cases_per_hospital"] = df.cases / df.hospitals

df["cases_per_healthcare_facility"] = df.cases / df.healthcare_facilities

df.sort_values("healthcare_facilities", ascending = False, inplace = True)

types = ["PrimaryHealthCenters", "CommunityHealthCenters", "SubDistrictHospitals", "DistrictHospitals"]



source_1 = ColumnDataSource(data = dict(

    state = df.state.values,

    PrimaryHealthCenters = df.NumPrimaryHealthCenters_HMIS.values,

    CommunityHealthCenters = df.NumCommunityHealthCenters_HMIS.values,

    SubDistrictHospitals = df.NumSubDistrictHospitals_HMIS.values,

    DistrictHospitals = df.NumDistrictHospitals_HMIS.values,

    hospitals = df.hospitals.values

))



tooltips_1 = [

    ("State", "@state"),

    ("Primary Health Centers", "@PrimaryHealthCenters"),

    ("Community Health Centers", "@CommunityHealthCenters"),

    ("Sub District Hospitals", "@SubDistrictHospitals"),

    ("District Hospitals", "@DistrictHospitals")

]



v1 = figure(plot_width = 650,plot_height = 400, x_range = df.state.values, tooltips = tooltips_1, title = "Healthcare Facilities by State")



v1.vbar_stack(types, x = "state", width = 0.9, color = Spectral4, source = source_1, legend_label = types)



v1.x_range.range_padding = 0.05

v1.xaxis.major_label_orientation = -pi / 4



v1.xaxis.axis_label = "State"

v1.yaxis.axis_label = "Count"

v1.legend.location = "top_right"



df.sort_values("cases_per_hospital", ascending = False, inplace = True)



source_2 = ColumnDataSource(data = dict(

    state = df.state.values,

    cases_per_healthcare_facility = df.cases_per_healthcare_facility.values,

    cases_per_hospital = df.cases_per_hospital.values

))



tooltips_21 = [

    ("Cases per Healthcare Facility", "@cases_per_healthcare_facility{0}")

]



tooltips_22 = [

    ("Cases per Hospital", "@cases_per_hospital{0}")

]



v2 = figure(plot_width = 650,plot_height = 400, x_range = df.state.values, title = "Cases per Facility / Hospital by State")



v21 = v2.vbar(x = dodge("state", 0.15, range = v2.x_range), top = "cases_per_healthcare_facility", width = 0.2, source = source_2, color = "green", legend_label = "Cases per Healthcare Facility")

v22 = v2.vbar(x = dodge("state", -0.15, range = v2.x_range), top = "cases_per_hospital", width = 0.2, source = source_2, color = "orange", legend_label = "Cases per Hospital")



v2.add_tools(HoverTool(renderers = [v21], tooltips = tooltips_21))

v2.add_tools(HoverTool(renderers = [v22], tooltips = tooltips_22))



v2.xaxis.major_label_orientation = -pi / 4



v2.xaxis.axis_label = "State"

v2.yaxis.axis_label = "Cases"



v2.legend.location = "top_right"



show(column(v1, v2))

df_covid = read_covid_data()

df_global = read_global_data()



df_country = df_global.groupby(["country", "continent"])["cases", "deaths", "recoveries"].max().reset_index()

df_country["completed_cases"] = df_country.deaths + df_country.recoveries

df_country["close_rate"] = df_country.completed_cases / df_country.cases

df_country.sort_values("close_rate", ascending = False, inplace = True)

df_world = pd.concat([df_country[df_country.cases >= 100].head(10), df_country[df_country.country == "India"]])

df_asia = pd.concat([df_country[(df_country.cases >= 100) & (df_country.continent == "Asia")].head(10), df_country[df_country.country == "India"]])



df_state = df_covid.groupby("state")["cases", "deaths", "recoveries"].max().reset_index()

df_state["completed_cases"] = df_state.deaths + df_state.recoveries

df_state["close_rate"] = df_state.completed_cases / df_state.cases

df_state.sort_values("close_rate", ascending = False, inplace = True)

source_1 = ColumnDataSource(data = dict(

    country = df_world.country.values,

    cases = df_world.cases.values,

    completed_cases = df_world.completed_cases.values,

    close_rate = df_world.close_rate.values * 100

))



tooltips_1 = [

    ("Cases", "@cases")

]



tooltips_2 = [

    ("Closed Cases", "@completed_cases")

]



tooltips_3 = [

    ("Close Rate", "@close_rate{0} %")

]



v1 = figure(plot_width = 650, plot_height = 400, x_range = df_world.country.values, y_range = Range1d(0, 1.1 * max(df_world.cases.values)), title = "Covid-19 Top Global Close Rates (At least 100 cases)")

v1.extra_y_ranges = {"Close Rate": Range1d(start = 0, end = 100)}



v11 = v1.vbar(x = dodge("country", 0.15, range = v1.x_range), top = "cases", width = 0.2, source = source_1, color = "blue", legend_label = "Cases")

v12 = v1.vbar(x = dodge("country", -0.15, range = v1.x_range), top = "completed_cases", width = 0.2, source = source_1, color = "green", legend_label = "Closed Cases")

v13 = v1.line("country", "close_rate", source = source_1, color = "orange", y_range_name = "Close Rate", legend_label = "Close Rate")



v1.add_tools(HoverTool(renderers = [v11], tooltips = tooltips_1))

v1.add_tools(HoverTool(renderers = [v12], tooltips = tooltips_2))

v1.add_tools(HoverTool(renderers = [v13], tooltips = tooltips_3))



v1.xaxis.major_label_orientation = pi / 4



v1.xaxis.axis_label = "Country"

v1.yaxis.axis_label = "Count"

v1.add_layout(LinearAxis(y_range_name = "Close Rate", axis_label = "Close Rate"), "right")



v1.legend.location = "top_right"



source_2 = ColumnDataSource(data = dict(

    country = df_asia.country.values,

    cases = df_asia.cases.values,

    completed_cases = df_asia.completed_cases.values,

    close_rate = df_asia.close_rate.values * 100

))



v2 = figure(plot_width = 650, plot_height = 400, x_range = df_asia.country.values, y_range = Range1d(0, 1.1 * max(df_asia.cases.values)), title = "Covid-19 Top Asian Close Rates (At least 100 cases)")

v2.extra_y_ranges = {"Close Rate": Range1d(start = 0, end = 100)}



v21 = v2.vbar(x = dodge("country", 0.15, range = v2.x_range), top = "cases", width = 0.2, source = source_2, color = "blue", legend_label = "Cases")

v22 = v2.vbar(x = dodge("country", -0.15, range = v2.x_range), top = "completed_cases", width = 0.2, source = source_2, color = "green", legend_label = "Closed Cases")

v23 = v2.line("country", "close_rate", source = source_2, color = "orange", y_range_name = "Close Rate", legend_label = "Close Rate")



v2.add_tools(HoverTool(renderers = [v21], tooltips = tooltips_1))

v2.add_tools(HoverTool(renderers = [v22], tooltips = tooltips_2))

v2.add_tools(HoverTool(renderers = [v23], tooltips = tooltips_3))



v2.xaxis.major_label_orientation = pi / 4



v2.xaxis.axis_label = "Country"

v2.yaxis.axis_label = "Count"

v2.add_layout(LinearAxis(y_range_name = "Close Rate", axis_label = "Close Rate"), "right")



v2.legend.location = "top_right"



source_3 = ColumnDataSource(data = dict(

    state = df_state[df_state.cases >= 100].state.values,

    cases = df_state[df_state.cases >= 100].cases.values,

    completed_cases = df_state[df_state.cases >= 100].completed_cases.values,

    close_rate = df_state[df_state.cases >= 100].close_rate.values * 100

))



v3 = figure(plot_width = 650, plot_height = 400, x_range = df_state[df_state.cases >= 100].state.values, y_range = Range1d(0, 1.1 * max(df_state[df_state.cases >= 100].cases.values)), title = "Covid-19 Top State Close Rates (At least 100 cases)")

v3.extra_y_ranges = {"Close Rate": Range1d(start = 0, end = 100)}



v31 = v3.vbar(x = dodge("state", 0.15, range = v3.x_range), top = "cases", width = 0.2, source = source_3, color = "blue", legend_label = "Cases")

v32 = v3.vbar(x = dodge("state", -0.15, range = v3.x_range), top = "completed_cases", width = 0.2, source = source_3, color = "green", legend_label = "Closed Cases")

v33 = v3.line("state", "close_rate", source = source_3, color = "orange", y_range_name = "Close Rate", legend_label = "Close Rate")



v3.add_tools(HoverTool(renderers = [v31], tooltips = tooltips_1))

v3.add_tools(HoverTool(renderers = [v32], tooltips = tooltips_2))

v3.add_tools(HoverTool(renderers = [v33], tooltips = tooltips_3))



v3.xaxis.major_label_orientation = pi / 4



v3.xaxis.axis_label = "State"

v3.yaxis.axis_label = "Count"

v3.add_layout(LinearAxis(y_range_name = "Close Rate", axis_label = "Close Rate"), "right")



v3.legend.location = "top_right"



show(column(v1, v2, v3))

coronavirus_trend = np.array([7,6,7,6,4,5,5,4,4,4,3,3,4,4,3,3,3,3,2,2,2,2,2,3,3,3,3,4,4,4,8,16,23,19,14,12,12,15,17,20,25,32,31,33,37,38,40,77,60,68,73,81,78,93,93,99,100,98,77,78,74,69,86,87,81,59,56,54,54,55,55,55,76,50,71,70,69,72,70,41,40])



df_covid = read_covid_data()



df = df_covid.groupby("date")["cases"].sum().reset_index()

df["lag_1_cases"] = df.cases.shift(1)

df["day_cases"] = df.cases - df.lag_1_cases

df = df[(df.date >= "2020-02-01") & (df.date <= "2020-04-21")]



df["cases_scaled"] = df.day_cases * 100 / max(df.day_cases)



df["coronavirus_google_trend"] = coronavirus_trend

source = ColumnDataSource(data = dict(

    date = np.array(df.date.values, dtype = np.datetime64),

    date_raw = df.date.values,

    cases_scaled = df.cases_scaled.values,

    coronavirus_google_trend = df.coronavirus_google_trend.values

))



tooltips_1 = [

    ("Date", "@date_raw"),

    ("Coronavirus Cases Scaled", "@cases_scaled{0}")

]



tooltips_2 = [

    ("Date", "@date_raw"),

    ("Coronavirus Google Trend", "@coronavirus_google_trend{0}")

]



v = figure(plot_width = 650, plot_height = 400, x_axis_type = "datetime", title = "Coronavirus search trend on Google")



v1 = v.line("date", "cases_scaled", source = source, color = "blue", legend_label = "Coronavirus Cases Scaled")

v2 = v.line("date", "coronavirus_google_trend", source = source, color = "green", legend_label = "Coronavirus Google Trend")



v.add_tools(HoverTool(renderers = [v1], tooltips = tooltips_1))

v.add_tools(HoverTool(renderers = [v2], tooltips = tooltips_2))



v.xaxis.major_label_orientation = pi / 4



v.xaxis.axis_label = "Date"

v.yaxis.axis_label = "Value"



v.legend.location = "top_left"



show(v)

df_covid = read_covid_data()



df = df_covid.groupby("date")["cases", "deaths", "recoveries"].sum().reset_index()

df = df[df.date >= "2020-03-01"]

source = ColumnDataSource(data = dict(

    date = np.array(df.date.values, dtype = np.datetime64),

    date_raw = df.date.values,

    cases = df.cases.values,

    deaths = df.deaths.values,

    recoveries = df.recoveries.values

))



tooltips_1 = [

    ("Date", "@date_raw"),

    ("Cases", "@cases")

]



tooltips_2 = [

    ("Date", "@date_raw"),

    ("Deaths", "@deaths")

]



tooltips_3 = [

    ("Date", "@date_raw"),

    ("Recoveries", "@recoveries")

]



v = figure(plot_width = 650, plot_height = 400, x_axis_type = "datetime", title = "Cumulative metric counts before and during lockdowns")



v1 = v.line("date", "cases", source = source, color = "blue", legend_label = "Cases")

v2 = v.line("date", "deaths", source = source, color = "brown", legend_label = "Deaths")

v3 = v.line("date", "recoveries", source = source, color = "green", legend_label = "Recoveries")



v.add_tools(HoverTool(renderers = [v1], tooltips = tooltips_1))

v.add_tools(HoverTool(renderers = [v2], tooltips = tooltips_2))

v.add_tools(HoverTool(renderers = [v3], tooltips = tooltips_3))



curfew = Span(location = 7.5, dimension = "height", line_color = "grey", line_dash = "dashed", line_width = 2)

v.add_layout(curfew)



lockdown_1_start_date = time.mktime(dt(2020, 3, 25, 0, 0, 0).timetuple()) * 1000

lockdown_2_start_date = time.mktime(dt(2020, 4, 15, 0, 0, 0).timetuple()) * 1000



lockdown_1 = BoxAnnotation(left = lockdown_1_start_date, right = lockdown_2_start_date, fill_alpha = 0.1, fill_color = "yellow")

lockdown_2 = BoxAnnotation(left = lockdown_2_start_date, fill_alpha = 0.1, fill_color = "orange")



v.add_layout(lockdown_1)

v.add_layout(lockdown_2)



v.xaxis.major_label_orientation = pi / 4



v.xaxis.axis_label = "Date"

v.yaxis.axis_label = "Count"



v.legend.location = "top_left"



show(v)

df_covid = read_covid_data()



df = df_covid.groupby("date")["cases", "deaths", "recoveries"].sum().reset_index()

df = df[(df.date >= "2020-03-15") & (df.date <= "2020-04-06")]

source = ColumnDataSource(data = dict(

    date = df.date.values,

    cases = df.cases.values,

    deaths = df.deaths.values,

    recoveries = df.recoveries.values

))



tooltips_1 = [

    ("Date", "@date"),

    ("Cases", "@cases")

]



tooltips_2 = [

    ("Date", "@date"),

    ("Deaths", "@deaths")

]



tooltips_3 = [

    ("Date", "@date"),

    ("Recoveries", "@recoveries")

]



v = figure(plot_width = 650, plot_height = 400, x_range = df.date.values, title = "Cumulative metric counts before and after curfew")



v1 = v.line("date", "cases", source = source, color = "blue", legend_label = "Cases")

v2 = v.line("date", "deaths", source = source, color = "orange", legend_label = "Deaths")

v3 = v.line("date", "recoveries", source = source, color = "green", legend_label = "Recoveries")



v.add_tools(HoverTool(renderers = [v1], tooltips = tooltips_1))

v.add_tools(HoverTool(renderers = [v2], tooltips = tooltips_2))

v.add_tools(HoverTool(renderers = [v3], tooltips = tooltips_3))



curfew = Span(location = 7.5, dimension = "height", line_color = "grey", line_dash = "dashed", line_width = 2)

v.add_layout(curfew)



v.xaxis.major_label_orientation = pi / 4



v.xaxis.axis_label = "Date"

v.yaxis.axis_label = "Count"



v.legend.location = "top_left"



show(v)

df_covid = read_covid_data()

df_global = read_global_data()



df_country = df_global.groupby(["country", "continent"])["deaths", "recoveries"].max().reset_index()

df_country["completed_cases"] = df_country.deaths + df_country.recoveries

df_country["mortality_rate"] = df_country.deaths / df_country.completed_cases

df_country.sort_values("mortality_rate", ascending = False, inplace = True)

df_world = pd.concat([df_country[df_country.completed_cases >= 100].head(10), df_country[df_country.country == "India"]])

df_asia = df_country[(df_country.completed_cases >= 100) & (df_country.continent == "Asia")].head(10)



df_state = df_covid.groupby("state")["deaths", "recoveries"].max().reset_index()

df_state["completed_cases"] = df_state.deaths + df_state.recoveries

df_state["mortality_rate"] = df_state.deaths / df_state.completed_cases

df_state.sort_values("mortality_rate", ascending = False, inplace = True)

source_1 = ColumnDataSource(data = dict(

    country = df_world.country.values,

    completed_cases = df_world.completed_cases.values,

    deaths = df_world.deaths.values,

    recoveries = df_world.recoveries.values,

    mortality_rate = df_world.mortality_rate.values * 100

))



tooltips_1 = [

    ("Recoveries", "@recoveries")

]



tooltips_2 = [

    ("Deaths", "@deaths")

]



tooltips_3 = [

    ("Mortality Rate", "@mortality_rate{0} %")

]



v1 = figure(plot_width = 650, plot_height = 400, x_range = df_world.country.values, y_range = Range1d(0, 1.1 * max(df_world.deaths.values)), title = "Covid-19 Top Global Mortality Rates (At least 100 completed cases)")

v1.extra_y_ranges = {"Mortality Rate": Range1d(start = 0, end = 100)}



v11 = v1.vbar(x = dodge("country", 0.15, range = v1.x_range), top = "recoveries", width = 0.2, source = source_1, color = "blue", legend_label = "Recoveries")

v12 = v1.vbar(x = dodge("country", -0.15, range = v1.x_range), top = "deaths", width = 0.2, source = source_1, color = "orange", legend_label = "Deaths")

v13 = v1.line("country", "mortality_rate", source = source_1, color = "red", y_range_name = "Mortality Rate", legend_label = "Mortality Rate")



v1.add_tools(HoverTool(renderers = [v11], tooltips = tooltips_1))

v1.add_tools(HoverTool(renderers = [v12], tooltips = tooltips_2))

v1.add_tools(HoverTool(renderers = [v13], tooltips = tooltips_3))



v1.xaxis.major_label_orientation = pi / 4



v1.xaxis.axis_label = "Country"

v1.yaxis.axis_label = "Count"

v1.add_layout(LinearAxis(y_range_name = "Mortality Rate", axis_label = "Mortality Rate"), "right")



v1.legend.location = "top_right"



source_2 = ColumnDataSource(data = dict(

    country = df_asia.country.values,

    completed_cases = df_asia.completed_cases.values,

    deaths = df_asia.deaths.values,

    recoveries = df_asia.recoveries.values,

    mortality_rate = df_asia.mortality_rate.values * 100

))



v2 = figure(plot_width = 650, plot_height = 400, x_range = df_asia.country.values, y_range = Range1d(0, 1.1 * max(df_asia.recoveries.values)), title = "Covid-19 Top Asian Mortality Rates (At least 100 completed cases)")

v2.extra_y_ranges = {"Mortality Rate": Range1d(start = 0, end = 100)}



v21 = v2.vbar(x = dodge("country", 0.15, range = v2.x_range), top = "recoveries", width = 0.2, source = source_2, color = "blue", legend_label = "Recoveries")

v22 = v2.vbar(x = dodge("country", -0.15, range = v2.x_range), top = "deaths", width = 0.2, source = source_2, color = "orange", legend_label = "Deaths")

v23 = v2.line("country", "mortality_rate", source = source_2, color = "red", y_range_name = "Mortality Rate", legend_label = "Mortality Rate")



v2.add_tools(HoverTool(renderers = [v21], tooltips = tooltips_1))

v2.add_tools(HoverTool(renderers = [v22], tooltips = tooltips_2))

v2.add_tools(HoverTool(renderers = [v23], tooltips = tooltips_3))



v2.xaxis.major_label_orientation = pi / 4



v2.xaxis.axis_label = "Country"

v2.yaxis.axis_label = "Count"

v2.add_layout(LinearAxis(y_range_name = "Mortality Rate", axis_label = "Mortality Rate"), "right")



v2.legend.location = "top_right"



source_3 = ColumnDataSource(data = dict(

    state = df_state[df_state.completed_cases >= 40].state.values,

    completed_cases = df_state[df_state.completed_cases >= 40].completed_cases.values,

    deaths = df_state[df_state.completed_cases >= 40].deaths.values,

    recoveries = df_state[df_state.completed_cases >= 40].recoveries.values,

    mortality_rate = df_state[df_state.completed_cases >= 40].mortality_rate.values * 100

))



v3 = figure(plot_width = 650, plot_height = 400, x_range = df_state[df_state.completed_cases >= 40].state.values, y_range = Range1d(0, 1.1 * max(df_state[df_state.completed_cases >= 40].recoveries.values)), title = "Covid-19 Top State Mortality Rates (At least 40 completed cases)")

v3.extra_y_ranges = {"Mortality Rate": Range1d(start = 0, end = 100)}



v31 = v3.vbar(x = dodge("state", 0.15, range = v3.x_range), top = "recoveries", width = 0.2, source = source_3, color = "blue", legend_label = "Recoveries")

v32 = v3.vbar(x = dodge("state", -0.15, range = v3.x_range), top = "deaths", width = 0.2, source = source_3, color = "orange", legend_label = "Deaths")

v33 = v3.line("state", "mortality_rate", source = source_3, color = "red", y_range_name = "Mortality Rate", legend_label = "Mortality Rate")



v3.add_tools(HoverTool(renderers = [v31], tooltips = tooltips_1))

v3.add_tools(HoverTool(renderers = [v32], tooltips = tooltips_2))

v3.add_tools(HoverTool(renderers = [v33], tooltips = tooltips_3))



v3.xaxis.major_label_orientation = pi / 4



v3.xaxis.axis_label = "State"

v3.yaxis.axis_label = "Count"

v3.add_layout(LinearAxis(y_range_name = "Mortality Rate", axis_label = "Mortality Rate"), "right")



v3.legend.location = "top_right"



show(column(v1, v2, v3))

df_census = read_census_data()

df_labs = read_test_labs_data()



df_state = df_labs.groupby("state")["lab"].count().reset_index().rename(columns = {"lab": "labs"}).merge(df_census, on = "state")

df_state["people_per_lab"] = df_state.population / df_state.labs

df_state["area_per_lab"] = df_state.area / df_state.labs



df_state_lab = pd.pivot_table(df_labs, values = "lab", index = "state", columns = "type", aggfunc = "count", fill_value = 0).reset_index()

df_state_lab["labs"] = df_state_lab.sum(axis = 1)

df_state_lab = df_state_lab.sort_values("labs", ascending = False).head(10)



df_city_lab = pd.pivot_table(df_labs, values = "lab", index = "city", columns = "type", aggfunc = "count", fill_value = 0).reset_index()

df_city_lab["labs"] = df_city_lab.sum(axis = 1)

df_city_lab = df_city_lab.sort_values("labs", ascending = False).head(10)

source_1 = {

    "state": df_state_lab.state.values,

    "Government Laboratory": df_state_lab["Government Laboratory"].values,

    "Private Laboratory": df_state_lab["Private Laboratory"].values,

    "Collection Site": df_state_lab["Collection Site"].values

}



types = ["Government Laboratory", "Private Laboratory", "Collection Site"]



v1 = figure(plot_width = 300, plot_height = 400, x_range = source_1["state"], title = "Top States with Testing Laboratories")

v1.vbar_stack(types, x = "state", width = 0.81, color = Spectral3, source = source_1, legend_label = types)

v1.xaxis.major_label_orientation = pi / 6

v1.legend.label_text_font_size = "5pt"



source_2 = {

    "city": df_city_lab.city.values,

    "Government Laboratory": df_city_lab["Government Laboratory"].values,

    "Private Laboratory": df_city_lab["Private Laboratory"].values,

    "Collection Site": df_city_lab["Collection Site"].values

}



v2 = figure(plot_width = 300, plot_height = 400, x_range = source_2["city"], title = "Top Cities with Testing Laboratories")

v2.vbar_stack(types, x = "city", width = 0.81, color = Spectral3, source = source_2, legend_label = types)

v2.xaxis.major_label_orientation = pi / 6

v2.legend.label_text_font_size = "5pt"



source_3 = ColumnDataSource(data = dict(

    state = df_state.state.values,

    labs = df_state.labs.values,

    people_per_lab = df_state.people_per_lab.values / 1000000,

    area_per_lab = df_state.area_per_lab.values / 1000

))



tooltips_3 = [

    ("State", "@state"),

    ("Labs", "@labs"),

    ("People per Lab", "@people_per_lab{0.00} M"),

    ("Area per Lab", "@area_per_lab{0.00} K")

]



h_mid = max(df_state.area_per_lab.values / 1000) / 2

v_mid = max(df_state.people_per_lab.values / 1000000) / 2



source_labels = ColumnDataSource(data = dict(

    state = df_state[(df_state.people_per_lab >= v_mid * 1000000) | (df_state.area_per_lab >= h_mid * 1000)].state.values,

    people_per_lab = df_state[(df_state.people_per_lab >= v_mid * 1000000) | (df_state.area_per_lab >= h_mid * 1000)].people_per_lab.values / 1000000,

    area_per_lab = df_state[(df_state.people_per_lab >= v_mid * 1000000) | (df_state.area_per_lab >= h_mid * 1000)].area_per_lab.values / 1000

))



labels = LabelSet(x = "people_per_lab", y = "area_per_lab", text = "state", source = source_labels, level = "glyph", x_offset = -19, y_offset = -23, render_mode = "canvas")



v3 = figure(plot_width = 600, plot_height = 600, tooltips = tooltips_3, title = "People and Area per Lab by State")

v3.circle("people_per_lab", "area_per_lab", source = source_3, size = 13, color = "blue", alpha = 0.41)



tl_box = BoxAnnotation(right = v_mid, bottom = h_mid, fill_alpha = 0.1, fill_color = "orange")

tr_box = BoxAnnotation(left = v_mid, bottom = h_mid, fill_alpha = 0.1, fill_color = "red")

bl_box = BoxAnnotation(right = v_mid, top = h_mid, fill_alpha = 0.1, fill_color = "green")

br_box = BoxAnnotation(left = v_mid, top = h_mid, fill_alpha = 0.1, fill_color = "orange")



v3.add_layout(tl_box)

v3.add_layout(tr_box)

v3.add_layout(bl_box)

v3.add_layout(br_box)



v3.add_layout(labels)



v3.xaxis.axis_label = "People per Lab (in Million)"

v3.yaxis.axis_label = "Area per Lab (in Thousand sq km)"



show(column(row(v1, v2), v3))

df_global = read_global_data()



country_list = ["Bangladesh", "Bhutan", "Burma", "China", "India", "Nepal", "Pakistan", "Sri Lanka"]



df = df_global[df_global.country.isin(country_list)]

df = df.groupby(["country", "date"])[["cases", "deaths", "recoveries"]].sum().reset_index()

v = figure(plot_width = 650, plot_height = 400, x_axis_type = "datetime", title = "Covid-19 log of cumulative cases by country")



tooltips = [

    ("Country", "@country"),

    ("Date", "@date{%F}"),

    ("Cases", "@cases")

]

    

formatters = {

    "@date": "datetime"

}



for i in range(len(country_list)):

    country = country_list[i]

    df_country = df[df.country == country]



    source = ColumnDataSource(data = dict(

        country = df_country.country.values,

        date = np.array(df_country.date.values, dtype = np.datetime64),

        cases = df_country.cases.values,

        log_cases = np.log10(df_country.cases.values)

    ))

    

    v.line("date", "log_cases", source = source, color = Spectral8[i], legend_label = country)



v.add_tools(HoverTool(tooltips = tooltips, formatters = formatters))



v.legend.location = "bottom_left"

v.legend.label_text_font_size = "8pt"



v.xaxis.axis_label = "Date"

v.yaxis.axis_label = "Log Cases"



show(v)

df_covid = read_covid_data()

df_census = read_census_data()

df_testing = read_test_samples_data()

df_labs = read_test_labs_data()



df_country = df_covid.groupby("date")["cases"].sum().reset_index().merge(df_testing[["date", "samples_tested"]], on = "date")

df_country["lag_1_cases"] = df_country.cases.shift(1)

df_country["day_cases"] = df_country.cases - df_country.lag_1_cases

df_country["lag_1_samples_tested"] = df_country.samples_tested.shift(1)

df_country["day_samples_tested"] = df_country.samples_tested - df_country.lag_1_samples_tested



df_country = df_country[df_country.date >= "2020-03-18"]

df_country.dropna(subset = ["day_cases", "day_samples_tested"], inplace = True)

df_country["case_rate"] = df_country.day_cases / df_country.day_samples_tested



df_state = df_labs.groupby("state")["lab"].count().reset_index().rename(columns = {"lab": "labs"}).merge(df_census, on = "state")

df_state["people_per_lab"] = df_state.population / df_state.labs

df_state["area_per_lab"] = df_state.area / df_state.labs

source = ColumnDataSource(data = dict(

    date = df_country.date.values,

    day_cases = df_country.day_cases.values,

    day_samples_tested = df_country.day_samples_tested.values,

    case_rate = df_country.case_rate.values

))



tooltips_1 = [

    ("Date", "@date"),

    ("Samples Tested", "@day_samples_tested")

]



tooltips_2 = [

    ("Date", "@date"),

    ("Cases", "@day_cases")

]



tooltips_3 = [

    ("Date", "@date"),

    ("Case Rate", "@case_rate{0.00}")

]



v = figure(plot_width = 650, plot_height = 400, x_range = df_country.date.values, title = "Covid-19 cases and testing from 19th March")

v.extra_y_ranges = {"Case Rate": Range1d(start = 0.0, end = 0.1)}



v1 = v.vbar(x = dodge("date", 0.25, range = v.x_range), top = "day_samples_tested", width = 0.2, source = source, color = "blue", legend_label = "Samples Tested")

v2 = v.vbar(x = dodge("date", -0.25, range = v.x_range), top = "day_cases", width = 0.2, source = source, color = "orange", legend_label = "Cases")

v3 = v.line("date", "case_rate", source = source, color = "red", y_range_name = "Case Rate", legend_label = "Case Rate")



v.add_tools(HoverTool(renderers = [v1], tooltips = tooltips_1))

v.add_tools(HoverTool(renderers = [v2], tooltips = tooltips_2))

v.add_tools(HoverTool(renderers = [v3], tooltips = tooltips_3))



v.xaxis.major_label_orientation = pi/4



v.xaxis.axis_label = "Date"

v.yaxis.axis_label = "Count"

v.add_layout(LinearAxis(y_range_name = "Case Rate", axis_label = "Case Rate"), "right")



v.legend.location = "top_left"



show(v)

df_covid = read_covid_data()



df = df_covid.copy()

df["log_cases"] = np.log10(df.cases)

df["lag_1_cases"] = df.groupby("state")["cases"].shift(1)

df["day_cases"] = df.cases - df.lag_1_cases

df["lag_1_day_cases"] = df.groupby("state")["day_cases"].shift(1)

df["lag_2_day_cases"] = df.groupby("state")["day_cases"].shift(2)

df["lag_3_day_cases"] = df.groupby("state")["day_cases"].shift(3)

df["lag_4_day_cases"] = df.groupby("state")["day_cases"].shift(4)

df["lag_5_day_cases"] = df.groupby("state")["day_cases"].shift(5)

df["lag_6_day_cases"] = df.groupby("state")["day_cases"].shift(6)

df["ma_7d_day_cases"] = df[["day_cases", "lag_1_day_cases", "lag_2_day_cases", "lag_3_day_cases",

                              "lag_4_day_cases", "lag_5_day_cases", "lag_6_day_cases"]].mean(axis = 1).values

df["log_ma_7d_day_cases"] = np.log10(df.ma_7d_day_cases)

df = df[df.state != "Unassigned"]

df.date = pd.to_datetime(df.date, format = "%Y-%m-%d")

df = df[df.date >= pd.datetime(2020, 3, 21)]

v1 = figure(plot_width = 333, plot_height = 333, x_axis_type = "datetime", title = "Covid-19 cases from 21st March")

v2 = figure(plot_width = 333, plot_height = 333, x_axis_type = "datetime", title = "Covid-19 cases from 21st March")



tooltips = [

    ("State", "@state"),

    ("Date", "@date{%F}"),

    ("Cases", "@cases")

]

    

formatters = {

    "@date": "datetime"

}

    

for i in range(len(df.state.unique())):

    state = df.state.unique()[i]

    df_state = df[df.state == state]

    

    source = ColumnDataSource(data = dict(

        state = df_state.state.values,

        date = np.array(df_state.date.values, dtype = np.datetime64),

        cases = df_state.cases.values,

        log_cases = df_state.log_cases.values

    ))

    

    v1.line("date", "cases", source = source, color = Category20[20][i % 20])

    v2.line("date", "log_cases", source = source, color = Category20[20][i % 20])



v1.add_tools(HoverTool(tooltips = tooltips, formatters = formatters))

v2.add_tools(HoverTool(tooltips = tooltips, formatters = formatters))



v1.xaxis.axis_label = "Date"

v1.yaxis.axis_label = "Cases"



v2.xaxis.axis_label = "Date"

v2.yaxis.axis_label = "Cases (Log Scale)"



show(row(v1, v2))

state_list = ["Delhi", "Kerala", "Maharashtra", "Tamil Nadu"]



v1 = figure(plot_width = 333, plot_height = 333, x_axis_type = "datetime", title = "Covid-19 cumulative cases since 21st March")

v2 = figure(plot_width = 333, plot_height = 333, x_axis_type = "datetime", title = "Covid-19 MA(7) day cases since 21st March")



tooltips_1 = [

    ("State", "@state"),

    ("Date", "@date{%F}"),

    ("Cases", "@cases")

]

    

tooltips_2 = [

    ("State", "@state"),

    ("Date", "@date{%F}"),

    ("Day Cases", "@ma_7d_day_cases{0}")

]



formatters = {

    "@date": "datetime"

}



for i in range(len(state_list)):

    state = state_list[i]

    df_state = df[df.state == state]



    source = ColumnDataSource(data = dict(

        state = df_state.state.values,

        date = np.array(df_state.date.values, dtype = np.datetime64),

        cases = df_state.cases.values,

        log_cases = df_state.log_cases.values,

        ma_7d_day_cases = df_state.ma_7d_day_cases.values,

        log_ma_7d_day_cases = df_state.log_ma_7d_day_cases.values

    ))

    

    v1.line("date", "log_cases", source = source, color = Spectral4[i], legend_label = state)

    v2.line("date", "log_ma_7d_day_cases", source = source, color = Spectral4[i], legend_label = state)



v1.add_tools(HoverTool(tooltips = tooltips_1, formatters = formatters))

v2.add_tools(HoverTool(tooltips = tooltips_2, formatters = formatters))



v1.legend.location = "top_left"

v2.legend.location = "top_left"



v1.legend.label_text_font_size = "5pt"

v2.legend.label_text_font_size = "5pt"



v1.xaxis.axis_label = "Date"

v1.yaxis.axis_label = "Cases (Log Scale)"



v2.xaxis.axis_label = "Date"

v2.yaxis.axis_label = "MA(7) Day Cases (Log Scale)"



show(row(v1, v2))

df_covid = read_covid_data()

df_census = read_census_data()



df = df_covid.groupby("state")["cases", "deaths", "recoveries"].max().reset_index().merge(df_census[["state", "gender_ratio"]], on = "state")

df["death_rate"] = df.deaths / (df.deaths + df.recoveries)

df["rank_gender_ratio"] = df["gender_ratio"].rank(ascending = False)

df["rank_cases"] = df["cases"].rank(ascending = False)

df["rank_deaths"] = df["deaths"].rank(ascending = False)

df["rank_death_rate"] = df["death_rate"].rank(ascending = False)

source = ColumnDataSource(data = dict(

    state = df.state.values,

    gender_ratio = df.gender_ratio.values,

    cases = df.cases.values,

    deaths = df.deaths.values,

    death_rate = df.death_rate.values

))



tooltips_1 = [

    ("state", "@state"),

    ("gender_ratio", "@gender_ratio"),

    ("cases", "@cases")

]



tooltips_2 = [

    ("state", "@state"),

    ("gender_ratio", "@gender_ratio"),

    ("deaths", "@deaths")

]



tooltips_3 = [

    ("state", "@state"),

    ("gender_ratio", "@gender_ratio"),

    ("death_rate", "@death_rate{0.000}")

]



v1 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_1, title = "Gender Ratio vs Cases by State")

v1.circle("gender_ratio", "cases", source = source, size = 13, color = "blue", alpha = 0.41)

v1.xaxis.axis_label = "Gender Ratio"

v1.yaxis.axis_label = "Cases"



v2 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_2, title = "Gender Ratio vs Deaths by State")

v2.circle("gender_ratio", "deaths", source = source, size = 13, color = "red", alpha = 0.41)

v2.xaxis.axis_label = "Gender Ratio"

v2.yaxis.axis_label = "Deaths"



v3 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_3, title = "Gender Ratio vs Death-Rate by State")

v3.circle("gender_ratio", "death_rate", source = source, size = 13, color = "orange", alpha = 0.41)

v3.xaxis.axis_label = "Gender Ratio"

v3.yaxis.axis_label = "Death Rate"



show(row(v1, v2, v3))

source = ColumnDataSource(data = dict(

    state = df.state.values,

    rank_gender_ratio = df.rank_gender_ratio.values,

    rank_cases = df.rank_cases.values,

    rank_deaths = df.rank_deaths.values,

    rank_death_rate = df.rank_death_rate.values

))



tooltips_1 = [

    ("State", "@state"),

    ("Rank of Gender Ratio", "@rank_gender_ratio{0}"),

    ("Rank of Cases", "@rank_cases{0}")

]



tooltips_2 = [

    ("State", "@state"),

    ("Rank of Gender Ratio", "@rank_gender_ratio{0}"),

    ("Rank of Cases", "@rank_deaths{0}")

]



tooltips_3 = [

    ("State", "@state"),

    ("Rank of Gender Ratio", "@rank_gender_ratio{0}"),

    ("Rank of Death Rate", "@rank_death_rate{0}")

]



v1 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_1, title = "Ranks of Gender Ratio vs Cases by State")

v1.circle("rank_gender_ratio", "rank_cases", source = source, size = 13, color = "blue", alpha = 0.41)

v1.x_range.flipped = True

v1.y_range.flipped = True

v1.xaxis.axis_label = "Rank of Gender Ratio"

v1.yaxis.axis_label = "Rank of Cases"



v2 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_2, title = "Ranks of Gender Ratio vs Deaths by State")

v2.circle("rank_gender_ratio", "rank_deaths", source = source, size = 13, color = "red", alpha = 0.41)

v2.x_range.flipped = True

v2.y_range.flipped = True

v2.xaxis.axis_label = "Rank of Gender Ratio"

v2.yaxis.axis_label = "Rank of Deaths"



v3 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_3, title = "Ranks of Gender Ratio vs Death-Rate by State")

v3.circle("rank_gender_ratio", "rank_death_rate", source = source, size = 13, color = "orange", alpha = 0.41)

v3.x_range.flipped = True

v3.y_range.flipped = True

v3.xaxis.axis_label = "Rank of Gender Ratio"

v3.yaxis.axis_label = "Rank of Death Rate"



show(row(v1, v2, v3))

df_covid = read_covid_data()



df = df_covid.copy()

df = df[df.state != "Unassigned"]

df.date = pd.to_datetime(df.date, format = "%Y-%m-%d")

tab_list = []



for state in sorted(df.state.unique()):

    df_state = df[df.state == state]

    

    source = ColumnDataSource(data = dict(

        date = np.array(df_state.date.values, dtype = np.datetime64),

        cases = df_state.cases.values,

        deaths = df_state.deaths.values,

        recoveries = df_state.recoveries.values

    ))

    

    tooltips_1 = [

        ("Date", "@date{%F}"),

        ("Cases", "@cases")

    ]

    

    tooltips_2 = [

        ("Deaths", "@deaths")

    ]

    

    tooltips_3 = [

        ("Recoveries", "@recoveries")

    ]

    

    formatters = {

        "@date": "datetime"

    }

    

    v = figure(plot_width = 650, plot_height = 400, x_axis_type = "datetime", title = "Covid-19 metrics over time")

    v1 = v.line("date", "cases", source = source, color = "blue", legend_label = "Cases")

    v2 = v.line("date", "deaths", source = source, color = "red", legend_label = "Deaths")

    v3 = v.line("date", "recoveries", source = source, color = "green", legend_label = "Recoveries")

    v.legend.location = "top_left"

    v.add_tools(HoverTool(renderers = [v3], tooltips = tooltips_3, formatters = formatters, mode = "vline"))

    v.add_tools(HoverTool(renderers = [v2], tooltips = tooltips_2, formatters = formatters, mode = "vline"))

    v.add_tools(HoverTool(renderers = [v1], tooltips = tooltips_1, formatters = formatters, mode = "vline"))

    v.xaxis.axis_label = "Date"

    v.yaxis.axis_label = "Count"

    tab = Panel(child = v, title = state)

    tab_list.append(tab)



tabs = Tabs(tabs = tab_list)

show(tabs)

df_covid = read_covid_data()

df_census = read_census_data()



df_census["urbanization"] = df_census.urban_population / df_census.population



df = df_covid.groupby("state")["cases", "deaths", "recoveries"].max().reset_index().merge(df_census[["state", "urbanization"]], on = "state")

df["death_rate"] = df.deaths / (df.deaths + df.recoveries)

df["rank_urbanization"] = df["urbanization"].rank(ascending = False)

df["rank_cases"] = df["cases"].rank(ascending = False)

df["rank_deaths"] = df["deaths"].rank(ascending = False)

df["rank_death_rate"] = df["death_rate"].rank(ascending = False)

source = ColumnDataSource(data = dict(

    state = df.state.values,

    urbanization = df.urbanization.values,

    cases = df.cases.values,

    deaths = df.deaths.values,

    death_rate = df.death_rate.values

))



tooltips_1 = [

    ("State", "@state"),

    ("Urbanization", "@urbanization{0.00}"),

    ("Cases", "@cases")

]



tooltips_2 = [

    ("State", "@state"),

    ("Urbanization", "@urbanization{0.00}"),

    ("Deaths", "@deaths")

]



tooltips_3 = [

    ("State", "@state"),

    ("Urbanization", "@urbanization{0.00}"),

    ("Death Rate", "@death_rate{0.000}")

]



v1 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_1, title = "Urbanization vs Cases by State")

v1.circle("urbanization", "cases", source = source, size = 13, color = "blue", alpha = 0.41)

v1.xaxis.axis_label = "Urbanization"

v1.yaxis.axis_label = "Cases"



v2 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_2, title = "Urbanization vs Deaths by State")

v2.circle("urbanization", "deaths", source = source, size = 13, color = "red", alpha = 0.41)

v2.xaxis.axis_label = "Urbanization"

v2.yaxis.axis_label = "Deaths"



v3 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_3, title = "Urbanization vs Death-Rate by State")

v3.circle("urbanization", "death_rate", source = source, size = 13, color = "orange", alpha = 0.41)

v3.xaxis.axis_label = "Urbanization"

v3.yaxis.axis_label = "Death Rate"



show(row(v1, v2, v3))

source = ColumnDataSource(data = dict(

    state = df.state.values,

    rank_urbanization = df.rank_urbanization.values,

    rank_cases = df.rank_cases.values,

    rank_deaths = df.rank_deaths.values,

    rank_death_rate = df.rank_death_rate.values

))



tooltips_1 = [

    ("State", "@state"),

    ("Rank of Urbanization", "@rank_urbanization{0}"),

    ("Rank of Cases", "@rank_cases{0}")

]



tooltips_2 = [

    ("State", "@state"),

    ("Rank of Urbanization", "@rank_urbanization{0}"),

    ("Rank of Deaths", "@rank_deaths{0}")

]



tooltips_3 = [

    ("State", "@state"),

    ("Rank of Urbanization", "@rank_urbanization{0}"),

    ("Rank of Death Rate", "@rank_death_rate{0}")

]



v1 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_1, title = "Ranks of Urbanization vs Cases by State")

v1.circle("rank_urbanization", "rank_cases", source = source, size = 13, color = "blue", alpha = 0.41)

v1.x_range.flipped = True

v1.y_range.flipped = True

v1.xaxis.axis_label = "Rank of Urbanization"

v1.yaxis.axis_label = "Rank of Cases"



v2 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_2, title = "Ranks of Urbanization vs Deaths by State")

v2.circle("rank_urbanization", "rank_deaths", source = source, size = 13, color = "red", alpha = 0.41)

v2.x_range.flipped = True

v2.y_range.flipped = True

v2.xaxis.axis_label = "Rank of Urbanization"

v2.yaxis.axis_label = "Rank of Deaths"



v3 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_3, title = "Ranks of Urbanization vs Death-Rate by State")

v3.circle("rank_urbanization", "rank_death_rate", source = source, size = 13, color = "orange", alpha = 0.41)

v3.x_range.flipped = True

v3.y_range.flipped = True

v3.xaxis.axis_label = "Rank of Urbanization"

v3.yaxis.axis_label = "Rank of Death Rate"



show(row(v1, v2, v3))

df_covid = read_covid_data()

df_census = read_census_data()



df_census["density"] = df_census.population / df_census.area



df = df_covid.groupby("state")["cases", "deaths", "recoveries"].max().reset_index().merge(df_census[["state", "density"]], on = "state")

df["death_rate"] = df.deaths / (df.deaths + df.recoveries)

df["rank_density"] = df["density"].rank(ascending = False)

df["rank_cases"] = df["cases"].rank(ascending = False)

df["rank_deaths"] = df["deaths"].rank(ascending = False)

df["rank_death_rate"] = df["death_rate"].rank(ascending = False)

source = ColumnDataSource(data = dict(

    state = df.state.values,

    density = df.density.values,

    cases = df.cases.values,

    deaths = df.deaths.values,

    death_rate = df.death_rate.values

))



tooltips_1 = [

    ("State", "@state"),

    ("Density", "@density{0}"),

    ("Cases", "@cases")

]



tooltips_2 = [

    ("State", "@state"),

    ("Density", "@density{0}"),

    ("Deaths", "@deaths")

]



tooltips_3 = [

    ("State", "@state"),

    ("Density", "@density{0}"),

    ("Death Rate", "@death_rate{0.000}")

]



v1 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_1, title = "Density vs Cases by State")

v1.circle("density", "cases", source = source, size = 13, color = "blue", alpha = 0.41)

v1.xaxis.axis_label = "Density"

v1.yaxis.axis_label = "Cases"



v2 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_2, title = "Density vs Deaths by State")

v2.circle("density", "deaths", source = source, size = 13, color = "red", alpha = 0.41)

v2.xaxis.axis_label = "Density"

v2.yaxis.axis_label = "Deaths"



v3 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_3, title = "Density vs Death-Rate by State")

v3.circle("density", "death_rate", source = source, size = 13, color = "orange", alpha = 0.41)

v3.xaxis.axis_label = "Density"

v3.yaxis.axis_label = "Death Rate"



show(row(v1, v2, v3))

source = ColumnDataSource(data = dict(

    state = df.state.values,

    rank_density = df.rank_density.values,

    rank_cases = df.rank_cases.values,

    rank_deaths = df.rank_deaths.values,

    rank_death_rate = df.rank_death_rate.values

))



tooltips_1 = [

    ("State", "@state"),

    ("Rank of Density", "@rank_density{0}"),

    ("Rank of Cases", "@rank_cases{0}")

]



tooltips_2 = [

    ("State", "@state"),

    ("Rank of Density", "@rank_density{0}"),

    ("Rank of Deaths", "@rank_deaths{0}")

]



tooltips_3 = [

    ("State", "@state"),

    ("Rank of Density", "@rank_density{0}"),

    ("Rank of Death Rate", "@rank_death_rate{0}")

]



v1 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_1, title = "Ranks of Density vs Cases by State")

v1.circle("rank_density", "rank_cases", source = source, size = 13, color = "blue", alpha = 0.41)

v1.x_range.flipped = True

v1.y_range.flipped = True

v1.xaxis.axis_label = "Rank of Density"

v1.yaxis.axis_label = "Rank of Cases"



v2 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_2, title = "Ranks of Density vs Deaths by State")

v2.circle("rank_density", "rank_deaths", source = source, size = 13, color = "red", alpha = 0.41)

v2.x_range.flipped = True

v2.y_range.flipped = True

v2.xaxis.axis_label = "Rank of Density"

v2.yaxis.axis_label = "Rank of Deaths"



v3 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_3, title = "Ranks of Density vs Death-Rate by State")

v3.circle("rank_density", "rank_death_rate", source = source, size = 13, color = "orange", alpha = 0.41)

v3.x_range.flipped = True

v3.y_range.flipped = True

v3.xaxis.axis_label = "Rank of Density"

v3.yaxis.axis_label = "Rank of Death Rate"



show(row(v1, v2, v3))

df_covid = read_covid_data()

df_census = read_census_data()



df = df_covid.groupby("state")["cases", "deaths", "recoveries"].max().reset_index().merge(df_census[["state", "population"]], on = "state")

df["death_rate"] = df.deaths / (df.deaths + df.recoveries)

df["rank_population"] = df.population.rank(ascending = False)

df["rank_cases"] = df.cases.rank(ascending = False)

df["rank_deaths"] = df.deaths.rank(ascending = False)

df["rank_death_rate"] = df.death_rate.rank(ascending = False)

source = ColumnDataSource(data = dict(

    state = df.state.values,

    population = df.population.values / 1000000,

    cases = df.cases.values,

    deaths = df.deaths.values,

    death_rate = df.death_rate.values

))



tooltips_1 = [

    ("State", "@state"),

    ("Population", "@population{0.00} M"),

    ("Cases", "@cases")

]



tooltips_2 = [

    ("State", "@state"),

    ("Population", "@population{0.00} M"),

    ("Deaths", "@deaths")

]



tooltips_3 = [

    ("State", "@state"),

    ("Population", "@population{0.00} M"),

    ("Death Rate", "@death_rate{0.000}")

]



v1 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_1, title = "Population vs Cases by State")

v1.circle("population", "cases", source = source, size = 13, color = "blue", alpha = 0.41)

v1.xaxis.axis_label = "Population"

v1.yaxis.axis_label = "Cases"



v2 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_2, title = "Population vs Deaths by State")

v2.circle("population", "deaths", source = source, size = 13, color = "red", alpha = 0.41)

v2.xaxis.axis_label = "Population"

v2.yaxis.axis_label = "Deaths"



v3 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_3, title = "Population vs Death-Rate by State")

v3.circle("population", "death_rate", source = source, size = 13, color = "orange", alpha = 0.41)

v3.xaxis.axis_label = "Population"

v3.yaxis.axis_label = "Death Rate"



show(row(v1, v2, v3))

source = ColumnDataSource(data = dict(

    state = df.state.values,

    rank_population = df.rank_population.values,

    rank_cases = df.rank_cases.values,

    rank_deaths = df.rank_deaths.values,

    rank_death_rate = df.rank_death_rate.values

))



tooltips_1 = [

    ("State", "@state"),

    ("Rank of Population", "@rank_population{0}"),

    ("Rank of Cases", "@rank_cases{0}")

]



tooltips_2 = [

    ("State", "@state"),

    ("Rank of Population", "@rank_population{0}"),

    ("Rank of Deaths", "@rank_deaths{0}")

]



tooltips_3 = [

    ("State", "@state"),

    ("Rank of Population", "@rank_population{0}"),

    ("Rank of Death Rate", "@rank_death_rate{0}")

]



v1 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_1, title = "Ranks of Population vs Cases by State")

v1.circle("rank_population", "rank_cases", source = source, size = 13, color = "blue", alpha = 0.41)

v1.x_range.flipped = True

v1.y_range.flipped = True

v1.xaxis.axis_label = "Rank of Density"

v1.yaxis.axis_label = "Rank of Cases"



v2 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_2, title = "Ranks of Population vs Deaths by State")

v2.circle("rank_population", "rank_deaths", source = source, size = 13, color = "red", alpha = 0.41)

v2.x_range.flipped = True

v2.y_range.flipped = True

v2.xaxis.axis_label = "Rank of Density"

v2.yaxis.axis_label = "Rank of Deaths"



v3 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_3, title = "Ranks of Population vs Death-Rate by State")

v3.circle("rank_population", "rank_death_rate", source = source, size = 13, color = "orange", alpha = 0.41)

v3.x_range.flipped = True

v3.y_range.flipped = True

v3.xaxis.axis_label = "Rank of Density"

v3.yaxis.axis_label = "Rank of Death Rate"



show(row(v1, v2, v3))
