## importing packages

import numpy as np

import pandas as pd



from bokeh.layouts import column, row

from bokeh.models import Panel, Tabs, LinearAxis, Range1d, BoxAnnotation, LabelSet

from bokeh.models.tools import HoverTool

from bokeh.palettes import Category20, Spectral4

from bokeh.plotting import ColumnDataSource, figure, output_notebook, show

from bokeh.transform import dodge



from math import pi



output_notebook()

## defining constants

PATH_COVID = "/kaggle/input/covid19-in-india/covid_19_india.csv"

PATH_CENSUS = "/kaggle/input/covid19-in-india/population_india_census2011.csv"

PATH_TESTS = "/kaggle/input/covid19-in-india/ICMRTestingDetails.csv"

PATH_LABS = "/kaggle/input/covid19-in-india/ICMRTestingLabs.csv"



def read_covid_data():

    """

    Reads the main covid-19 data and preprocesses it.

    """

    

    df = pd.read_csv(PATH_COVID)

    df.rename(columns = {"State/UnionTerritory": "state",

                         "Confirmed": "cases",

                         "Deaths": "deaths",

                         "Cured": "recoveries"},

              inplace = True)

    df["date"] = pd.to_datetime(df.Date, format = "%d/%m/%y").dt.date.astype(str)



    return df



def read_census_data():

    """

    Reads the 2011 Indian census data and preprocesses it.

    """

    

    df = pd.read_csv(PATH_CENSUS)

    df.rename(columns = {"State / Union Territory": "state",

                         "Population": "population",

                         "Urban population": "urban_population",

                         "Gender Ratio": "gender_ratio"},

              inplace = True)



    df["area"] = df.Area.str.replace(",", "").str.split("km").str[0].astype(int)



    return df



def read_test_samples_data():

    """

    Reads the ICMR test samples data and preprocesses it.

    """

    

    df = pd.read_csv(PATH_TESTS)

    df.drop(index = 0, inplace = True)

    df.rename(columns = {"TotalSamplesTested": "samples_tested"},

              inplace = True)

    df["date"] = pd.to_datetime(df.DateTime, format = "%d/%m/%y %H:%S").dt.date.astype(str)

    

    return df



def read_test_labs_data():

    """

    Reads the ICMR testing labs data and preprocesses it.

    """

    

    df = pd.read_csv(PATH_LABS, encoding = "ISO-8859-1")

    

    return df

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



v = figure(plot_width = 650, plot_height = 400, x_range = df_country.date.values, title = "Covid-19 cases and test from 19th March")

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

h_mid = max(df_state.area_per_lab.values / 1000) / 2

v_mid = max(df_state.people_per_lab.values / 1000000) / 2



source = ColumnDataSource(data = dict(

    state = df_state.state.values,

    labs = df_state.labs.values,

    people_per_lab = df_state.people_per_lab.values / 1000000,

    area_per_lab = df_state.area_per_lab.values / 1000

))



source_labels = ColumnDataSource(data = dict(

    state = df_state[(df_state.people_per_lab >= v_mid * 1000000) | (df_state.area_per_lab >= h_mid * 1000)].state.values,

    people_per_lab = df_state[(df_state.people_per_lab >= v_mid * 1000000) | (df_state.area_per_lab >= h_mid * 1000)].people_per_lab.values / 1000000,

    area_per_lab = df_state[(df_state.people_per_lab >= v_mid * 1000000) | (df_state.area_per_lab >= h_mid * 1000)].area_per_lab.values / 1000

))



tooltips = [

    ("State", "@state"),

    ("Labs", "@labs"),

    ("People per Lab", "@people_per_lab{0.00} M"),

    ("Area per Lab", "@area_per_lab{0.00} K")

]



labels = LabelSet(x = "people_per_lab", y = "area_per_lab", text = "state", source = source_labels, level = "glyph", x_offset = -19, y_offset = -23, render_mode = "canvas")



v = figure(plot_width = 500, plot_height = 500, tooltips = tooltips, title = "People and Area per Lab by State")

v.circle("people_per_lab", "area_per_lab", source = source, size = 13, color = "blue", alpha = 0.41)



tl_box = BoxAnnotation(right = v_mid, bottom = h_mid, fill_alpha = 0.1, fill_color = "orange")

tr_box = BoxAnnotation(left = v_mid, bottom = h_mid, fill_alpha = 0.1, fill_color = "red")

bl_box = BoxAnnotation(right = v_mid, top = h_mid, fill_alpha = 0.1, fill_color = "green")

br_box = BoxAnnotation(left = v_mid, top = h_mid, fill_alpha = 0.1, fill_color = "orange")



v.add_layout(tl_box)

v.add_layout(tr_box)

v.add_layout(bl_box)

v.add_layout(br_box)



v.add_layout(labels)



v.xaxis.axis_label = "People per Lab (in Million)"

v.yaxis.axis_label = "Area per Lab (in Thousand sq km)"



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

    ("Day Cases", "@ma_7d_day_cases")

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



v1.legend.location = "bottom_right"

v2.legend.location = "bottom_right"



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
