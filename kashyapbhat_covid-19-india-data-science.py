import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objects as go

init_notebook_mode(connected=True)



import matplotlib.style as style

style.available



style.use('seaborn-poster')

style.use('ggplot')
age_group_dataset = pd.read_csv("../input/covid19-in-india/AgeGroupDetails.csv")

covid19_dataset = pd.read_csv("../input/covid19-in-india/covid_19_india.csv")

hospital_beds_dataset = pd.read_csv("../input/covid19-in-india/HospitalBedsIndia.csv")

icmr_testing_details_dataset = pd.read_csv("../input/covid19-in-india/ICMRTestingDetails.csv")

icmr_testing_labs_dataset = pd.read_csv("../input/covid19-in-india/ICMRTestingLabs.csv")

individual_details_dataset = pd.read_csv("../input/covid19-in-india/IndividualDetails.csv")

population_dataset = pd.read_csv("../input/covid19-in-india/population_india_census2011.csv")

statewise_testing_dataset = pd.read_csv("../input/covid19-in-india/StatewiseTestingDetails.csv")
def addStyleToPlots(ax, title, x_label, ylabel):

    ax.set_title(title, fontsize=24)

    ttl = ax.title

    ttl.set_position([.5, 1.05])

    ax.set_xlabel(x_label, fontsize=16)

    ax.set_ylabel(ylabel, fontsize=16)

    ax.tick_params(labelsize=11)
plt.figure(figsize=(15,8))



city_counts = individual_details_dataset.detected_city.value_counts()[individual_details_dataset.detected_city.value_counts().values > individual_details_dataset.detected_city.value_counts().values.mean()+individual_details_dataset.detected_city.value_counts().values.mean()]



ax = sns.barplot(city_counts.index, city_counts.values, palette="rocket");

ax.set_xticklabels(ax.get_xticklabels(), rotation=80);

addStyleToPlots(ax, 'Confirmed Covid-19 cases with respect to Cities', 'Cities', 'Number of cases')

plt.plot();
plt.figure(figsize=(15,8))



district_counts = individual_details_dataset.detected_district.value_counts()[individual_details_dataset.detected_district.value_counts().values > individual_details_dataset.detected_district.value_counts().values.mean()+individual_details_dataset.detected_district.value_counts().values.mean()]



ax = sns.barplot(district_counts.index, district_counts.values, palette="rocket");

ax.set_xticklabels(ax.get_xticklabels(), rotation=80);

addStyleToPlots(ax, 'Confirmed Covid-19 cases with respect to Districts', 'Districts', 'Number of cases')



plt.plot();
covid19_dataset = covid19_dataset.sort_values(by="State/UnionTerritory")

aggregation_functions = {'Confirmed': 'max','Cured':'max','Deaths':'max'}

state_wise = covid19_dataset.groupby('State/UnionTerritory', as_index=False).aggregate(aggregation_functions)
plt.figure(figsize=(15,8))

state_confirmed_cases = state_wise.sort_values('Confirmed',ascending=False)

ax = sns.barplot(state_confirmed_cases['State/UnionTerritory'], state_confirmed_cases['Confirmed'], palette="rocket")

ax.set_xticklabels(ax.get_xticklabels(), rotation=80)

addStyleToPlots(ax, 'Confirmed Covid-19 cases', 'States', 'Number of cases')

plt.plot();
plt.figure(figsize=(15,8))

state_cured_cases = state_wise.sort_values('Cured',ascending=False)

ax = sns.barplot(state_cured_cases['State/UnionTerritory'], state_cured_cases['Cured'], palette="rocket")

ax.set_xticklabels(ax.get_xticklabels(), rotation=80)

addStyleToPlots(ax, 'Recovered Covid-19 cases', 'States', 'Number of cases')

plt.plot();
plt.figure(figsize=(15,8))

state_death_cases = state_wise.sort_values('Deaths', ascending=False)

ax = sns.barplot(state_death_cases['State/UnionTerritory'], state_death_cases['Deaths'], palette="rocket")

ax.set_xticklabels(ax.get_xticklabels(), rotation=80)

addStyleToPlots(ax, 'Deaths due to Covid-19', 'States', 'Number of cases')

plt.plot();
def visualizeCasesWRTStates(state_name):

    state_data = statewise_testing_dataset[statewise_testing_dataset.State == state_name]



    plt.figure(figsize=(15,8))

    ax = sns.lineplot(state_data.Date, state_data.Positive)

    plt.xticks(rotation=80)

    title = "Covid-19 cases in " + str(state_name) + " with respect to time"

    addStyleToPlots(ax, title, "Dates", "Number of cases")

    plt.plot();


# print(statewise_testing_dataset.State.drop_duplicates().values, "\n\nInput the state name that you want to visualize. ")

# state_name = input()

visualizeCasesWRTStates("Karnataka")

visualizeCasesWRTStates("Kerala")
covid19_dataset["Date"] = pd.to_datetime(covid19_dataset['Date'])

covid19_dataset = covid19_dataset.sort_values(by="Date")

covid19_dataset = covid19_dataset[covid19_dataset['Date'] < pd.Timestamp("today").strftime("%d/%m/%Y")]

covid19_dataset.head(25)

aggregation_functions = {'Confirmed': 'sum'}

confirmed_cases_with_dates = covid19_dataset.groupby(covid19_dataset['Date']).aggregate(aggregation_functions)



import plotly.express as px



fig = go.Figure(data=[go.Scatter(x=confirmed_cases_with_dates.index, y=confirmed_cases_with_dates['Confirmed'])])



fig.update_xaxes(

    rangeslider_visible=True,

    rangeselector=dict(

        buttons=list([

            dict(count=1, label="1m", step="month", stepmode="backward"),

            dict(count=6, label="6m", step="month", stepmode="backward"),

            dict(count=1, label="1y", step="year", stepmode="backward"),

            dict(step="all")

        ])

    )

)

covid19_dataset["Date"] = pd.to_datetime(covid19_dataset['Date'])

covid19_dataset = covid19_dataset.sort_values(by="Date")

covid19_dataset = covid19_dataset[covid19_dataset['Date'] < pd.Timestamp("today").strftime("%d/%m/%Y")]

covid19_dataset.head(25)

aggregation_functions = {'Confirmed': 'sum'}

confirmed_cases_with_dates = covid19_dataset.groupby(covid19_dataset['Date']).aggregate(aggregation_functions)



fig = go.Figure(data=[go.Scatter(x=confirmed_cases_with_dates.index, y=confirmed_cases_with_dates['Confirmed'])])



fig.update_xaxes(

    rangeslider_visible=True,

    rangeselector=dict(

        buttons=list([

            dict(count=1, label="1m", step="month", stepmode="backward"),

            dict(count=6, label="6m", step="month", stepmode="backward"),

            dict(count=1, label="1y", step="year", stepmode="backward"),

            dict(step="all")

        ])

    )

)



fig.update_layout(

    title="Live dashboard of Covid-19 cases reported in India",

    xaxis_title="Dates",

    yaxis_title="Number of cases",

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="#7f7f7f"

    )

)



fig.show();
plt.figure(figsize=(15,8))

health_fac = hospital_beds_dataset.sort_values('TotalPublicHealthFacilities_HMIS', ascending=False)

health_fac = health_fac[health_fac['State/UT'] != 'All India']

ax = sns.barplot(health_fac['State/UT'], health_fac['TotalPublicHealthFacilities_HMIS'], palette="rocket")

ax.set_xticklabels(ax.get_xticklabels(), rotation=80)

addStyleToPlots(ax, 'Public health facilities','States', 'Number of facilities') 



plt.plot();
plt.figure(figsize=(15,8))

public_beds = hospital_beds_dataset.sort_values('NumPublicBeds_HMIS', ascending=False)

public_beds = public_beds[public_beds['State/UT'] != 'All India']

ax = sns.barplot(public_beds['State/UT'], public_beds['NumPublicBeds_HMIS'], palette="rocket")

ax.set_xticklabels(ax.get_xticklabels(), rotation=80)

addStyleToPlots(ax, 'Public beds availablilty','States', 'Number of beds') 



plt.plot();
plt.figure(figsize=(15,8))

ax = sns.scatterplot(icmr_testing_details_dataset.TotalSamplesTested, icmr_testing_details_dataset.TotalPositiveCases, palette="rocket");



addStyleToPlots(ax, 'Confirmed positive with respect to samples tested','Samples Tested', 'Positive cases') 

plt.plot();

plt.figure(figsize=(10,7))

gender_data = individual_details_dataset.sort_values('gender', ascending=False)

ax = sns.countplot(gender_data.gender);

addStyleToPlots(ax, 'Confirmed Covid-19 cases with respect to gender','Gender', 'Count') 



plt.plot();
plt.figure(figsize=(10,7))

lab_type = icmr_testing_labs_dataset.type[icmr_testing_labs_dataset.type!='Collection Site']

ax = sns.countplot(lab_type)

addStyleToPlots(ax, 'Covid-19 testing labs','Labs', 'Count') 

plt.plot();
def drawPieGraph(labels, sizes, title):

    explode = []



    for i in labels:

        explode.append(0.05)

    

    plt.figure(figsize= (10,10))

    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=9, explode =explode)

    centre_circle = plt.Circle((0,0),0.75,fc='white')

    plt.suptitle(title, size=20, y=1.005);      



    fig = plt.gcf()

    fig.gca().add_artist(centre_circle)

    plt.axis('equal')  

    plt.tight_layout()
drawPieGraph(age_group_dataset['AgeGroup'], age_group_dataset['TotalCases'], "Age of the people effected by Covid-19 with percentages")
def drawIndianMap(data_values):

    # where /r "c:\Users\lenovo" epsg.*

    # c:\Users\lenovo\miniconda3\Library\share\epsg

    # c:\Users\lenovo\miniconda3\pkgs\proj4-5.2.0-h6538335_1006\Library\share\epsg

    # c:\Users\lenovo\miniconda3\pkgs\proj4-5.2.0-ha925a31_1\Library\share\epsg    

    

    import matplotlib.pyplot as plt

    import matplotlib.cm

 

    import os

    os.environ['PROJ_LIB'] = "C:\\Users\lenovo\\miniconda3\\Library\\share"

    import mpl_toolkits.basemap as Basemap



    from mpl_toolkits.basemap import Basemap

    from matplotlib.patches import Polygon

    from matplotlib.collections import PatchCollection

    from matplotlib.colors import Normalize

    

    data = list(data_values.fillna(0).values)

    fig, ax = plt.subplots()



    basemap = Basemap(resolution= "c", projection='merc', lat_0=54.5, lon_0=-4.36 ,

                 llcrnrlon=68, llcrnrlat=6.,urcrnrlon=97.,urcrnrlat=37.)



    basemap.drawmapboundary(fill_color='#46bcec')

    basemap.fillcontinents(color='#f2f2f2', lake_color='#46bcec')

    basemap.drawcoastlines()

    

    basemap.readshapefile('../input/covid19data/INDIA','INDIA');





    shapes = [Polygon(np.array(shape), True) for shape in basemap.INDIA]

    cmap = plt.get_cmap('Greys')



    pc = PatchCollection(shapes, zorder=2)

    

    india_info = pd.DataFrame(basemap.INDIA_info)

    aggregation_functions = {'RINGNUM': 'min'}

    grouped_india_info = pd.DataFrame(india_info.groupby('ST_NAME'))

    a = pd.DataFrame(grouped_india_info[1][0])

    state_list = list(a['RINGNUM'].fillna(0).values)



    norm = Normalize()

    pc.set_facecolor(cmap(norm(state_list)))

    ax.add_collection(pc)

    

    mapper = matplotlib.cm.ScalarMappable(cmap=cmap)

    mapper.set_array(data)

    plt.colorbar(mapper, shrink=0.4)



    ax.set_title('Covid-19 India saturation map', fontsize=24)

    ttl = ax.title

    ttl.set_position([.5, 1.05])

    plt.figure(figsize=(20,25))

    plt.show();
drawIndianMap(state_wise['Confirmed'])