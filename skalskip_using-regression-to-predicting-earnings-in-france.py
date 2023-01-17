import numpy as np

import pandas as pd

import geopandas as gpd

from scipy.optimize import curve_fit

import seaborn as sns



from math import radians, cos, sin, asin, sqrt



from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

import matplotlib.colors as colors

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls
# Datasets

geography = pd.read_csv("../input/name_geographic_information.csv")

industry = pd.read_csv("../input/base_etablissement_par_tranche_effectif.csv")

salary = pd.read_csv("../input/net_salary_per_town_categories.csv")

population = pd.read_csv("../input/population.csv")

# Geojson for map creations

departments_map = gpd.read_file('../input/departements.geojson')
geography.info()
industry.info()
salary.info()
population.head(5)
population.info()
salary = salary[salary["CODGEO"].apply(lambda x: str(x).isdigit())]

salary["CODGEO"] = salary["CODGEO"].astype(int)
salary.describe()
salary_copy = salary.copy()
age = ["18-25 years old", "26-50 years old", ">50 years old"]

woman_age = ["SNHMF1814", "SNHMF2614", "SNHMF5014"]

woman_salary_age = salary[woman_age].mean().tolist()

man_age = ["SNHMH1814", "SNHMH2614", "SNHMH5014"]

man_salary_age = salary[man_age].mean().tolist()



dif_in_prc_age = []

for w, m in zip(woman_salary_age, man_salary_age):

    dif_in_prc_age.append(round(abs(w-m)/m * 100, 2))

    

trace1 = go.Bar(

    x = age,

    y = woman_salary_age,

    name='Women',

    marker=dict(

        color='rgba(55, 128, 191, 0.7)',

        line=dict(

            color='rgba(55, 128, 191, 1.0)',

            width=2,

        )

    )

)

trace2 = go.Bar(

    x = age,

    y = man_salary_age,

    name='Men',

    marker=dict(

        color='rgba(219, 64, 82, 0.7)',

        line=dict(

            color='rgba(219, 64, 82, 1.0)',

            width=2,

        )

    )

)



trace3 = go.Scatter(

    x = age,

    y = dif_in_prc_age,

    name='Earnings difference',

    mode = 'lines+markers',

    yaxis='y2'

)



data = [trace1, trace2, trace3]

layout = go.Layout(

    barmode='group',

    title = 'Age and sex are',

    width=850,

    height=500,

    paper_bgcolor='rgb(244, 238, 225)',

    plot_bgcolor='rgb(244, 238, 225)',

    yaxis = dict(

        title= 'Average earnings [€/hour]',

        anchor = 'x',

        rangemode='tozero'

    ),

    xaxis = dict(title= 'Age'),

    

    yaxis2=dict(

        title='Earnings difference',

        titlefont=dict(

            color='rgb(148, 103, 189)'

        ),

        tickfont=dict(

            color='rgb(148, 103, 189)'

        ),

        overlaying='y',

        side='right',

        anchor = 'x',

        rangemode = 'tozero',

        dtick = 7.3

    ),

    #legend=dict(x=-.1, y=1.2)

    legend=dict(x=0.72, y=0.05)

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
positions = ["Executive", "Middle manager", "Employee", "Worker"]

woman_positions = ["SNHMFC14", "SNHMFP14", "SNHMFE14", "SNHMFO14"]

woman_salary_positions = salary[woman_positions].mean().tolist()

man_positions = ["SNHMHC14", "SNHMHP14", "SNHMHE14", "SNHMHO14"]

man_salary_positions = salary[man_positions].mean().tolist()



dif_in_prc = []

for w, m in zip(woman_salary_positions, man_salary_positions):

    dif_in_prc.append(round(abs(w-m)/m * 100, 2))



trace1 = go.Bar(

    x = positions,

    y = woman_salary_positions,

    name='Women',

    marker=dict(

        color='rgba(55, 128, 191, 0.7)',

        line=dict(

            color='rgba(55, 128, 191, 1.0)',

            width=2,

        )

    )

)

trace2 = go.Bar(

    x = positions,

    y = man_salary_positions,

    name='Men',

    marker=dict(

        color='rgba(219, 64, 82, 0.7)',

        line=dict(

            color='rgba(219, 64, 82, 1.0)',

            width=2,

        )

    )

)



trace3 = go.Scatter(

    x = positions,

    y = dif_in_prc,

    name='Earnings difference',

    mode = 'lines+markers',

    yaxis='y2'

)



data = [trace1, trace2, trace3]

layout = go.Layout(

    barmode='group',

    title = 'Stereotype is real',

    width=850,

    height=500,

    paper_bgcolor='rgb(244, 238, 225)',

    plot_bgcolor='rgb(244, 238, 225)',

    yaxis = dict(

        title= 'Average earnings [€/hour]',

        anchor = 'x',

        rangemode='tozero'

    ),

    xaxis = dict(title= 'Position'),

    

    yaxis2=dict(

        title='Earnings difference',

        titlefont=dict(

            color='rgb(148, 103, 189)'

        ),

        tickfont=dict(

            color='rgb(148, 103, 189)'

        ),

        overlaying='y',

        side='right',

        anchor = 'x',

        rangemode = 'tozero',

        dtick = 8

    ),

    #legend=dict(x=-.1, y=1.2)

    legend=dict(x=0.05, y=0.05)

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
row_fields = {}

row_fields["SNHMFC14"] = {"SEX": 1, "POSITION": 4}

row_fields["SNHMFP14"] = {"SEX": 1, "POSITION": 3}

row_fields["SNHMFE14"] = {"SEX": 1, "POSITION": 2}

row_fields["SNHMFO14"] = {"SEX": 1, "POSITION": 1}

row_fields["SNHMHC14"] = {"SEX": 2, "POSITION": 4}

row_fields["SNHMHP14"] = {"SEX": 2, "POSITION": 3}

row_fields["SNHMHE14"] = {"SEX": 2, "POSITION": 2}

row_fields["SNHMHO14"] = {"SEX": 2, "POSITION": 1}



reformatted_salary = []

for index, row in salary.iterrows():

    for key, value in row_fields.items(): 

        row_dict = {}

        row_dict["CODGEO"] = row["CODGEO"]

        row_dict["SEX"] = value["SEX"]

        row_dict["POSITION"] = value["POSITION"]

        row_dict["WAGE"] = row[key]

        reformatted_salary.append(row_dict)

        

reformatted_salary = pd.DataFrame(reformatted_salary)  
reformatted_salary.info()
# 1

geography["longitude"] = geography["longitude"].apply(lambda x: str(x).replace(',','.'))

# 2

mask = geography["longitude"] == '-'

geography.drop(geography[mask].index, inplace=True)

# 3

geography.dropna(subset = ["longitude", "latitude"], inplace=True)

# 4

geography["longitude"] = geography["longitude"].astype(float)
geography.drop_duplicates(subset=["code_insee"], keep="first", inplace=True)
def distance(lon1, lat1, lon2, lat2):

    # convert decimal degrees to radians 

    lon1 = radians(lon1)

    lat1 = radians(lat1)

    lon2 = radians(lon2)

    lat2 = radians(lat2)

    

    # haversine formula 

    dlon = lon2 - lon1 

    dlat = lat2 - lat1 

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2

    c = 2 * asin(sqrt(a)) 

    

    # Radius of earth in kilometers is 6371

    km = 6371* c

    return km



def distance_bigger_than(limit, lon1, lat1, lon2, lat2):

    dist = distance(lon1, lat1, lon2, lat2)

    if dist > limit:

        return 0

    else:

        return 1
# extracting the coordinates of paris

paris_lat = geography.loc[geography['nom_commune'] == "Paris"].iloc[0]["latitude"]

paris_lon = geography.loc[geography['nom_commune'] == "Paris"].iloc[0]["longitude"]



# auxiliary list that will hold calculated values of PARIS_CLOSE

distances = []



# calculating values of PARIS_CLOSE

for index, row in geography.iterrows():

    distances.append(distance_bigger_than(30, row["longitude"], row["latitude"], paris_lon, paris_lat))



# adding new column to DataFrame

geography["PARIS_CLOSE"] = pd.Series(distances, index=geography.index)
# extracting the coordinates of capitals of French provinces

majors =  geography[geography["nom_commune"] == geography["chef.lieu_région"]]

major_lats = majors["latitude"].tolist()

major_lons = majors["longitude"].tolist()



# auxiliary list that will hold calculated values of MAJOR_CITY_DISTANCE

distances = []



# calculating values of MAJOR_CITY_DISTANCE

for index, row in geography.iterrows():

    

    single_distances = []

    for lat, lon in zip(major_lats, major_lons):

        single_distances.append(int(distance(row["longitude"], row["latitude"], lon, lat)))

    

    distances.append(min(single_distances))



# adding new column to DataFrame

geography["MAJOR_CITY_DISTANCE"] = pd.Series(distances, index=geography.index)
salary_location = salary.merge(geography, how="left", left_on='CODGEO', right_on="code_insee")
salaries_by_dep = salary_location.groupby("numéro_département").mean()

departments_map = departments_map.merge(salaries_by_dep, how="left", left_on="code", right_index=True)

departments_map.dropna(subset = ["longitude", "latitude"], inplace=True)



fig, ax = plt.subplots(1, figsize=(15,14))

ax.set_title('Salary by Departments', size=32, x = 0.25, y=0.90)

fig.patch.set_facecolor((202/255, 204/255, 206/255))

departments_map.plot(ax=ax, column="SNHM14", cmap=plt.cm.plasma, scheme='fisher_jenks', k=10, legend=True)

leg = ax.get_legend()

ax.set_axis_off()

leg.set_bbox_to_anchor((0., 0., 0.2, 0.45))

leg.set_title("Mean net salary")
trace1 = go.Bar(

    x = ["Far [distance > 30km]", "Close [distance < 30km]"],

    y = salary_location.groupby("PARIS_CLOSE")["SNHM14"].mean().tolist(),

    name='Distance from Paris',

    marker=dict(

        color='rgba(55, 128, 191, 0.7)',

        line=dict(

            color='rgba(55, 128, 191, 1.0)',

            width=2,

        )

    )

)



data = [trace1]

layout = go.Layout(

    title = 'Distance from Paris',

    width=850,

    height=500,

    paper_bgcolor='rgb(202, 204, 206)',

    plot_bgcolor='rgb(202, 204, 206)',

    yaxis = dict(

        title= 'Average earnings [€/hour]',

        anchor = 'x',

        rangemode='tozero'

    ),

    xaxis = dict(title= 'Distance from Paris')

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
plt.figure(figsize=(16,10))

sns.set_style("whitegrid")

plt.title('Distance from closest major city', fontsize=20, fontweight='bold', y=1.05,)

plt.xlabel('Distance from closest major city [km]', fontsize=15)

plt.ylabel('Average earnings [€/hour]', fontsize=15)



years = salary_location["MAJOR_CITY_DISTANCE"].values

memory = salary_location["SNHM14"].values



plt.scatter(years, memory, edgecolors='black')

plt.show()
industry = industry[industry["CODGEO"].apply(lambda x: str(x).isdigit())]

industry["CODGEO"] = industry["CODGEO"].astype(int)
industry['MICRO'] = industry['E14TS1'] + industry['E14TS6']

industry['SMALL'] = industry['E14TS10'] + industry['E14TS20']

industry['MEDIUM'] = industry['E14TS50'] + industry['E14TS100']

industry['LARGE'] = industry['E14TS200'] + industry['E14TS500']



industry['SUM'] = industry['E14TS1'] + industry['E14TS6'] + industry['E14TS10'] + industry['E14TS20'] + industry['E14TS50'] + industry['E14TS100'] + industry['E14TS200'] + industry['E14TS500']
# merging datasets

full_dataset = reformatted_salary.merge(geography, how="left", left_on='CODGEO', right_on="code_insee")

full_dataset = full_dataset.merge(industry, how="left", on='CODGEO')

# deleting incomplete rows

full_dataset.dropna(inplace = True)

# selecting relevant columns

full_dataset = full_dataset[["POSITION", "SEX", "PARIS_CLOSE", "MAJOR_CITY_DISTANCE", "MICRO", "SMALL", "MEDIUM", "LARGE", "SUM", "WAGE"]]
colormap = plt.cm.viridis

plt.figure(figsize=(15,15))

plt.title('Correlation of Features', y=1.05, size=15)

sns.heatmap(full_dataset.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

plt.show()
X = full_dataset.iloc[:, :-1].values

y = full_dataset.iloc[:, 9].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression



regressor_lr = LinearRegression()

regressor_lr.fit(X_train, y_train)



# Predicting the Test set results

y_pred_lr = regressor_lr.predict(X_test)
from sklearn.ensemble import RandomForestRegressor



regressor_rfr = RandomForestRegressor(n_estimators = 10, random_state = 0)

regressor_rfr.fit(X_train, y_train)



# Predicting the Test set results

y_pred_rfr = regressor_rfr.predict(X_test)
def standard_error(y_pred, y_test):

    sum_value = 0

    for p, t in zip(y_pred, y_test):

        sum_value += (p - t)**2

    

    return((sum_value/len(y_test))**(1/2))
from sklearn.metrics import r2_score



score_lr = r2_score(y_pred_lr, y_test)

print("r2 for linear regression is equal " + str(round(score_lr, 3)))

score_rfr = r2_score(y_pred_rfr, y_test)

print("r2 for random forest regression is equal " + str(round(score_rfr, 3)))
error_lr = standard_error(y_pred_lr.tolist(), y_test.tolist())

print("standard error for linear regression is equal " + str(round(error_lr, 3)))

error_rfr = standard_error(y_pred_rfr.tolist(), y_test.tolist())

print("standard error for random forest regression is equal " + str(round(error_rfr, 3)))