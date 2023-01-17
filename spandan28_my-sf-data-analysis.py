import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

data = pd.read_csv("../input/restaurant-scores-lives-standard.csv")

data["inspection_date"] = pd.to_datetime(data["inspection_date"],format="%Y-%m-%dT%H:%M:%S")

inspection_by_year_month = data["inspection_date"].groupby([data.inspection_date.dt.year, data.inspection_date.dt.month]).agg("count")

inspection_by_year_month = inspection_by_year_month.to_frame()

# CReate a column of tuples which incdicate inspection motnh and date

inspection_by_year_month["Date"] = inspection_by_year_month.index

# create a new column just for inspections

inspection_by_year_month = inspection_by_year_month.rename(columns={inspection_by_year_month.columns[0]:"Inspection"})



inspection_by_year_month["Date"] = pd.to_datetime(inspection_by_year_month["Date"],format="(%Y, %m)")

#reset the index, to remove dates from index

inspection_by_year_month = inspection_by_year_month.reset_index(drop=True)

inspection_by_year_month["month"] = inspection_by_year_month.Date.dt.month

inspection_by_year_month["year"] = inspection_by_year_month.Date.dt.year

data["inspection_date"] = pd.to_datetime(data["inspection_date"],format="%Y-%m-%dT%H:%M:%S")

data[["business_name"]].groupby([data.inspection_date.dt.year, data.inspection_date.dt.month])

business = data.groupby(["business_name"])

risk_measurements = business.risk_category.value_counts(dropna=True).to_frame()

risk_name = pd.DataFrame(risk_measurements.index.values.tolist(),columns=["Name","risk_cat"])

risk_name["occurances"] = risk_measurements.risk_category.values.tolist()

risk_name = risk_name.pivot_table(index="Name",values="occurances",columns="risk_cat")

risk_name = risk_name.fillna(0)

risk_name["BusinessName"] = risk_name.index

risk_name = risk_name.reset_index().drop(columns="Name")

risk_name["High Risk"] = 100* risk_name["High Risk"]/(risk_name.drop(columns="BusinessName")).sum(axis=1)

risk_name["Low Risk"] = 100* risk_name["Low Risk"]/(risk_name.drop(columns="BusinessName")).sum(axis=1)

risk_name["Moderate Risk"] = 100* risk_name["Moderate Risk"]/(risk_name.drop(columns="BusinessName")).sum(axis=1)

def find_long_lat(data_frame,list_buss):

    long=dict()

    lat = dict()

    for business in list_buss:

        long[business] = data_frame.loc[data_frame.business_name==business,"business_longitude"].unique()[0]

        lat[business] = data_frame.loc[data_frame.business_name==business,"business_latitude"].unique()[0]

    return long, lat   

    
long, lat = find_long_lat(data,risk_name.BusinessName.values.tolist() )

long = pd.DataFrame.from_dict(long, orient="index")

lat = pd.DataFrame.from_dict(lat, orient="index")

risk_name["longitude"] = long.iloc[:,0].values.tolist()

risk_name["latitude"] = lat.iloc[:,0].values.tolist()

risk_name = risk_name.dropna()
import plotly.plotly as py

import plotly.graph_objs as go

#display the code in notebook

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()

trace = go.Scatter(x=inspection_by_year_month.Date, y= inspection_by_year_month.Inspection)

data_inspection = [trace]

layout = dict(title="Number of inspections", xaxis=dict(title="Month/Year",  zeroline=False))

figure = dict(data=data_inspection, layout=layout)

iplot(figure)
import folium

map_sf = folium.Map(location=[risk_name.longitude.mean(), risk_name.latitude.mean()],tiles = "Stamen Toner",zoom_start=10)

risk_name.apply(lambda row:folium.CircleMarker(location=[row.latitude,row.longitude]).add_child(folium.Popup(row.BusinessName)).add_to(map_sf),axis=1)

map_sf