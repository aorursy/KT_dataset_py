# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df_confirmed  =pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
df_deaths = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

from fbprophet import Prophet
from datetime import datetime
df_deaths.shape
df_deaths.head()
now=datetime.now().strftime("%Y-%m-%d")
today = datetime.strptime(now,"%Y-%m-%d")
last_date = datetime.strptime("2020-05-10","%Y-%m-%d")
period = abs(last_date-today).days

def data_mun(rows):
    deaths = rows.iloc[4:]
    periods =len(rows)-4
    dates = pd.date_range(start="2020-01-22",periods=periods,freq="D")

    frame = {'ds':dates,'y':deaths}
    main_df = pd.DataFrame(frame)
    #print (main_df.head(),type(main_df))
    m = Prophet()
    cls = m.fit(main_df)
    
#print ("We are predicting the deaths for the next "+str(period)[:8]+" from today")
    future =m.make_future_dataframe(periods=int(period))    
    forecast = m.predict(future)
    forecast=forecast[['ds', 'yhat']]
    #forecast.columns = forecast.iloc[0]
    #forecast has all the predicted values for futures dates.
    
    return forecast
    
coloumns=['ds','yhat','country','lat-lon']
master_df = pd.DataFrame(columns=coloumns)
c=0
death_prediction=[]
for index, row in df_deaths.iterrows():
    predicted =data_mun(row)
    predicted['country'],predicted['lat'], predicted['lon']= row.iloc[1],row.iloc[2],row.iloc[3]
    predicted = predicted[predicted['ds']>now]
    c+=1
    death_prediction.append(predicted)
    
   # master_df = pd.concat([master_df,predicted])
    #here i have a dataframe for with predicted deaths on each future date and i am appeding these data frames to a list,
    #which i will concatenate later on.
    
       
 

#concatenating all the df's for the diff countreuis to single data frame
master_df = pd.concat(death_prediction).reset_index()

#master_df =master_df.query('yhat>50').reset_index()
master_df= master_df.drop(['index'],axis=1)


# grouping the master df on the basis of countries and will find the mean of the death rate till 10 may 2020.
master_df_mean =master_df.groupby(['country']).mean()[['yhat','lat','lon']].reset_index()

import mapclassify
import geoplot
import geopandas

world = geopandas.read_file(
    geopandas.datasets.get_path('naturalearth_lowres')
)
gdf = geopandas.GeoDataFrame(
    master_df_mean ,geometry=geopandas.points_from_xy(master_df_mean.lat, master_df_mean.lon))
print(gdf.head())

scheme = mapclassify.Quantiles(gdf['yhat'], k=10)
geoplot.choropleth(world, hue=gdf['yhat'], scheme=scheme,cmap='OrRd', figsize=(20, 20), legend=True
)  

#lets find the death prediction for the Italy
india = master_df.loc[(master_df.country == "India")]
india.shape
india.head()
#lets plot the data 
import matplotlib.pyplot as plt
fig  = plt.figure(figsize=(10,6))
ax= fig.add_axes([0,0,1,1])
ax.set_title("Predicted deaths in India for til MAy 10 2020")
ax.plot(india['ds'],india['yhat'])
import plotly.offline as pyo
import plotly.express as px
import plotly.graph_objects as go
pyo.init_notebook_mode()
fig= go.Figure([go.Scatter(x=india['ds'], y=india['yhat'])])
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=20, label="1m", step="day", stepmode="backward"),
            
            
        ])
    )
)
fig.show()
#i found plotly better than ,matplotlib as its interactive

#fig = px.line(india, x=india['ds'], y=india['yhat'])
#fig.show()