import pandas as pd

google_df=pd.read_csv("/kaggle/input/enriched-global-mobility-data-apple-google/google_mobility.csv")

len(google_df.country_region.unique())
google_df["retail_and_recreation_percent_change_from_baseline"]
google_df["parks_percent_change_from_baseline"]
def changeFromPercent(x):

    if x!=0:

        return (1+(x/100))*100



    else:

        return 100
google_df["retail_and_recreation_percent_change_from_baseline"]=google_df["retail_and_recreation_percent_change_from_baseline"].apply(changeFromPercent)

google_df["grocery_and_pharmacy_percent_change_from_baseline"]=google_df["grocery_and_pharmacy_percent_change_from_baseline"].apply(changeFromPercent)

google_df["parks_percent_change_from_baseline"]=google_df["parks_percent_change_from_baseline"].apply(changeFromPercent)

google_df["transit_stations_percent_change_from_baseline"]=google_df["transit_stations_percent_change_from_baseline"].apply(changeFromPercent)

google_df["workplaces_percent_change_from_baseline"]=google_df["workplaces_percent_change_from_baseline"].apply(changeFromPercent)

google_df['residential_percent_change_from_baseline']=google_df['residential_percent_change_from_baseline'].apply(changeFromPercent)
google_df
google_df.drop(["wikidata","sub_region_1","sub_region_2"],axis=1,inplace=True)
import datetime



string = "2020-02-15"

date = datetime.datetime.strptime(string, "%Y-%m-%d")

print(date.timetuple().tm_yday)

print(date.day)
def stripthedate(x):

    return datetime.datetime.strptime(x, "%Y-%m-%d")

      

    

       



google_df["day"]=google_df["date"].apply(stripthedate)

google_df["month"]=google_df["date"].apply(stripthedate)

google_df["year"]=google_df["date"].apply(stripthedate)

google_df["day_in_year"]=google_df["date"].apply(stripthedate)



       



google_df["day"]=google_df["day"].apply(lambda x: int(x.day))

google_df["month"]=google_df["month"].apply(lambda x: int(x.month))

google_df["year"]=google_df["year"].apply(lambda x: int(x.year))

google_df["day_in_year"]=google_df["day_in_year"].apply(lambda x: int(x.timetuple().tm_yday))
retail='retail_and_recreation_percent_change_from_baseline'



grocery='grocery_and_pharmacy_percent_change_from_baseline'

parks=   'parks_percent_change_from_baseline'

transit=  'transit_stations_percent_change_from_baseline'

workplace= 'workplaces_percent_change_from_baseline'

residential= 'residential_percent_change_from_baseline'
india_df=google_df[google_df["country_region"]=="India"]

india_df["day"]
#!pip install -U vega_datasets notebook vega


import altair_render_script


import altair as alt

import json





#alt.renderers.enable('kaggle')

a=alt.Chart(india_df).transform_fold([grocery,parks,residential,workplace,transit],as_=["pro","values"]).mark_area().encode(

    alt.X('day_in_year:Q'),

    alt.Y('values:Q'),  alt.Color('pro:N'),

    alt.Row("pro:N"),

    ).properties(

    

    

).interactive(bind_y=False)

a

alt.Chart(india_df.groupby(["month"]).sum().reset_index()).transform_fold([grocery,parks,residential,workplace,transit],as_=["pro","values"]).mark_bar().encode(

    alt.X('pro:O',axis=alt.Axis(labels=False),title=None),

    alt.Y('values:Q',axis=alt.Axis(labels=False)),  alt.Color('pro:N',title=None),

    alt.Column('month:N')

   ).interactive()

