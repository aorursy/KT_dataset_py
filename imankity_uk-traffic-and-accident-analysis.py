from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import col,broadcast
from pyspark.sql import functions as F
from matplotlib import pyplot as plt
import pandas as pd
sc = SparkContext()
spark = SparkSession(sc)
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
accident_info = spark.read.csv("../input/Accident_Information.csv",header=True,inferSchema=True)
vehicle_info = spark.read.csv("../input/Vehicle_Information.csv",header=True,inferSchema=True)
combinedDF = vehicle_info.join(accident_info,['Accident_Index'],'inner').drop(vehicle_info.Year)
combinedDF.filter((combinedDF['Sex_of_Driver'] == 'Male') & (col('Age_Band_of_Driver').like('26%'))).limit(10).show()
dataDF=combinedDF.select(combinedDF.Accident_Index,combinedDF.Year,combinedDF.Number_of_Casualties).groupBy('Year').agg({'Accident_Index' : 'count'}).orderBy('Year').toPandas()
year = dataDF['Year'].tolist()

casualties = dataDF['count(Accident_Index)'].tolist()
trace = go.Bar(
    x=year,
    y=casualties,
    name='Female Drivers',
    marker=dict(color='#f7bb31'),
    opacity=0.8
)
data = [trace]
layout = go.Layout(
    barmode='group',
    legend=dict(dict(x=-.1, y=1.2)),
    margin=dict(b=120),
    title = 'Number of Casualties per Year',
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='grouped-bar')

plt.show()
vehicle_ManoeuvreDF=combinedDF.select('Accident_Index','Vehicle_Manoeuvre').groupBy('Vehicle_Manoeuvre').agg({'Accident_Index' : 'count'}).orderBy('count(Accident_Index)').limit(10).toPandas()
manoeuvre = vehicle_ManoeuvreDF['Vehicle_Manoeuvre'].tolist()

count_manoeuvre = vehicle_ManoeuvreDF['count(Accident_Index)'].tolist()
pie_visual = go.Pie(labels=manoeuvre, values=count_manoeuvre, marker=dict(colors=['#25e475', '#ee1c96',]))

layout = go.Layout(title='Reason of Accident', width=800, height=500)
fig = go.Figure(data=[pie_visual], layout=layout)
iplot(fig)

genderDF=combinedDF.select('Accident_Index','Year','Age_Band_of_Driver','Sex_of_Driver').groupBy('Sex_of_Driver','Age_Band_of_Driver').agg({'Accident_Index' : 'count'}).orderBy('count(Accident_Index)',ascending=False).toPandas()
maleDF=genderDF[genderDF['Sex_of_Driver']=='Male'].head(6)
male_age_band=maleDF['Age_Band_of_Driver'].tolist()
male_accident_count=maleDF['count(Accident_Index)'].tolist()
femaleDF=genderDF[genderDF['Sex_of_Driver']=='Female'].head(6)
female_age_band=femaleDF['Age_Band_of_Driver'].tolist()
female_accident_count=femaleDF['count(Accident_Index)'].tolist()
trace1 = go.Bar(
    x=male_age_band,
    y=male_accident_count,
    name='Male Drivers',
    marker=dict(color='#f259d6'),
    opacity=0.8
)
trace2 = go.Bar(
    x=female_age_band,
    y=female_accident_count,
    name='Female Drivers',
    marker=dict(color='#f7bb31'),
    opacity=0.8
)
data = [trace1, trace2]
layout = go.Layout(
    barmode='group',
    legend=dict(dict(x=-.1, y=1.2)),
    margin=dict(b=120),
    title = 'Accidents categorized by gender',
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='grouped-bar')

plt.show()
journey_purposeDF=combinedDF.select('Accident_Index','Journey_Purpose_of_Driver').groupBy('Journey_Purpose_of_Driver').agg({'Accident_Index' : 'count'}).orderBy('count(Accident_Index)',ascending=False).toPandas()
journey_label=journey_purposeDF['Journey_Purpose_of_Driver'].tolist()
accident_count=journey_purposeDF['count(Accident_Index)'].tolist()
accident_count
pie_visual = go.Pie(labels=journey_label, values=accident_count, marker=dict(colors=['#25e475', '#ee1c96',]))

layout = go.Layout(title='Journey Purpose while Accident Happened', width=800, height=500)
fig = go.Figure(data=[pie_visual], layout=layout)
iplot(fig)
accident_roadDF=combinedDF.select('Accident_Index','Road_Type').groupBy('Road_Type').agg({'Accident_Index' : 'count'}).orderBy('count(Accident_Index)',ascending=False).toPandas()
road_type=accident_roadDF['Road_Type'].tolist()
accident_count=accident_roadDF['count(Accident_Index)'].tolist()
pie_visual = go.Pie(labels=road_type, values=accident_count, marker=dict(colors=['#25e475', '#ee1c96',]))

layout = go.Layout(title='Journey Purpose while Accident Happened', width=800, height=500)
fig = go.Figure(data=[pie_visual], layout=layout)
iplot(fig)
day_of_week_DF=combinedDF.select('Accident_Index','Day_of_Week').groupBy('Day_of_Week').agg({'Accident_Index' : 'count'}).orderBy('count(Accident_Index)',ascending=False).toPandas()
weekday_name=day_of_week_DF['Day_of_Week'].tolist()
accident_count=day_of_week_DF['count(Accident_Index)'].tolist()
pie_visual = go.Pie(labels=weekday_name, values=accident_count, marker=dict(colors=['#25e475', '#ee1c96',]))

layout = go.Layout(title='Accidents categorized in Day of Week', width=800, height=500)
fig = go.Figure(data=[pie_visual], layout=layout)
iplot(fig)
vehicle_makerDF=combinedDF.select('Accident_Index','make').groupBy('make').agg({'Accident_Index' : 'count'}).orderBy('count(Accident_Index)',ascending=False).limit(10).toPandas()
maker_name=vehicle_makerDF['make'].tolist()
accident_count=vehicle_makerDF['count(Accident_Index)'].tolist()
pie_visual = go.Pie(labels=maker_name, values=accident_count, marker=dict(colors=['#25e475', '#ee1c96',]))

layout = go.Layout(title='Accidents categorized by vehicle maker', width=800, height=500)
fig = go.Figure(data=[pie_visual], layout=layout)
iplot(fig)
vehicleDF= combinedDF.select(combinedDF.Accident_Index,combinedDF.Age_Band_of_Driver,combinedDF.Age_of_Vehicle,combinedDF.Journey_Purpose_of_Driver,combinedDF.make,combinedDF.model, combinedDF.Junction_Location).filter((col('Age_Band_of_Driver').like('26%')))

accidentDF= combinedDF.select(combinedDF.Accident_Index,combinedDF.Date,combinedDF.Day_of_Week,combinedDF.Road_Type,combinedDF.Speed_limit,combinedDF.Year).filter(combinedDF['Day_of_Week'] == 'Tuesday').orderBy('Year',ascending=True)

''' Performing Left Outer Join on '''

filteredDF = accidentDF.join(vehicleDF,['Accident_Index'], "left_outer").drop(accidentDF.Year)
'''Performing Broadcast Join'''

broadcastDF = combinedDF.join(filteredDF,['Accident_Index'])

df1 = combinedDF.alias('df1')
df2 = filteredDF.alias('df2')

drop_list = [c for c in df1.columns if c in df2.columns]
for col in drop_list:
    broadcastDF = broadcastDF.drop(df2[col])

broadcastDF.show()
genderDF=combinedDF.select('Accident_Index','Number_of_Casualties','Sex_of_Driver').groupBy('Sex_of_Driver','Number_of_Casualties').agg({'Accident_Index' : 'count'}).orderBy('count(Accident_Index)',ascending=False).toPandas()
maleDF=genderDF[genderDF['Sex_of_Driver']=='Male'].head(6)
male_age_band=maleDF['Number_of_Casualties'].tolist()
male_accident_count=maleDF['count(Accident_Index)'].tolist()

femaleDF=genderDF[genderDF['Sex_of_Driver']=='Female'].head(6)
female_age_band=femaleDF['Number_of_Casualties'].tolist()
female_accident_count=femaleDF['count(Accident_Index)'].tolist()

male_age_band
plt.bar(male_age_band,male_accident_count,label="Men",color='r')
plt.bar(female_age_band,female_accident_count,label="Women",color='b')
plt.legend()
plt.xlabel("Gender Wise Casuaties")
plt.ylabel("Number of Casualties")
plt.title("Gender")
