import pandas as pd
import numpy as np
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
import squarify as sq

sns.set(color_codes=True)

%matplotlib inline
#from datetime import datetime
source = pd.read_csv("../input/train.csv", sep=",", low_memory=False)
#source = pd.read_csv('train.csv', sep=',', low_memory=False)
#source.head()
no_poly_source = source.loc[:, source.columns != 'POLYLINE']
#no_poly_source.head()
miss_false_no_poly_source = no_poly_source[no_poly_source.MISSING_DATA == False]
miss_false_no_poly_source.head()
new_source = miss_false_no_poly_source.copy()
new_source['WEEK_DAY'] = new_source.TIMESTAMP.apply(lambda x: dt.datetime.fromtimestamp(x).weekday())
new_source['YEAR'] = new_source.TIMESTAMP.apply(lambda x: dt.datetime.fromtimestamp(x).year)
new_source['MONTH'] = new_source.TIMESTAMP.apply(lambda x: dt.datetime.fromtimestamp(x).month)
new_source['MONTH_DAY'] = new_source.TIMESTAMP.apply(lambda x: dt.datetime.fromtimestamp(x).day)
new_source['HOUR'] = new_source.TIMESTAMP.apply(lambda x: dt.datetime.fromtimestamp(x).hour)
new_source['DATE'] = new_source.TIMESTAMP.apply(lambda x: dt.datetime.fromtimestamp(x).date().isoformat())
new_source['DATE_DATE'] = new_source.TIMESTAMP.apply(lambda x: dt.datetime.fromtimestamp(x).date())
new_source.tail()
plt.figure(figsize=(10,10))
patches, texts, autotexts = plt.pie(new_source.YEAR.value_counts().values
        , labels=new_source.YEAR.value_counts().keys()
        , autopct='%1.2f%%')

for t in texts:
    t.set_fontsize(20)
for t in autotexts:
    t.set_fontsize(20)
autotexts[0].set_color('y')
plt.show()
time_plot = pd.DataFrame({'value': new_source.DATE.value_counts().values, 'date': new_source.DATE.value_counts().keys()})
#time_plot.set_index('date', inplace=True)
time_plot = time_plot.set_index('date')['value']
time_plot.sort_index(inplace=True)
time_plot.keys().astype(np.datetime64)

# Plotting
plt.figure(figsize=(20,10))
plt.rc('font', size=25)          # controls default text sizes
plt.rc('axes', titlesize=25)
plt.rc('axes', labelsize=17)
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels

plt.ylabel('Number of Trips')
plt.xlabel('Date of Trips')
plt.title('Evolution of number of trips (2013/07 - 2014/07)')

plt.plot_date(time_plot.keys(), time_plot.values, 'bo-', alpha=0.6)

plt.xticks(rotation = 45
           , size = 14)
# No pinta las fechas
#time_plot.plot(alpha=0.5, style='bo-')

plt.show();
keys_month = np.arange(1,13)
values_month = ['January','February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
month_names = dict(zip(keys_month,values_month))

plt.figure(figsize=(15,10))
my_colors = 'g'
ax = plt.bar(new_source.MONTH.value_counts().keys()
        , new_source.MONTH.value_counts().values
        , color = my_colors)
plt.xticks(new_source.MONTH.value_counts().keys()
           , new_source.MONTH.map(month_names).value_counts().keys()
           , rotation = 45
           , size = 14)
plt.xlabel('Months', size = 14)
plt.ylabel('Number of trips', size = 14)
plt.show()
plt.figure(figsize=(15,10))
my_colors = 'b'
ax = plt.bar(new_source.MONTH_DAY.value_counts().keys()
        , new_source.MONTH_DAY.value_counts().values
        , color = my_colors)
plt.xticks(new_source.MONTH_DAY.value_counts().keys()
           , new_source.MONTH_DAY.value_counts().keys()
           , rotation = 45
           , size = 14)
plt.xlabel('Day of the month', size = 14)
plt.ylabel('Number of trips', size = 14)
plt.show()

#new_source.MONTH_DAY.value_counts().plot(kind='bar', figsize=(7,7), cmap='Paired', use_index=False)
keys_week = np.arange(0,7)
values_week = ['Monday','Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
week_days_names = dict(zip(keys_week,values_week))

plt.figure(figsize=(15,10))
my_colors = 'rgbkymc'
ax = plt.bar(new_source.WEEK_DAY.value_counts().keys()
        , new_source.WEEK_DAY.map(week_days_names).value_counts().values
        , color = my_colors)

plt.xticks(new_source.WEEK_DAY.value_counts().keys()
           , new_source.WEEK_DAY.map(week_days_names).value_counts().keys()
           , rotation = 45
           , size = 14)
plt.xlabel('Week day', size = 14)
plt.ylabel('Number of trips', size = 14)
plt.show()
plt.figure(figsize=(7,7))
plt.rc('font', size=25)          # controls default text sizes
plt.rc('axes', titlesize=25)
plt.rc('axes', labelsize=17)
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
plt.pie(new_source.CALL_TYPE.value_counts().values
        , labels=new_source.CALL_TYPE.value_counts().keys()
        , autopct='%1.2f%%')
plt.show()
plt.figure(figsize=(7,7))
plt.rc('font', size=25)          # controls default text sizes
plt.rc('axes', titlesize=25)

colors = ['#ff0000','#0080ff','#ff8000']
plt.title('Call_type comparation')
sq.plot(sizes=new_source.CALL_TYPE.value_counts().values
, label=new_source.CALL_TYPE.value_counts().keys()
, color=colors, alpha=.4);

plt.show()
call_type_df = new_source[['TRIP_ID','CALL_TYPE','WEEK_DAY']]
call_type_df = call_type_df.groupby(['WEEK_DAY','CALL_TYPE']).count()
unstack_call_type_df = call_type_df.unstack()
unstack_call_type_df
r = np.arange(7)
plt.figure(figsize=(10,10))
plt.rc('font', weight='bold')

p1 = plt.bar(unstack_call_type_df.index, unstack_call_type_df.TRIP_ID.A.values, label = 'A')
p2 = plt.bar(unstack_call_type_df.index, unstack_call_type_df.TRIP_ID.B.values, bottom=unstack_call_type_df.TRIP_ID.A.values, label = 'B')
p3 = plt.bar(unstack_call_type_df.index, unstack_call_type_df.TRIP_ID.C.values, bottom=unstack_call_type_df.TRIP_ID.B.values, label = 'C')

plt.xticks(r
           , values_week
           , rotation = 45
           , size = 14)
plt.xlabel('Week Day', size = 14)
plt.ylabel('Number of trips', size = 14)

plt.legend()

plt.show()
width = 0.25       # the width of the bars

plt.figure(figsize=(25,15))
plt.rc('font', size=25)          # controls default text sizes
plt.rc('axes', titlesize=25)
plt.rc('axes', labelsize=17)
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels

rects1 = plt.bar(unstack_call_type_df.index, unstack_call_type_df.TRIP_ID.A.values, width, color='b', label = 'A')
rects2 = plt.bar(unstack_call_type_df.index + width, unstack_call_type_df.TRIP_ID.B.values, width, color='g', label = 'B')
rects3 = plt.bar(unstack_call_type_df.index + (width*2), unstack_call_type_df.TRIP_ID.C.values, width, color='r', label = 'C')

plt.xticks(r
           , values_week
           , rotation = 45
           , size = 24)

plt.rc('font', weight='bold')
plt.xlabel('Día de la semana', size = 25)
plt.ylabel('Numero de viajes', size = 25)

plt.legend()

plt.show();
miss_false_poly_source = source[source.MISSING_DATA == False]
miss_false_poly_source2 = miss_false_poly_source[miss_false_poly_source.POLYLINE != '[]']
miss_false_polyline_source = miss_false_poly_source2[['TRIP_ID', 'CALL_TYPE', 'POLYLINE', 'TIMESTAMP']]
polyline_source = miss_false_polyline_source.copy()
polyline_source.reset_index(inplace=True)
polyline_source['PICK_UP_LOCATION'] = polyline_source.POLYLINE.apply(lambda x: eval(x.split()[0])[0])
polyline_source['PICK_UP_LOCATION'] = polyline_source.PICK_UP_LOCATION.apply(lambda x: np.flip(x,0))
polyline_source['DISTANCE'] = polyline_source.POLYLINE.apply(lambda x: len(x))
polyline_source['TIME_TRIP_MIN'] = polyline_source.DISTANCE.apply(lambda x: float((x-1)*15)/60)
polyline_source['WEEK_DAY'] = polyline_source.TIMESTAMP.apply(lambda x: dt.datetime.fromtimestamp(x).weekday())
polyline_source = polyline_source.loc[:, polyline_source.columns != 'index']
polyline_source.head()
C_polyline_source = polyline_source.copy()
C_polyline_source = C_polyline_source[C_polyline_source.CALL_TYPE == 'C']
mondays_pick_up = C_polyline_source[C_polyline_source.WEEK_DAY == 0].PICK_UP_LOCATION.tolist()
sundays_pick_up = C_polyline_source[C_polyline_source.WEEK_DAY == 6].PICK_UP_LOCATION.tolist()
list_mondays_pick_up = []
list_sundays_pick_up = []
#list_a_pick_up = []
for i in mondays_pick_up:
    list_mondays_pick_up.append(i.tolist())
    
for i in sundays_pick_up:
    list_sundays_pick_up.append(i.tolist())
# Create Monday Map
monday = folium.Map([41.155, -8.63], zoom_start=13)
# Add heatMap 
plugins.HeatMap(list_mondays_pick_up, radius=11).add_to(monday)
# Print heatMap
monday
# Create Sunday Map
sunday = folium.Map([41.155, -8.63], zoom_start=13)
# Add heatMap 
plugins.HeatMap(list_sundays_pick_up, radius=11).add_to(sunday)
# Print heatMap
sunday