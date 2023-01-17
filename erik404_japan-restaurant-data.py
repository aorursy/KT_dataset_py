

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import datetime

%matplotlib inline

from mpl_toolkits.basemap import Basemap

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.


# # CSV Data

# Air stores data [ air_store_id, air_genre_name, air_area_name, latitude, longitude ] 

air_stores = pd.read_csv('../input/air_store_info.csv')

# Air visit data  [ air_store_id, visit_date, visitors ]

air_visits = pd.read_csv('../input/air_visit_data.csv')

# Air reserve     [ air_store_id, visit_datetime, reserve_datetime, reserve_visitors ]

#air_reserve= pd.read_csv('../input/air_reserve.csv')



# Add locations to all restaurant visits

air_visit_loc = air_visits.merge(air_stores, on='air_store_id', how='left')



# Prindi filtre

air_visit_loc.head(3)



# # Statistics Functions



# Dets - Veebr Talv

def getWinter(d):

    w = [(d.visit_date >= '2015-12-01') & (d.visit_date < '2016-03-01'), 

         (d.visit_date >= '2016-12-01') & (d.visit_date < '2017-03-01')]

    return d[w[0] | w[1]]

# Jun - Aug Suvi

def getSummer(d):

    w = [(d.visit_date >= '2016-06-01') & (d.visit_date < '2016-09-01'), 

         (d.visit_date >= '2017-06-01') & (d.visit_date < '2017-09-01')]

    return d[w[0] | w[1]]

# Märts - Mai Kevad 

def getSpring(d):

    w = [(d.visit_date >= '2016-03-01') & (d.visit_date < '2016-05-01'), 

         (d.visit_date >= '2017-03-01') & (d.visit_date < '2017-05-01')]

    return d[w[0] | w[1]]

# Sept - Nov Sügis

def getAutumn(d):

    w = [(d.visit_date >= '2016-09-01') & (d.visit_date < '2016-11-01'), 

         (d.visit_date >= '2017-09-01') & (d.visit_date < '2017-11-01')]

    return d[w[0] | w[1]]



# Return top 3 popular genres [data, ammountOfReturnResults]

def genreTop(d, am):

    g = d.groupby(d.air_genre_name).agg({

        'visit_date':'first',

        'visitors':sum,

        'latitude':'first',

        'longitude':'first',

        'air_area_name': len,

        'air_genre_name':'first'

    }).rename(columns={'air_area_name':'stores'}).sort_values(by=['visitors'], ascending=False).head(am)

    # [visit_date, visitors, latitude, longitude, stores]

    return g



# Return object with every season top3 dataFrame

def getSeasonsTopGenre(d, am):

    tops = {

        'winter': genreTop(getWinter(d), am),

        'Spring': genreTop(getSpring(d), am),

        'Summer': genreTop(getSummer(d), am),

        'Autumn': genreTop(getAutumn(d), am)

    }

    return tops



# Get Popular percentage i.e. [60, 30, 10] represent percentages for top 3

def getPopPercentage(d):

    topSum = d.visitors.sum()

    topPer = [round(i/topSum*100) for i in d.visitors]

    return topPer



def makePlot(d):

    genrs = len(d.air_genre_name)

    ticks = list(d.visitors)

    

    labels = list(d.air_genre_name)

    

    plt.bar(genrs, ticks, align='center')

    plt.xticks(genrs, labels)

    plt.show()

# # Scatter - Total visits to restaurants AllOverJapan

h=air_visit_loc.longitude

v=air_visit_loc.latitude



jv = [24,46] # Japans latitude range

jh = [123,146] # Japans longitude



plt.axis(jh+jv)

plt.scatter(h,v, air_visit_loc.visitors*0.525, alpha=0.2, c=[0.3,0.91,0.27], zorder=1)



plt.show()





# setting the two corners of the map 

lon0, lat0 = (125, 25)

lon1, lat1 = (150, 46)

# setup Lambert Conformal basemap.

# set resolution=None to skip processing of boundary datasets.

m = Basemap(projection='merc',llcrnrlon=lon0, llcrnrlat=lat0,

            urcrnrlon=lon1, urcrnrlat=lat1,  resolution='l')

#m.bluemarble()

m.bluemarble(scale=3)   # full scale will be overkill



fig = plt.gcf()

fig.set_size_inches(8, 6.5)

plt.show()
# # Regions 

# Sendai

lon = air_visit_loc.latitude >= 40.0 # longitude

df_sen = air_visit_loc[lon] # Sendai region df

# Tokyo

lon = air_visit_loc.latitude < 40.0 # longitude

lat = air_visit_loc.longitude >= 137.0 # latitude

df_tok = air_visit_loc[lon & lat] # Tokio region df

# Osaka 

lon = air_visit_loc.latitude < 40.0 

lat = air_visit_loc.longitude < 137.0

lat1= air_visit_loc.longitude >= 132.0

df_osa = air_visit_loc[lon & lat & lat1] # Osaka region df

# Fukuoka

lon = air_visit_loc.latitude < 40.0 

lat = air_visit_loc.longitude < 132.0

df_fuk = air_visit_loc[lon & lat] # Fukuoka region df



# REGIONS [ sen , tok , osa , fuk ]

regionsTop = [genreTop(i, 3) for i in [df_sen, df_tok, df_osa, df_fuk] ]



def makePlot(d):

    

    genrs = np.arange(len(d.air_genre_name))

    ticks = list(d.visitors)

    

    labels = list(d.air_genre_name)

    plt.title('TEST')

    plt.bar(genrs, ticks, align='center')

    plt.xticks(genrs, labels)

    plt.show()



#makePlot(regionsTop[0])

# REGIONS [ sen , tok , osa , fuk ]

regionsTop[0]



#print(regionsTop[0], regionsTop[1], regionsTop[2], regionsTop[3])



#print(list(regionsTop[0].air_genre_name))

#t = np.arange(len(regionsTop[0].air_genre_name))

#print(np.arange(10))
# Sendai

regionsTop[0]
# Tokyo 

regionsTop[1]
# Osaka

regionsTop[2]
# Fukuoka

regionsTop[3]
# # Scatter - Relation of Restaurant count to its popularity

topShops = getSeasonsTopGenre(air_visit_loc, 8)





#  Colors  Autumn, Spring, Summer, Winter

c = [[0.11,1.00,0.42] ,[0.23,0.53,0.58] , [0.3,0.51,0.27], [0.49,0.93,0.47]]



# Labels

plt.ylabel('Külastajate arv kvartalis')

plt.xlabel('Restoranide arv')



# Colour counter

cc=0

# Restaurant genre providers relation to visitors

for i in topShops:

    h=topShops[i].stores 

    v=topShops[i].visitors

    plt.scatter(h,v, 55*(cc+1), alpha=1, c=c[cc], zorder=1-cc)

    cc+=1

plt.show()

#topShops

#  Colors heleroheline Autumn, Spring, Summer, Winter








# Loen air_visit_data sisse

df = pd.read_csv("../input/air_visit_data.csv")



# Võtan tulbad visit_date ja visitors. Grupeerin kuupäeva järgi ja liidan kõik ühe päeva külastajad kokku.

df = df[["visit_date", 'visitors']]

df = df.groupby(['visit_date']).sum().reset_index()



# Võtan iga kuupäeva ja lisa 1, sest muidu algab 0 indeksiga ehk nüüd saame kuupäevad nädalapäevadeks. 

dt = pd.to_datetime(df["visit_date"]).dt

weekdays = dt.weekday





df = pd.DataFrame({"week_day_nr" : weekdays,

                            "visitors" : df["visitors"]})



# Grupeerib nädalapäeva järgi ja summeerib külastajad. 

df = df.groupby(['week_day_nr']).sum().reset_index()

n = df

#df



#Graafiku tegemine

def day_fun(day_nr):

    days = ['E','T','K','N','R','L','P']

    return days[day_nr]



week_day_nrs = df.week_day_nr

week_days = week_day_nrs.apply(day_fun)



df = pd.DataFrame({

                   "week_day" : week_days,

                   "visitors" : df["visitors"]})

#df



#Kaotan ühe tulba, mida enam pole vaja.





#Graafik

df.set_index('week_day').plot(kind="bar", color="#dd55ff", position=0.05, width=0.95, rot=0);

n.sort_values(by=['visitors'])

sumV = genreTop(air_visit_loc, 90)

y_pos = np.arange(len(sumV.air_genre_name))

y_tick = list(sumV.visitors/1000)

y_name = list(sumV.air_genre_name)



plt.xlabel('1k Visitors per year')

plt.barh(y_pos, y_tick, align='center', alpha=0.9)

plt.yticks(y_pos, y_name);
