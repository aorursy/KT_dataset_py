import numpy as np

import pandas as pd

import math

import matplotlib.pyplot as plt

import urllib

from google.cloud import bigquery

from kaggle_secrets import UserSecretsClient



bigquery_client = bigquery.Client(project="ny-data-263413")
### BigQuery ###

#

sql = """

      SELECT EXTRACT(YEAR FROM starttime) AS year,

             EXTRACT(MONTH FROM starttime) AS month,

             usertype AS user_type,

             COUNT(usertype) AS count

      FROM `bigquery-public-data.new_york.citibike_trips`

      GROUP BY year, month, user_type

      ORDER BY year, month, user_type

      """



query_job = bigquery_client.query(sql)

results = query_job.result()



#

data_users = pd.DataFrame()

for row in results :

    data_users = data_users.append([[int(row.year),

                                     int(row.month),

                                     row.user_type,

                                     row.count]])

data_users = data_users.rename(columns = {0 : "year",

                                          1 : "month",

                                          2 : "user_type",

                                          3 : "count"})



data_users = data_users.reset_index(drop = True)
### Plot Percentages ###

# Calculate

data_users["percent"] = (data_users.groupby(["year", "month"])["count"]

                                   .apply(lambda x : 100*x/sum(x))

                        )

data_users["month_num"] = (data_users["year"]-min(data_users["year"]))*12+data_users["month"]

data_users = data_users.sort_values("month_num")



# Plot totals

fig = plt.figure(figsize=(6,4), dpi = 150)

ax_total = fig.add_subplot(1,1,1)



ax_total.plot(data_users.loc[data_users["user_type"] == "Subscriber", "month_num"],

              data_users.loc[data_users["user_type"] == "Subscriber", "count"],

              linewidth = 3)

ax_total.plot(data_users.loc[data_users["user_type"] == "Customer", "month_num"],

              data_users.loc[data_users["user_type"] == "Customer", "count"],

              linewidth = 3)



data_total = data_users.groupby("month_num", as_index=False)["count"].aggregate("sum")

ax_total.plot(data_total["month_num"], data_total["count"], linewidth = 3)



# Aesthetics

ax_total.set_title("Total Number of Journeys By Month", fontsize = 16)

ax_total.set_ylabel("Number of Journeys (millions)", fontsize = 16)

ax_total.set_xlim(left = (7,45))

ax_total.set_ylim(bottom = (0,1.8*10**6))

ax_total.set_xticks(range(7,46, 3))

ax_total.set_xticklabels(["{}/{}".format(row["month"], row["year"])

                          for index, row in data_users.loc[data_users["user_type"]=="Subscriber"]

                                                      .iterrows()

                          if row["month"] in [1,4,7,10]

                         ],

                         rotation = -45,

                         fontsize = 12)

ax_total.set_yticks([i for i in range(0, int(1.8*10**6)+1, int(0.2*10**6))])

ax_total.set_yticklabels([i/10 for i in range(0, 19, 2)], fontsize = 12)

ax_total.legend(("Subscribers", "Customers", "Total"),

                fontsize=16, 

                ncol=3,

                bbox_to_anchor = (1.15,-0.3))

ax_total.spines["right"].set_visible(False)

ax_total.spines["top"].set_visible(False)



# Plot percentages

fig = plt.figure(figsize=(6,4), dpi = 125)



ax_percentage = fig.add_subplot(1,1,1)

ax_percentage.plot(data_users.loc[data_users["user_type"] == "Subscriber", "month_num"],

                   data_users.loc[data_users["user_type"] == "Subscriber", "percent"],

                   linewidth = 3)

ax_percentage.plot(data_users.loc[data_users["user_type"] == "Customer", "month_num"], 

                   data_users.loc[data_users["user_type"] == "Customer", "percent"],

                   linewidth = 3)

# Aesthetics

ax_percentage.set_title("Percentage of Total Journeys by User Type", fontsize = 16)

ax_percentage.set_ylabel("Percentage", fontsize = 16)

ax_percentage.set_xticks(range(7,46, 3))

ax_percentage.set_xticklabels(["{}/{}".format(row["month"], row["year"])

                               for index, row in data_users.loc[data_users["user_type"]=="Subscriber"]

                                                           .iterrows()

                               if row["month"] in [1,4,7,10]

                              ],

                              rotation = -45,

                              fontsize = 12)

ax_percentage.set_yticks([i for i in range(0,110,10)])

ax_percentage.set_yticklabels([i for i in range(0,110,10)], fontsize = 12)

ax_percentage.set_xlim(left = (7,45))

ax_percentage.set_ylim(bottom = (0,100))

ax_percentage.spines["right"].set_visible(False)

ax_percentage.spines["top"].set_visible(False)



plt.show()
### NY Data ###

# BigQuery

sql = """

        SELECT start_year AS year,

               start_season AS season,

               station_id,

               longitude,

               latitude,

               start_usertype AS user_type,

               start_table.start_count + end_table.end_count AS count

        FROM (

                (

                    SELECT EXTRACT(YEAR FROM starttime) AS start_year,

                           CASE WHEN EXTRACT(MONTH FROM starttime) <= 2

                                THEN 1

                                ELSE CEILING(ABS(EXTRACT(MONTH FROM starttime)-2)/3)

                                END AS start_season,

                           start_station_id,

                           usertype AS start_usertype,

                           COUNT(start_station_id) AS start_count,

                    FROM `bigquery-public-data.new_york.citibike_trips`

                    GROUP BY start_year, 

                             start_season, 

                             start_station_id,

                             start_usertype

                ) AS start_table

                FULL JOIN

                (

                    SELECT EXTRACT(YEAR FROM stoptime) AS stop_year,

                           CASE WHEN EXTRACT(MONTH FROM stoptime) <= 2

                                THEN 1

                                ELSE CEILING(ABS(EXTRACT(MONTH FROM stoptime)-2)/3)

                                END AS stop_season,

                           end_station_id,

                           usertype AS end_usertype,

                           COUNT(end_station_id) AS end_count

                    FROM `bigquery-public-data.new_york.citibike_trips`

                    GROUP BY stop_year,

                             stop_season,

                             end_station_id,

                             end_usertype

                ) AS end_table

                ON start_year=stop_year

                   AND start_season=stop_season

                   AND start_station_id=end_station_id

                   AND start_usertype=end_usertype

             )

             JOIN `bigquery-public-data.new_york.citibike_stations`

             ON start_station_id=station_id

        ORDER BY year, season, user_type DESC, count DESC

        LIMIT 100000

       """



query_job = bigquery_client.query(sql)

results = query_job.result()



#

data = pd.DataFrame()

for row in results :

    data = data.append([[int(row.year),

                         int(row.season),

                         row.station_id,

                         row.longitude,

                         row.latitude,

                         row.user_type,

                         row.count]])

data = data.rename(columns = {0 : "year",

                              1 : "season",

                              2 : "station_id",

                              3 : "longitude",

                              4 : "latitude",

                              5 : "user_type",

                              6 : "count"})



data["count"] = pd.to_numeric(data["count"])
### Maps Static API ###

#

user_secrets = UserSecretsClient()

maps_static_key = user_secrets.get_secret("maps_static")



# Convert longitude, latitude to pixel coordinates with zoom and vice versa

def lon_lat_to_pixel(lon, lat, zoom) :

    lon_rad = 2*math.pi*lon/360

    x = 256 * 2**zoom * (lon_rad + math.pi) / (2*math.pi)

    

    lat_rad = 2*math.pi*lat/360

    y = 256 * 2**zoom / (2 * math.pi) * (math.log(math.tan(math.pi/4 + lat_rad/2)))

    ##256 * 2**zoom / (2 * math.pi) * (math.pi - math.log(math.tan(math.pi/4 + lat_rad/2)))

    

    return [x, y]

def pixel_to_lon_lat(x, y, zoom) :

    lon = x / (256 * 2**zoom) - 180

    lat = 2 * math.atan(math.exp(math.pi - y / (256 * 2**zoom / (2 * math.pi)))) - math.pi / 2

    

    return [lon, lat]

data["pixel_x"] = data.apply(lambda row : lon_lat_to_pixel(row["longitude"],

                                                           row["latitude"],

                                                           12)[0],

                             axis = 1)

data["pixel_y"] = data.apply(lambda row : lon_lat_to_pixel(row["longitude"],

                                                           row["latitude"],

                                                           12)[1],

                             axis = 1)



#

lon = -73.987081

lat = 40.736539

zoom = 12

size_x, size_y = 600, 600

scale = 1

url = ("https://maps.googleapis.com/maps/api/staticmap?" ##implement above

       + "center=" + "{},{}".format(lat, lon) #lat,lon

       + "&zoom=" + str(zoom)

       + "&size=" + str(size_x) + "x" + str(size_y)

       + "&scale=" + str(scale)

       + "&style=feature:all%7Celement:labels%7Cvisibility:off"

       + "&key=" + maps_static_key)

urllib.request.urlretrieve(url, "map.png")



## &style=feature:road.highway%7Celement:geometry%7Cvisibility:simplified%7Ccolor:0xc280e9&style=feature:transit.line%7Cvisibility:simplified%7Ccolor:0xbababa&style=feature:road.highway%7Celement:labels.text.stroke%7Cvisibility:on%7Ccolor:0xb06eba&style=feature:road.highway%7Celement:labels.text.fill%7Cvisibility:on%7Ccolor:0xffffff

## url = "https://maps.googleapis.com/maps/api/staticmap?center=Berkeley,CA&zoom=14&size=400x400&key=" + maps_static_key
### Plot ###

#

img = plt.imread("map.png")

fig = plt.figure(figsize = (6,18), dpi = 150)

pixel_center = lon_lat_to_pixel(lon, lat, zoom)



#

num = 0

year = 2014

norm = max(data.loc[data["year"] == year]["count"])

axlist = []

for user_type in ["Subscriber", "Customer"] :

    season_name = {1 : "Spring", 2 : "Summer", 3 : "Autumn", 4 : "Winter"}

    for season in range(1,5) :

        ax = fig.add_subplot(4, 2, 1+2*(season-1)+num)

        axlist.append(ax)

        

        # Image and pixel coordinates

        ax.imshow(img, extent = [pixel_center[0] - size_x/2,

                                 pixel_center[0] + size_x/2,

                                 pixel_center[1] - size_y/2,

                                 pixel_center[1] + size_y/2])

        

        # Plot

        scatter = ax.scatter(data.loc[(data["season"] == season)

                                      & (data["year"] == year)

                                      & (data["user_type"] == user_type)]

                                     ["pixel_x"],

                             data.loc[(data["season"] == season)

                                      & (data["year"] == year)

                                      & (data["user_type"] == user_type)]

                                     ["pixel_y"],

                             1,

                             c = data.loc[(data["season"] == season)

                                          & (data["year"] == year)

                                          & (data["user_type"] == user_type)]

                                         ["count"]

                                     .apply(math.log, args = (10,)),

                             vmin = 0,

                             vmax = math.log(norm, 10),

                             cmap = "inferno"

                            )

        

        # Appearance

        if season == 1 :

            ax.set_title(user_type+"s", fontsize = 16)                

        if user_type == "Subscriber" :

            ax.set_ylabel(season_name[season], fontsize = 16)

        ax.tick_params(bottom=False, left=False, top=False, right=False)

        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

        for key, spine in ax.spines.items() :

            spine.set_visible(False)

    

    num += 1



ticks = [0] + [math.log(j,10) for j in range(1,10)]

ticklabels = [0] + ["" for j in range(1,10)]

for i in range(1,5) :

    ticks = ticks + [math.log(j * 10**i,10) for j in range(1,10)]

    ticklabels = ticklabels + [10**i

                               if j == 1

                               else ""

                               for j in range(1,10)]



cbar = fig.colorbar(scatter,

                    ax=axlist,

                    ticks=ticks,

                    orientation="horizontal",

                    anchor=(0.5,2))

cbar.ax.set_xticklabels(ticklabels, rotation = -45)



plt.show()
### BigQuery ###

sql = """

        SELECT start_hour AS hour,

               start_station_id AS station_id,

               longitude,

               latitude,

               start_user_type AS user_type,

               AVG(start_count+end_count) AS average

        FROM (

                 (

                    SELECT DATE(starttime) AS start_date,

                           EXTRACT(HOUR FROM starttime) AS start_hour,

                           start_station_id,

                           usertype AS start_user_type,

                           COUNT(*) AS start_count

                    FROM `bigquery-public-data.new_york.citibike_trips`

                    GROUP BY start_date, start_hour, start_station_id, start_user_type

                  ) AS start_table

                  FULL JOIN

                  (

                    SELECT DATE(starttime) AS stop_date,

                           EXTRACT(HOUR FROM stoptime) AS stop_hour,

                           end_station_id,

                           usertype AS end_user_type,

                           COUNT(*) AS end_count

                    FROM `bigquery-public-data.new_york.citibike_trips`

                    GROUP BY stop_date, stop_hour, end_station_id, end_user_type

                  ) AS end_table

                  ON start_date=stop_date

                     AND start_hour=stop_hour

                     AND start_station_id=end_station_id

                     AND start_user_type=end_user_type

             )

             JOIN `bigquery-public-data.new_york.citibike_stations`

             ON start_station_id=station_id

        GROUP BY hour, station_id, longitude, latitude, user_type

        ORDER BY hour, user_type DESC, average DESC

        LIMIT 400000

      """



query_job = bigquery_client.query(sql)

results = query_job.result()



#

data_hours = pd.DataFrame()

for row in results :

    data_hours = data_hours.append([[int(row.hour),

                                     row.station_id,

                                     row.longitude,

                                     row.latitude,

                                     row.user_type,

                                     row.average]])

data_hours = data_hours.rename(columns = {0 : "hour",

                                          1 : "station_id",

                                          2 : "longitude",

                                          3 : "latitude",

                                          4 : "user_type",

                                          5 : "average"})



data_hours = data_hours.reset_index(drop = True)
### Plot ###

fig = plt.figure(figsize = (10,5), dpi = 150)

ax = fig.add_subplot(1,1,1)



ax.plot(data_hours.loc[data_hours["user_type"] == "Subscriber","hour"]

                  .unique(),

        data_hours.loc[data_hours["user_type"] == "Subscriber"]

                .groupby(["hour"], as_index=False)

                ["average"]

                .aggregate(["sum"]),

        linewidth = 3

       )

ax.plot(data_hours.loc[data_hours["user_type"] == "Customer","hour"]

                  .unique(), 

        data_hours.loc[data_hours["user_type"] == "Customer"]

                .groupby(["hour"], as_index=False)

                ["average"]

                .aggregate(["sum"]),

        linewidth = 3

       )



ax.set_title("Average Total Journeys in a Day by Hour", fontsize = 25)

ax.set_ylabel("Average", fontsize = 25)

ax.set_xlabel("Hour of the Day", fontsize = 25)

ax.set_xlim(left = (0,24))

ax.set_xticks(range(0,24,3))

ax.set_xticklabels(labels = range(0,24,3), fontsize = 20)

ax.set_yticks(range(0,8001,1000))

ax.set_yticklabels(labels = range(0,8001,1000), fontsize = 20)

ax.legend(bbox_to_anchor=(1, 0.7), fontsize = 20, labels = ("Subscribers", "Customers"))

ax.spines["right"].set_visible(False)

ax.spines["top"].set_visible(False)



plt.show()
### Map ###

data_hours["pixel_x"] = data_hours.apply(lambda row : lon_lat_to_pixel(row["longitude"],

                                                                       row["latitude"],

                                                                       12)[0],

                             axis = 1)

data_hours["pixel_y"] = data_hours.apply(lambda row : lon_lat_to_pixel(row["longitude"],

                                                                       row["latitude"],

                                                                       12)[1],

                             axis = 1)



### Plot ###

from matplotlib.animation import FuncAnimation



data_hours = data_hours.loc[~np.isnan(data_hours["average"])]



#

user_type = "Subscriber"



#

fig = plt.figure(dpi = 150)

ax = fig.add_subplot(1, 1, 1)



ax.imshow(img, extent = [pixel_center[0] - size_x/2,

                         pixel_center[0] + size_x/2,

                         pixel_center[1] - size_y/2,

                         pixel_center[1] + size_y/2])





norm = max(data_hours.loc[data_hours["user_type"] == user_type]["average"])



scatter = ax.scatter([], [], 2, c=[], vmin=0, vmax=math.log(100, 10), cmap="inferno")

ann = ax.annotate(s = "Hour "+str(0),

                  xy = (pixel_center[0] - size_x/2, pixel_center[1] - size_y/2 - 30),

                  annotation_clip=False)





ax.tick_params(bottom=False, left=False, top=False, right=False)

plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

for key, spine in ax.spines.items() :

    spine.set_visible(False)



#

ax.set_title("Number of Uses By "+user_type+"s", fontsize = 16)

ticks = [0] + [math.log(j,10) for j in range(1,10)]

ticklabels = [0] + ["" for j in range(1,10)]

for i in range(1,3) :

    ticks = ticks + [math.log(j * 10**i,10) for j in range(1,10)]

    ticklabels = ticklabels + [10**i

                               if j == 1

                               else ""

                               for j in range(1,10)]

cbar = fig.colorbar(scatter,

                    ax=ax,

                    ticks=ticks)

cbar.ax.set_yticklabels(ticklabels)



#

def animate(i) :        

    x = (data_hours.loc[(data_hours["user_type"] == user_type)

                        & (data_hours["hour"] == i)]

                       ["pixel_x"]

         )

    y = (data_hours.loc[(data_hours["user_type"] == user_type)

                        & (data_hours["hour"] == i)]

                       ["pixel_y"]

        )

    

    scatter.set_offsets(np.c_[x,y])    

    scatter.set_sizes([2]*len(x))

    

    scatter.set_array(data_hours.loc[(data_hours["hour"] == i)

                                     & (data_hours["user_type"] == user_type)]

                                    ["average"]

                          .apply(math.log, args = (10,))

                     )

    

    ann.set_text(s = "Hour "+str(i))



anim = FuncAnimation(fig, animate, interval=100, frames=24)

plt.close()



#

from matplotlib import rc

from IPython.display import HTML

rc('animation', html='jshtml')



HTML(anim.to_jshtml())
#

user_type = "Customer"



#

fig = plt.figure(dpi = 150)

ax = fig.add_subplot(1, 1, 1)



ax.imshow(img, extent = [pixel_center[0] - size_x/2,

                         pixel_center[0] + size_x/2,

                         pixel_center[1] - size_y/2,

                         pixel_center[1] + size_y/2])





norm = max(data_hours.loc[data_hours["user_type"] == user_type]["average"])



scatter = ax.scatter([], [], 2, c=[], vmin=0, vmax=math.log(30, 10), cmap="inferno")

ann = ax.annotate(s = "Hour "+str(0),

                  xy = (pixel_center[0] - size_x/2, pixel_center[1] - size_y/2 - 30),

                  annotation_clip=False)



ax.tick_params(bottom=False, left=False, top=False, right=False)

plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

for key, spine in ax.spines.items() :

    spine.set_visible(False)



#

ax.set_title("Number of Uses By "+user_type+"s", fontsize = 16)

ticks = [0] + [math.log(j,10) for j in range(1,10)]

ticklabels = [0] + ["" for j in range(1,10)]

for i in range(1,3) :

    ticks = ticks + [math.log(j * 10**i,10) for j in range(1,10)]

    ticklabels = ticklabels + [j*10**i

                               if j == 1 or (j == 3 and i == 1)

                               else ""

                               for j in range(1,10)]

cbar = fig.colorbar(scatter,

                    ax=ax,

                    ticks=ticks)

cbar.ax.set_yticklabels(ticklabels)



#

def animate(i) :        

    x = (data_hours.loc[(data_hours["user_type"] == user_type)

                        & (data_hours["hour"] == i)]

                       ["pixel_x"]

        )

    y = (data_hours.loc[(data_hours["user_type"] == user_type)

                        & (data_hours["hour"] == i)]

                       ["pixel_y"]

        )

    

    scatter.set_offsets(np.c_[x,y])

    scatter.set_sizes([2]*len(x))



    scatter.set_array(data_hours.loc[(data_hours["hour"] == i)

                                     & (data_hours["user_type"] == user_type)]

                                    ["average"]

                          .apply(math.log, args = (10,))

                     )

    

    ann.set_text(s = "Hour "+str(i))



anim = FuncAnimation(fig, animate, interval=100, frames=24)

plt.close()



#

HTML(anim.to_jshtml())