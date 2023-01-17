# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as dt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
res_level = pd.read_csv("../input/chennai_reservoir_levels.csv")

res_rfall = pd.read_csv("../input/chennai_reservoir_rainfall.csv")
res_level.head()
res_rfall.head()
res_level.describe()
res_rfall.describe()
# Rename the columns in both the data sets.



res_level.columns = res_level.columns.str.lower() + "_level"

res_rfall.columns = res_rfall.columns.str.lower() + "_rfall"
# Join water level and rainfall data sets to form a single dataset.



res_master_df = res_level.merge(res_rfall, left_on = "date_level", right_on = "date_rfall")

res_master_df = res_master_df.drop(columns = ["date_rfall"], errors = "ignore")

res_master_df = res_master_df.rename(index = str, columns = {"date_level":"recorded_date"})

res_master_df.head()
# Convert object to date data type.



res_master_df["recorded_date"] = pd.to_datetime(res_master_df["recorded_date"], format = "%d-%m-%Y")
# Extract year and month from recorded_date field.



res_master_df["year"] = res_master_df["recorded_date"].dt.year

res_master_df["month"] = res_master_df["recorded_date"].dt.month
res_master_df.head()
res_master_df["season"] = np.select([res_master_df["month"].isin([12, 1, 2]), 

                                     res_master_df["month"].isin([3, 4, 5]),

                                     res_master_df["month"].isin([6, 7, 8, 9]),

                                     res_master_df["month"].isin([10, 11])], 

                                    ["winter", "summer", "monsoon", "autumn"])
# Group the data by year and season on each of the water level columns and merge all these data frames 

# into a single data frame for further analysis.



df_1 = pd.DataFrame(res_master_df.groupby(["year", "season"])[["poondi_level", "poondi_rfall"]].agg(np.mean)).round(2)

df_2 = pd.DataFrame(res_master_df.groupby(["year", "season"])[["cholavaram_level", "cholavaram_rfall"]].agg(np.mean)).round(2)

df_3 = pd.DataFrame(res_master_df.groupby(["year", "season"])[["redhills_level", "redhills_rfall"]].agg(np.mean)).round(2)

df_4 = pd.DataFrame(res_master_df.groupby(["year", "season"])[["chembarambakkam_level", "chembarambakkam_rfall"]].agg(np.mean)).round(2)

df_1.reset_index(inplace = True)

df_2.reset_index(inplace = True)

df_3.reset_index(inplace = True)

df_4.reset_index(inplace = True)



df_1 = df_1.merge(df_2, left_on = ["year", "season"], right_on = ["year", "season"])

df_1 = df_1.merge(df_3, left_on = ["year", "season"], right_on = ["year", "season"])

df_1 = df_1.merge(df_4, left_on = ["year", "season"], right_on = ["year", "season"])



year_season_group_df = df_1



del (df_1, df_2, df_3, df_4)
def plot_water_level (ip_df, ip_reservoir):

    

    fig = plt.figure(figsize = (12, 9))

    

    df = ip_df[['year', 'season', ip_reservoir + '_level']]

    df_1 = df.groupby('year')[ip_reservoir + '_level'].sum()

    df_1 = df_1.reset_index()

    df = df.merge(df_1, right_on = 'year', left_on = 'year')

    df = df.rename(index = str, columns = {ip_reservoir + '_level_x':'water_level', ip_reservoir + '_level_y':'total_water_level'})

    df['perc'] = round(df['water_level'] * 100 / df['total_water_level'], 2)

    df['water_level'] = df['perc'].copy()

    df = df.drop(columns = ['total_water_level', 'perc'], axis = 1)



    def_water_level_df = pd.DataFrame({'year':df["year"].unique().astype(str).tolist(), 

                                       'water_level':list(map(int, "0" * df["year"].nunique()))})



    water_level_df = pd.DataFrame({'year':df["year"].unique().astype(str).tolist(),

                                   'bar_bottom':list(map(int, "0" * df["year"].nunique()))

                                  })



    water_level_rec = pd.DataFrame({'season':[], 'year':[], 'water_level':[]})



    tmp_plot = plt.bar([], [], [])

    plot_legend = {"color":[], "plot":[]}



    for season in (['winter', 'summer', 'monsoon', 'autumn']):

        water_level_rec = df.loc[df["season"] == season, ['year', 'water_level']].copy()

        water_level_rec['year'] = water_level_rec['year'].astype(str)



        water_level_rec = def_water_level_df.merge(water_level_rec, 

                                                   how = 'left', 

                                                   right_on = 'year', 

                                                   left_on = 'year')



        water_level_rec = water_level_rec.drop(columns = 'water_level_x', axis = 1)

        water_level_rec = water_level_rec.rename(index = str, columns = {'water_level_y':'water_level'})

        water_level_rec['water_level'] = water_level_rec['water_level'].fillna(0)



        x = water_level_rec['year'].tolist()

        y = water_level_rec['water_level'].tolist()    

        bar_bottom = water_level_df['bar_bottom']



        tmp_plot = plt.bar(x, y, bottom = bar_bottom)



        for i in (water_level_df.itertuples(index = True, name = 'Pandas')):



            i_iter_year = getattr(i, 'year')



            for j in (water_level_rec.itertuples(index = True, name = 'Pandas')):

                j_iter_year = getattr(j, 'year')

                j_water_level = getattr(j, 'water_level')



                if (i_iter_year == j_iter_year):

                    water_level_df.loc[water_level_df['year'] == j_iter_year, 'bar_bottom'] += j_water_level

                    break



        for i in (np.arange(len(x))):



            if (y[i] == 0):

                continue



            y_coord = round(water_level_df.loc[i, 'bar_bottom'] - y[i] / 2)



            x_coord = x[i]



            plt.text(x_coord, y_coord, str(y[i]) + '%', ha = 'center', fontsize = 8, color = 'white')



        plot_legend["color"].append(season.title())

        plot_legend["plot"].append(tmp_plot[0])



    plt.xlabel("Year")

    plt.ylabel('Water level (Average per year per season - Proportionately)')

    fig.suptitle("Reservoir : " + ip_reservoir.title() + "\nAverage water level Year-wise Season-wise")

    plt.legend(plot_legend["plot"], plot_legend["color"])

    plt.show()
plot_water_level (year_season_group_df, 'poondi')
year_season_group_df.loc[year_season_group_df['year']==2019, ]
plot_water_level (year_season_group_df, 'cholavaram')
plot_water_level (year_season_group_df, 'redhills')
plot_water_level (year_season_group_df, 'chembarambakkam')
def plot_rain_fall (ip_df, ip_reservoir):

    

    fig = plt.figure(figsize = (12, 9))

    

    df = ip_df[['year', 'season', ip_reservoir + '_rfall']]

    df_1 = df.groupby('year')[ip_reservoir + '_rfall'].sum()

    df_1 = df_1.reset_index()

    df = df.merge(df_1, right_on = 'year', left_on = 'year')

    df = df.rename(index = str, columns = {ip_reservoir + '_rfall_x':'rain_fall', ip_reservoir + '_rfall_y':'total_rain_fall'})

    df['perc'] = round(df['rain_fall'] * 100 / df['total_rain_fall'], 2)

    df['rain_fall'] = df['perc'].copy()

    df = df.drop(columns = ['total_rain_fall', 'perc'], axis = 1)



    def_rain_fall_df = pd.DataFrame({'year':df["year"].unique().astype(str).tolist(), 

                                     'rain_fall':list(map(int, "0" * df["year"].nunique()))})



    rain_fall_df = pd.DataFrame({'year':df["year"].unique().astype(str).tolist(),

                                 'bar_bottom':list(map(int, "0" * df["year"].nunique()))

                                 })



    rain_fall_rec = pd.DataFrame({'season':[], 'year':[], 'rain_fall':[]})



    tmp_plot = plt.bar([], [], [])

    plot_legend = {"color":[], "plot":[]}



    for season in (['winter', 'summer', 'monsoon', 'autumn']):

        rain_fall_rec = df.loc[df["season"] == season, ['year', 'rain_fall']].copy()

        rain_fall_rec['year'] = rain_fall_rec['year'].astype(str)



        rain_fall_rec = def_rain_fall_df.merge(rain_fall_rec, 

                                               how = 'left', 

                                               right_on = 'year', 

                                               left_on = 'year')



        rain_fall_rec = rain_fall_rec.drop(columns = 'rain_fall_x', axis = 1)

        rain_fall_rec = rain_fall_rec.rename(index = str, columns = {'rain_fall_y':'rain_fall'})

        rain_fall_rec['rain_fall'] = rain_fall_rec['rain_fall'].fillna(0)



        x = rain_fall_rec['year'].tolist()

        y = rain_fall_rec['rain_fall'].tolist()    

        bar_bottom = rain_fall_df['bar_bottom']



        tmp_plot = plt.bar(x, y, bottom = bar_bottom)



        for i in (rain_fall_df.itertuples(index = True, name = 'Pandas')):



            i_iter_year = getattr(i, 'year')



            for j in (rain_fall_rec.itertuples(index = True, name = 'Pandas')):

                j_iter_year = getattr(j, 'year')

                j_rain_fall = getattr(j, 'rain_fall')



                if (i_iter_year == j_iter_year):

                    rain_fall_df.loc[rain_fall_df['year'] == j_iter_year, 'bar_bottom'] += j_rain_fall

                    break



        for i in (np.arange(len(x))):



            if (y[i] == 0):

                continue



            y_coord = round(rain_fall_df.loc[i, 'bar_bottom'] - y[i] / 2)



            x_coord = x[i]



            plt.text(x_coord, y_coord, str(y[i]) + '%', ha = 'center', fontsize = 8, color = 'white')



        plot_legend["color"].append(season.title())

        plot_legend["plot"].append(tmp_plot[0])



    plt.ylabel('Rain fall (Average per year per season - Proportionately)')

    fig.suptitle("Reservoir : " + ip_reservoir.title() + "\nAverage rain fall Year-wise Season-wise")

    plt.legend(plot_legend["plot"], plot_legend["color"])

    plt.show()
plot_rain_fall (year_season_group_df, 'poondi')
plot_rain_fall (year_season_group_df, 'cholavaram')
plot_rain_fall (year_season_group_df, 'redhills')
plot_rain_fall (year_season_group_df, 'chembarambakkam')
# Water reservoir which is filled to the maximum capacity (as per data).



year_season_group_df[['poondi_level', 'cholavaram_level', 'redhills_level', 'chembarambakkam_level']].max()