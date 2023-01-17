# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/netflix-shows/netflix_titles.csv")

df.head()
#df.fillna("Unknown", inplace=True)

df_known_date_added = df.dropna(subset=["date_added"])

total_movies = len(df[:][df["type"] == "Movie"])

total_tv_show = len(df[:][df["type"] == "TV Show"])

print("Total Movies On Netflix : ", total_movies)

print("Total TV Shows On Netflix : ", total_tv_show)
df = df.replace({'duration': ' min'},'', regex=True)

df["duration"] = df["duration"][df["type"] == "Movie"].astype(int)



print("The maximum duration of moive is ",int(df["duration"].max()), "min.")

print("The averange duration of moive is ",int(df["duration"].mean()), "min.")

print("The minimum duration of moive is ",int(df["duration"].min()), "min.")
def get_dataframe_for_year(df,year):

    

    for index,value in enumerate(df.date_added):

        if year in str(value):  

            df.date_added[index] = int(year)

            

    df = df[df.date_added == int(year)]

       

    return df
df_2010 = get_dataframe_for_year(df,'2010')

df_2011 = get_dataframe_for_year(df,'2011')

df_2012 = get_dataframe_for_year(df,'2012')

df_2013 = get_dataframe_for_year(df,'2013')

df_2014 = get_dataframe_for_year(df,'2014')

df_2015 = get_dataframe_for_year(df,'2015')

df_2016 = get_dataframe_for_year(df,'2016')

df_2017 = get_dataframe_for_year(df,'2017')

df_2018 = get_dataframe_for_year(df,'2018')

df_2019 = get_dataframe_for_year(df,'2019')
def total_data(df_year):

    total_moives = len(df_year[:][df_year["type"] == "Movie"])

    total_tv_shows = len(df_year[:][df_year["type"] == "TV Show"])

    total_added_movies_shows = total_moives + total_tv_shows

    

    return total_moives, total_tv_shows, total_added_movies_shows
total_moives_2010, total_tv_shows_2010, total_added_movies_shows_2010 = total_data(df_2010)

total_moives_2011, total_tv_shows_2011, total_added_movies_shows_2011 = total_data(df_2011)

total_moives_2012, total_tv_shows_2012, total_added_movies_shows_2012 = total_data(df_2012)

total_moives_2013, total_tv_shows_2013, total_added_movies_shows_2013 = total_data(df_2013)

total_moives_2014, total_tv_shows_2014, total_added_movies_shows_2014 = total_data(df_2014)

total_moives_2015, total_tv_shows_2015, total_added_movies_shows_2015 = total_data(df_2015)

total_moives_2016, total_tv_shows_2016, total_added_movies_shows_2016 = total_data(df_2016)

total_moives_2017, total_tv_shows_2017, total_added_movies_shows_2017 = total_data(df_2017)

total_moives_2018, total_tv_shows_2018, total_added_movies_shows_2018 = total_data(df_2018)

total_moives_2019, total_tv_shows_2019, total_added_movies_shows_2019 = total_data(df_2019)

years = [year for year in range(2010,2020)]
filtered_data = {

    "year" : [year for year in range(2010,2020)], 

    "added moives" : [total_moives_2010, total_moives_2011, total_moives_2012, total_moives_2013, total_moives_2014,

                     total_moives_2015, total_moives_2016, total_moives_2017, total_moives_2018, total_moives_2019], 

    "added TV shows" : [total_tv_shows_2010, total_tv_shows_2011, total_tv_shows_2012, total_tv_shows_2013, total_tv_shows_2014,

                       total_tv_shows_2015, total_tv_shows_2016, total_tv_shows_2017, total_tv_shows_2018, total_tv_shows_2019], 

    "total" : [total_added_movies_shows_2010, total_added_movies_shows_2011, total_added_movies_shows_2012, total_added_movies_shows_2013,

              total_added_movies_shows_2014, total_added_movies_shows_2015, total_added_movies_shows_2016, total_added_movies_shows_2017,

              total_added_movies_shows_2018, total_added_movies_shows_2019]

}

filtered_dataframe = pd.DataFrame(filtered_data)

filtered_dataframe
X = np.array([

    [2010],

    [2011],

    [2012],

    [2013],

    [2014],

    [2015],

    [2016],

    [2017],

    [2018],

    [2019],

])
y = np.array(filtered_dataframe["added moives"]).flatten()

y.reshape(1,-1)

plt.plot(X.flatten(),y)

plt.xlabel("years")

plt.ylabel("Added movies")

plt.title("Netflix dataframe for movies")
y = np.array(filtered_dataframe["added TV shows"]).flatten()

plt.plot(X.flatten(),y)

plt.xlabel("years")

plt.ylabel("Added TV Shows")

plt.title("Netflix dataframe for TV Shows")
reg = LinearRegression()

X = np.array([

    [2015],

    [2016],

    [2017],

    [2018],

    [2019],

])

X_predict = np.array([[2020]])

y = np.array(filtered_dataframe["added moives"][filtered_dataframe["year"]>2014]).flatten()

y.reshape(1,-1)

reg = reg.fit(X,y)

predict = reg.predict(X_predict)

print("The prediction for number of movies could is added : ", int(predict))
y = np.array(filtered_dataframe["added TV shows"][filtered_dataframe["year"]>2014]).flatten()

y.reshape(1,-1)

reg = reg.fit(X,y)

predict = reg.predict(X_predict)

print("The prediction for number of TV Shows could is added : ", int(predict))