# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

from pandas import DataFrame



performance = {"id" : [1, 2, 3, 4],

               "date" : ["19/12/2018", "20/12/2018", "21/12/2018", "22/12/2018"],

               "time" : [45, 50, 90 ,50],

               "km" : [6.0, 5.5, 6.0, 4.0],

               "rider_performance" : [3, 4, 4, 4],

               "horse_performance" : [4, 4, 5, 5],

               "avg_performance" : [3.5, 4.0, 4.5, 4.5]

               }



df = DataFrame(performance, columns= ['Id', 'date', 'time', 'km', 'rider_performance', 'horse_performance', 'avg_performance'])

df
time_graph = df.plot.bar(x="date", y="time", rot=0)

time_graph.set_xlabel("Date")

time_graph.set_ylabel("Time")
km_graph = df.plot.bar(x="date", y="km", rot=0)

km_graph.set_xlabel("Date")

km_graph.set_ylabel("Km")
rider_performance_graph = df.plot.bar(x="date", y="rider_performance", rot=0)

rider_performance_graph.set_xlabel("Date")

rider_performance_graph.set_ylabel("Rider perforamce")
horse_performance_graph = df.plot.bar(x="date", y="horse_performance", rot=0)

horse_performance_graph.set_xlabel("Date")

horse_performance_graph.set_ylabel("Horse perforamce")
avg_performance_graph = df.plot.bar(x="date", y="avg_performance", rot=0)

avg_performance_graph.set_xlabel("Date")

avg_performance_graph.set_ylabel("Average perforamce")
performance_df = pd.DataFrame({'Rider performance': df["rider_performance"],

                               'Horse performance': df["horse_performance"]})

perfrormance_graph_comparison1 = performance_df.plot.bar(rot=0)
performance_df2 = pd.DataFrame({'Rider performance': df["rider_performance"],

                                'Horse performance': df["horse_performance"],

                                'Average performance' : df["avg_performance"]})

perfrormance_graph_comparison2 = performance_df2.plot.bar(rot=0)
from sklearn.tree import DecisionTreeRegressor



y = df.horse_performance

features = ["time", "km", "rider_performance"]

X = df[features]



horse_performance_model = DecisionTreeRegressor(random_state=1)

horse_performance_model.fit(X, y)

predictions = horse_performance_model.predict(X)

print(predictions)



df
y2 = df.rider_performance

features2 = ["time", "km", "horse_performance"]

X2 = df[features2]



rider_performance_model = DecisionTreeRegressor(random_state=1)

rider_performance_model.fit(X2, y2)

predictions2 = rider_performance_model.predict(X2)

print(predictions2)



df
y3= df.km

features3 = ["time"]

X3 = df[features3]



km_model = DecisionTreeRegressor(random_state=1)

km_model.fit(X3, y3)

predictions3 = km_model.predict(X3)

print(predictions3)



df
train_X2, val_X2, train_y2, val_y2 = train_test_split(X3, y3, random_state=1)

val_predictions2 = km_model.predict(val_X2)

print(val_predictions2)



from sklearn.metrics import mean_absolute_error

val_mae = mean_absolute_error(val_predictions2, val_y2)

print(val_mae)



df
y4= df.time

features4 = ["km"]

X4 = df[features4]



time_model = DecisionTreeRegressor(random_state=1)

time_model.fit(X4, y4)

predictions4 = time_model.predict(X4)

print(predictions4)



df
from sklearn.model_selection import train_test_split



train_X, val_X, train_y, val_y = train_test_split(X4, y4, random_state=1)

val_predictions = time_model.predict(val_X)

print(val_predictions)



from sklearn.metrics import mean_absolute_error

val_mae = mean_absolute_error(val_predictions, val_y)

print(val_mae)
y5= df.km

features5 = ["time","rider_performance"]

X5 = df[features5]



km_model2 = DecisionTreeRegressor(random_state=1)

km_model2.fit(X5, y5)

predictions5 = km_model2.predict(X5)

print(predictions5)

print(predictions3)



df
y6= df.km

features6 = ["time","horse_performance"]

X6 = df[features6]



km_model3 = DecisionTreeRegressor(random_state=1)

km_model3.fit(X6, y6)

predictions6 = km_model3.predict(X6)

print(predictions6)

print(predictions3)



df
y7 = df.time

features7 = ["km", "horse_performance", "rider_performance"]

X7 = df[features7]



time_model2 = DecisionTreeRegressor(random_state=1)

time_model2.fit(X7, y7)

predictions7 = time_model2.predict(X7)

print("Second time predictions", predictions7)

print("First time predictions", predictions4)



df