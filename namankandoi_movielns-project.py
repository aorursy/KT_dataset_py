import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
filepath = "../input/simplilearn-projects/movies.dat"
movies = pd.read_table(filepath, sep="::", header=None, engine="python")
movies = movies.rename(columns={0:"MovieID",1:"Title", 2:"Genres"})
movies
filepath = "../input/simplilearn-projects/ratings.dat"
ratings = pd.read_table(filepath, sep="::", header=None, engine="python")
ratings = ratings.rename(columns={0:"UserID", 1:"MovieID", 2:"Rating", 3:"Timestamp"})
ratings
filepath = "../input/simplilearn-projects/users.dat"
users = pd.read_table(filepath, sep="::", header=None, engine="python")
users = users.rename(columns={0:"UserID", 1:"Gender", 2:"Age", 3:"Occupation", 4:"Zip-code"})
users
master = pd.merge(ratings, users, on="UserID")
master
master = pd.merge(master, movies, on="MovieID")
master
master_select = master[["MovieID", "Title", "UserID", "Age", "Gender", "Occupation", "Rating"]]
master_select
master_select["Age"].hist(bins=10)
toy = master_select[master_select["Title"] == "Toy Story (1995)"]
toy
toy["Rating"].hist(bins=10)
pd.crosstab(toy["Age"], toy["Rating"]).plot()
top = master_select.groupby("Title")["Rating"].agg("mean").sort_values(ascending=False).iloc[:25].sort_values()
top
top.plot(kind="barh", figsize=(24,12))
user_2696 = master_select[master_select["UserID"] == 2696]
user_2696 = user_2696[["Title", "Rating"]]
user_2696
user_2696.plot(x="Rating", y="Title", kind="scatter")
master_predict = master[["Age", "Gender", "Occupation", "Genres", "Rating"]]
master_predict
genres = master_predict["Genres"]
genres
genres.value_counts()
genres = genres.str.get_dummies()
genres
master_predict = pd.merge(master_predict, genres, left_index=True, right_index=True)
master_predict
master_predict = master_predict.drop("Genres", axis=1)
gender = master_predict["Gender"].str.get_dummies()
master_predict = pd.merge(master_predict, gender, left_index=True, right_index=True)
master_predict = master_predict.drop("Gender", axis=1)
master_predict
from sklearn.model_selection import train_test_split
x = master_predict.iloc[:, [0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]]
y = master_predict.iloc[:, 2]
x_train, x_test, y_train, y_test = train_test_split(x, y)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(x_train, y_train)
y_lin_predict = linreg.predict(x_test)
y_lin_predict = pd.Series(y_lin_predict)
y_test = y_test.reset_index().drop("index", axis=1)
predict_table = pd.concat([y_test, y_lin_predict], axis=1)
predict_table = predict_table.rename(columns={0:"LinearRegression"})
predict_table
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(x_train, y_train)
y_mnb_predict = mnb.predict(x_test)
y_mnb_predict = pd.Series(y_mnb_predict)
predict_table = pd.concat([predict_table, y_mnb_predict], axis=1)
predict_table = predict_table.rename(columns={0:"MultinomialNB"})
predict_table
from sklearn.metrics import mean_squared_error
lin_error = mean_squared_error(predict_table["Rating"], predict_table["LinearRegression"], squared=False)
lin_error
mnb_error = mean_squared_error(predict_table["Rating"], predict_table["MultinomialNB"], squared=False)
mnb_error
