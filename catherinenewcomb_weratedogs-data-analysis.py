import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/weratedogs-tweepy-archive/twitter_dogs.csv')

images_df = pd.read_csv('../input/weratedogs-tweepy-archive/dog_predictions.csv')
df.head()
images_df.head()
df.info()
#There are 55 "None" values for rating_numerator, rating_denominator and subsequently rating_score.



df.replace(to_replace=["None"], value=np.nan, inplace=True)
import datetime as dt

df["timestamp"] = df.timestamp.str[:-6]
#When uploading the csv, all data types converted to string.



def to_float(column_name):

    return df[column_name].astype(float)



df["favorite_count"] = to_float("favorite_count")

df["retweet_count"] = to_float("retweet_count")

df["rating_score"] = to_float("rating_score")

df["timestamp"] = pd.to_datetime(df["timestamp"], format='%Y-%m-%d %H:%M:%S')



import plotly.express as px

fig = px.histogram(df, x="rating_score")

fig.show()
df_rating_analysis = df[df["rating_score"] <= 2]
fig = px.histogram(df_rating_analysis, x="rating_score", title="Distribution of Rating Scores")

fig.show()
fig = px.histogram(df_rating_analysis, x="favorite_count", title="Distribution of Number of Favorites")

fig.show()
fig = px.histogram(df_rating_analysis, x="retweet_count", title="Distribution of Number of Retweets")

fig.show()
fig = px.scatter(x=df_rating_analysis["timestamp"], y=df_rating_analysis["rating_score"])

fig.show()
fig = px.scatter(x=df_rating_analysis["favorite_count"], y=df_rating_analysis["rating_score"], trendline='ols')

fig.show()
#do linear and logistic fit to see what is best for 



import statsmodels.api as sm



df_rating_analysis['intercept'] = 1

lm = sm.OLS(df_rating_analysis["rating_score"], df_rating_analysis[["intercept", "favorite_count"]])

results = lm.fit()

results.summary()

df_rating_analysis["favorite_x_retweet"] = df_rating_analysis["favorite_count"] * df_rating_analysis["retweet_count"]

lm = sm.OLS(df_rating_analysis["rating_score"], df_rating_analysis[["intercept", "favorite_count", "retweet_count", "favorite_x_retweet"]])

results = lm.fit()

results.summary()

fig = px.scatter(x=df_rating_analysis["favorite_count"], y=df_rating_analysis["retweet_count"], trendline='ols', 

                 title = "Retweet Count vs. Favorite Count", labels={'x': "Number of Favorites", 'y':"Number of Retweets"})

fig.show()
df_dog_type = df.dropna(subset=["dog_type"])
types = pd.DataFrame(df_dog_type.dog_type.value_counts())

types
fig = px.bar(types, x=types.index, y="dog_type", labels={"index": "Dog Stage", "dog_type":"Stage Frequency"})

fig.show()
fig = px.scatter(df_dog_type, x="rating_score", y="favorite_count", trendline='ols', color="dog_type", 

                 title="Rating Score vs. Number of Favorites by Dog Type")

fig.show()
rs_avg = pd.DataFrame(df_dog_type.groupby("dog_type").rating_score.mean())

rs_avg

fig = px.bar(rs_avg, x=rs_avg.index, y='rating_score', color=rs_avg.index, title="Average Rating Score by Dog Stage")

fig.show()
df_no_type = df[df["dog_type"].isnull()]



print("Average Rating Score of Dogs with No Assigned Stage: ", df_no_type.rating_score.mean())
#dog_predictions = []

#images_df["most_likely_dog"] = images_df.apply(lambda x: images_df.prediction_1 if images_df.p1_dog == 1 else images_df.prediction_2)

images_df['most_likely_dog'] = ''

for i in range(images_df.shape[0]):

    if images_df['p1_dog'][i] == True:

        images_df["most_likely_dog"][i] = images_df["prediction_1"][i]

    elif images_df['p2_dog'][i] == True:

        images_df["most_likely_dog"][i] = images_df["prediction_2"][i]

    elif images_df['p3_dog'][i] == True:

        images_df["most_likely_dog"][i] = images_df["prediction_3"][i]

    else:

        images_df["most_likely_dog"][i] = "non-dog object"
images_df.most_likely_dog.value_counts()




fig = px.histogram(images_df, x="most_likely_dog", title="Distribution of Highest Confidence Dog Predictions")

fig.show()
images_df.shape
df_golden = images_df[(images_df["most_likely_dog"] == 'Labrador retriever') | 

                      (images_df["most_likely_dog"] == 'golden retriever')]

df_golden["tweet_id"] = df_golden["tweet_id"].astype(str)
df_golden = df_golden[["tweet_id", "most_likely_dog"]]
df_golden = df_golden.merge(df, on='tweet_id', how='inner')
df_golden
print("golden retriever predicted: ", len(df_golden.query("most_likely_dog == 'golden retriever'")))

print("labrador retriever predicted: ", len(df_golden.query("most_likely_dog == 'Labrador retriever'")))
df_golden['golden_descript'] = df_golden.text.str.contains('golden', case=False)

df_golden['lab_descript'] = df_golden.text.str.contains('lab', case=False)
df_golden.head(10)
print("Number of golden predictions confirmed by text: ", len(df_golden.query("(most_likely_dog == 'golden retriever') & (golden_descript == 1)")))

print("Number of labrador predictions confirmed by text: ", len(df_golden.query("(most_likely_dog == 'Labrador retriever') & (lab_descript == 1)")))
print("Number of tweets describing golden breed dog: ", sum(df_golden.golden_descript))

print("Number of tweets describing labrador breed dog: ", sum(df_golden.lab_descript))