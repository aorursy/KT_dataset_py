# import packages

import csv

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import style; style.use('ggplot')

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

import time

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier
#read data

train_data = pd.read_csv("../input/kuc-hackathon-winter-2018/drugsComTrain_raw.csv", sep=',')



train_data.head()

train_data['review'][0]
#read data

test_data = pd.read_csv("../input/kuc-hackathon-winter-2018/drugsComTest_raw.csv", sep=',')



test_data.head()

#check if features of train&test datasets are same or not

list(train_data) == list(test_data)
#check the sizes of train&test datasets(train-test-split: 75:25)

train_data.values.shape[0], test_data.values.shape[0], (train_data.values.shape[0] / test_data.values.shape[0])
#unique values of "condition" col

train_data.loc[:,"condition"].unique().size, test_data.loc[:, "condition"].unique().size
#unique values of "drugName" col

train_data.loc[:, "drugName"].unique().size, test_data.loc[:, "drugName"].unique().size
conditions = train_data["condition"].value_counts().sort_values(ascending=False)

conditions[:10]
top_10_drugs=train_data["drugName"].value_counts().sort_values(ascending=False)[:10]



top_10_drugs_df=pd.DataFrame(zip(top_10_drugs.index, top_10_drugs), columns = ["drugName", "count"])

import seaborn as sns 



#counts of top 10 drugs

sns.set(style="whitegrid")

plt.figure(figsize=(10, 5))

ax = sns.barplot(x="drugName", y="count", data=top_10_drugs_df, palette=sns.color_palette("cubehelix", 10))

plt.xticks(rotation=90)

plt.title("Top-10 drug counts", {"fontname":"fantasy", "fontweight":"bold", "fontsize":"medium"})

plt.ylabel("count", {"fontname": "serif", "fontweight":"bold"})

plt.xlabel("drug name", {"fontname": "serif", "fontweight":"bold"})
#ratings of top-10 drugs 

rating_top_10_df=train_data.loc[train_data["drugName"].isin(top_10_drugs.index), :]

rating_top_10_df
#ratings(average-variance) of top-10 drugs

sns.set(style="whitegrid")

plt.figure(figsize=(10, 5))

ax = sns.barplot(x="drugName", y="rating", data=rating_top_10_df, palette=sns.color_palette("RdBu_r", 10))

plt.xticks(rotation=90)

plt.title("Average Rating vs Drug Name", {"fontname":"fantasy", "fontweight":"bold", "fontsize":"medium"})

plt.ylabel("average rating&variance", {"fontname": "serif", "fontweight":"bold"})

plt.xlabel("drug name", {"fontname": "serif", "fontweight":"bold"})
top_10_condition=train_data["condition"].value_counts().sort_values(ascending=False)[:10]



top_10_condition_df=pd.DataFrame(zip(top_10_condition.index, top_10_condition), columns = ["condition", "count"])

top_10_condition_df
#counts of top-10 conditions



sns.set(style="whitegrid")

plt.figure(figsize=(10, 5))

ax = sns.barplot(x="condition", y="count", data=top_10_condition_df, palette=sns.cubehelix_palette(10, start=.5, rot=-.20))

plt.xticks(rotation=90)

plt.title("Top-10 conditions", {"fontname":"fantasy", "fontweight":"bold", "fontsize":"medium"})

plt.ylabel("count", {"fontname": "serif", "fontweight":"bold"})

plt.xlabel("condition", {"fontname": "serif", "fontweight":"bold"})
#count of ratings 



sns.set(style="whitegrid")

plt.figure(figsize=(10, 5))

ax = sns.countplot(x="rating", data=train_data, palette=sns.cubehelix_palette(10, start=.5, rot=-.20))

plt.title("Rating counts", {"fontname":"fantasy", "fontweight":"bold", "fontsize":"medium"})

plt.ylabel("count", {"fontname": "serif", "fontweight":"bold"})

plt.xlabel("rating", {"fontname": "serif", "fontweight":"bold"})
#conditions of top-1000 review(according to the "usefulCount" values)



top_100_reviews=train_data["usefulCount"].sort_values(ascending=False)[:100]

top_100_reviews_df=train_data.loc[top_100_reviews.index, :]

print(top_100_reviews_df["usefulCount"].max())

print(top_100_reviews_df["usefulCount"].min())
#conditions of most useful(popular) reviews 



sns.set(style="whitegrid")

plt.figure(figsize=(10, 5))

ax = sns.countplot(x="condition", data=top_100_reviews_df, palette=sns.color_palette("RdBu_r", 10))

plt.xticks(rotation=90)

plt.title("Conditions of top-100 reviews", {"fontname":"fantasy", "fontweight":"bold", "fontsize":"medium"})

plt.ylabel("count", {"fontname": "serif", "fontweight":"bold"})

plt.xlabel("condition", {"fontname": "serif", "fontweight":"bold"})
#create a new feature review_length

train_data["review_length"]=train_data["review"].str.len()



test_data["review_length"]=test_data["review"].str.len()
#density of the review length

plt.figure(figsize=(12.8,6))

sns.distplot(train_data['review_length']).set_title('review length distribution')
quantile_95 = train_data['review_length'].quantile(0.95)

df_95 = train_data[train_data['review_length'] < quantile_95]



plt.figure(figsize=(12.8,6))

sns.distplot(df_95['review_length']).set_title('review length distribution')
quantile_95
df_more10k = train_data[train_data['review_length'] > 10000]

print(len(df_more10k))

print("-------")

print(df_more10k["review"].values)
#box-plot



top_10=train_data["condition"].value_counts().sort_values(ascending=False)[:10]

top_10_df=train_data.loc[train_data["condition"].isin(top_10.index), :]



plt.figure(figsize=(16,6))

sns.boxplot(data=top_10_df, x='condition', y='review_length', width=.5)

plt.title("Box-plot", {"fontname":"fantasy", "fontweight":"bold", "fontsize":"medium"})

plt.ylabel("review length", {"fontname": "serif", "fontweight":"bold"})

plt.xlabel("condition", {"fontname": "serif", "fontweight":"bold"})
#box-plot



top_10_version2=df_95["condition"].value_counts().sort_values(ascending=False)[:10]

top_10_version2_df=df_95.loc[df_95["condition"].isin(top_10.index), :]



plt.figure(figsize=(16,6))

sns.boxplot(data=top_10_version2_df, x='condition', y='review_length', width=.5)

plt.title("Box-plot of 0.95 quantile", {"fontname":"fantasy", "fontweight":"bold", "fontsize":"medium"})

plt.ylabel("review length", {"fontname": "serif", "fontweight":"bold"})

plt.xlabel("condition", {"fontname": "serif", "fontweight":"bold"})
top_10_condition=train_data["condition"].value_counts().sort_values(ascending=False)[:10]

top_10_condition_df=train_data.loc[train_data["condition"].isin(top_10_condition.index), :]

len(top_10_condition_df)
import pickle



with open('top_10_train.pickle', 'wb') as output:

    pickle.dump(top_10_condition_df, output)
#pickle for test_data(selecting top-10 conditions)

top_10_condition_test=test_data["condition"].value_counts().sort_values(ascending=False)[:10]

top_10_condition_test_df=test_data.loc[test_data["condition"].isin(top_10_condition_test.index), :]

len(top_10_condition_test_df)

import pickle

with open('top_10_test.pickle', 'wb') as output:

    pickle.dump(top_10_condition_test_df, output)