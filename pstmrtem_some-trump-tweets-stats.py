import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sn

import datetime



%pylab inline

pylab.rcParams['figure.figsize'] = (10, 7)



df = pd.read_csv("../input/trump-tweets/trumptweets.csv", sep=",", engine="python")

df['date'] = pd.to_datetime(df['date']) # So pandas understands it's a date



number_of_days = (df["date"].max() - df["date"].min()).days

print("Amount of tweets: {}".format(len(df)))

print("Published in {} days, which makes {:.2f} tweets per day".format(number_of_days, len(df)/number_of_days))

df.head()
print("Number of NaN values")

df.isna().sum()
df = df.drop(["geo", "hashtags", "mentions", "link"], axis=1) # Not so much to deal with
max_retweets = df.sort_values("retweets", ascending=False)

content, date, retweets, favorites = 1, 2, 3, 4

place = 1



for tweet in max_retweets.values[:10]:

    print("\033[94mBest retweet {}:\033[0m \n{}\n{}\t\033[93m{} retweets\t\t{} favorites\033[0m\n".format(

        place, tweet[content], tweet[date], tweet[retweets], tweet[favorites]))

    place += 1
plt.subplot(2, 2, 1)

plt.title("Number of Tweets per Year")

df["date"].groupby(df["date"].dt.year).count().plot(kind="bar")

plt.xlabel("year")



plt.subplot(2, 2, 2)

plt.title("Number of Tweets per Day")

tweet_cpt = df["date"].groupby(df["date"].dt.day_name()).count()

tweet_cpt.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).plot(kind="bar")

plt.xlabel("year")

plt.show()
after_campaign = df[df["date"] >= np.datetime64("2015-01-01")]

before_campaign = df[df["date"] < np.datetime64("2015-01-01")]

after_2013 = df[df["date"] >= np.datetime64("2013-01-01")]

past_years = df[df["date"] >= np.datetime64("2017-01-01")]
X1 = np.log(after_campaign["retweets"]+1) # +1 to ensure values != 0

X2 = np.log(before_campaign["retweets"]+1)

plt.hist(X1, bins=20, alpha=0.5, label="After 2015")

plt.hist(X2, bins=20, alpha=0.5, label="Before 2015")



plt.title("Retweets Histogram")

plt.xlabel("# of Retweets (logarithmic scale)")

plt.ylabel("# of Rows")

plt.legend(loc='upper left')

plt.show()
after_2013["retweets"].describe()
after_campaign["retweets"].describe() # 2015 and +
past_years["retweets"].describe() # 2017 and +
df["retweets"].groupby(df["date"].dt.year).median().plot()

plt.title("Median Number of Retweets")

plt.xlabel("year")

plt.plot()
plt.plot(range(0, len(past_years["retweets"])), sorted(np.log(past_years["retweets"])))

plt.title("Sorted # of Retweets")

plt.ylabel("# of Retweets (logarithmic scale)")

frame = plt.gca()

frame.axes.get_xaxis().set_visible(False)

plt.show()
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LinearRegression



train = past_years.sample(frac=0.8,random_state=200)

test = past_years.drop(train.index)

y_train = train["retweets"]

y_test = test["retweets"]
def draw_predicted(model, X_test, y_test):

    """

    Draw the model's prediction, and superpose it with the

    expected values.

    """

    predicts = model.predict(X_test)

    df = pd.DataFrame({"predicted":predicts, "actual":y_test})

    df = df.sort_values("actual")

    plt.plot(range(0, len(df)), df["actual"], label="Actual", color='r', ls='dotted')

    plt.plot(range(0, len(df)), df["predicted"], label="Predicted", color='b', ls='dotted', alpha=0.3)

    plt.title("Model evaluation")

    plt.ylabel("# of Retweets")

    plt.legend(loc='upper left')

    frame = plt.gca()

    frame.axes.get_xaxis().set_visible(False)

    plt.show()
vect = CountVectorizer(min_df=5, stop_words="english")

X_train = vect.fit_transform(train["content"])

X_test = vect.transform(test["content"])



model = LinearRegression()

model.fit(X_train, y_train)

print("Score: {:.2f}".format(model.score(X_test, y_test)))

draw_predicted(model, X_test, y_test)
common_tweets = past_years[8 < np.log(past_years["retweets"])]

common_tweets = common_tweets[np.log(common_tweets["retweets"]) < 11]



plt.plot(range(0, len(common_tweets["retweets"])), sorted(np.log(common_tweets["retweets"])))

plt.title("Sorted # of Retweets")

plt.ylabel("# of Retweets (logarithmic scale)")

frame = plt.gca()

frame.axes.get_xaxis().set_visible(False)

plt.show()



train = common_tweets.sample(frac=0.8,random_state=200)

test = common_tweets.drop(train.index)

y_train = train["retweets"]

y_test = test["retweets"]
vect = CountVectorizer(min_df=5, stop_words="english")

X_train = vect.fit_transform(train["content"])

X_test = vect.transform(test["content"])



model = LinearRegression()

model.fit(X_train, y_train)

print("Score: {:.2f}".format(model.score(X_test, y_test)))

draw_predicted(model, X_test, y_test)
from sklearn.tree import DecisionTreeRegressor



vect = CountVectorizer(min_df=5, stop_words="english")

X_train = vect.fit_transform(train["content"])

X_test = vect.transform(test["content"])



tree = DecisionTreeRegressor(max_depth=5, random_state=180)

tree.fit(X_train, y_train)

print("Score: {:.2f}".format(tree.score(X_test, y_test)))

draw_predicted(model, X_test, y_test)
from sklearn.ensemble import RandomForestRegressor



vect = CountVectorizer(min_df=5, stop_words="english")

X_train = vect.fit_transform(train["content"])

X_test = vect.transform(test["content"])



tree = RandomForestRegressor(n_estimators=10, max_depth=100, random_state=180)

tree.fit(X_train, y_train)

print("Score: {:.2f}".format(tree.score(X_test, y_test)))

draw_predicted(model, X_test, y_test)
from wordcloud import WordCloud



vect = CountVectorizer(min_df=100, stop_words="english")

vect.fit(past_years["content"])

wc = WordCloud(width=1920, height=1080).generate_from_frequencies(vect.vocabulary_)

plt.imshow(wc, interpolation="bilinear")

plt.axis("off")

plt.show()