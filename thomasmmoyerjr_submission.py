# train-test split evaluation of random forest model

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



# import sentiment analyzer libraries as well as download

import nltk # https://www.nltk.org/install.html

import matplotlib.pyplot # https://matplotlib.org/downloads.html

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from statistics import mean

from nltk.tokenize import word_tokenize

from nltk.text import Text

nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
# load data

train_data = pd.read_csv("train.csv", usecols=["class","viewCount", "likeCount","dislikeCount","commentCount","title"])

train_data["title"] = train_data["title"].apply(lambda x: len(x))

test_data = pd.read_csv("test_2.csv", usecols=["ID","viewCount", "likeCount","dislikeCount","commentCount","title"])

test_data["title"] = test_data["title"].apply(lambda x: len(x))
# running comment analysis 

# comment analyzer for training data

SIA = SentimentIntensityAnalyzer()

all_comments = pd.read_csv("train.csv", usecols=["user_comment_1", "user_comment_2", "user_comment_3", "user_comment_4", "user_comment_5", "user_comment_6", "user_comment_7", "user_comment_8", "user_comment_9", "user_comment_10"])

avg_compound = []

averages_list = []

for i in range(len(train_data)):

    testing_row = all_comments.loc[i]

    for sentence in testing_row:

        scores = SIA.polarity_scores(sentence)

        compound = scores['compound']

        avg_compound.append(compound)

    mean = np.mean(avg_compound)

    avg_compound = []

    averages_list.append(mean)

train_data['compoundScore'] = averages_list



# comment analyzer for testing data

all_comments = pd.read_csv("test_2.csv", usecols=["user_comment_1", "user_comment_2", "user_comment_3", "user_comment_4", "user_comment_5", "user_comment_6", "user_comment_7", "user_comment_8", "user_comment_9", "user_comment_10"])

avg_compound = []

averages_list = []

for i in range(len(test_data)):

    testing_row = all_comments.loc[i]

    for sentence in testing_row:

        scores = SIA.polarity_scores(sentence)

        compound = scores['compound']

        avg_compound.append(compound)

    mean = np.mean(avg_compound)

    avg_compound = []

    averages_list.append(mean)

test_data['compoundScore'] = averages_list
# checking the amount of capitals in the titles of the data

# getting amount of capital letters in title of training data

all_titles = pd.read_csv("train.csv", usecols=["title"])

caps_list = []

for i in range(len(train_data)):

    testing_row = all_titles.loc[i]

    for sentence in testing_row:

        count=0

        for letter in sentence:

            if(letter.isupper()):

                count=count+1

    caps_list.append(count)

train_data["numberOfTitleCapitals"] = caps_list



# getting amount of capital letters in title of testing data

all_titles = pd.read_csv("test_2.csv", usecols=["title"])

caps_list = []

for i in range(len(test_data)):

    testing_row = all_titles.loc[i]

    for sentence in testing_row:

        count=0

        for letter in sentence:

            if(letter.isupper()):

                count=count+1

    caps_list.append(count)

test_data["numberOfTitleCapitals"] = caps_list
# gathering exclamation points in titles of the data

# getting amount of exclamation points in title of training data

all_titles = pd.read_csv("train.csv", usecols=["title"])

ex_list = []

for i in range(len(train_data)):

    testing_row = all_titles.loc[i]

    for sentence in testing_row:

        count=0

        for letter in sentence:

            if(letter == "!"):

                count=count+1

    ex_list.append(count)

train_data["numberOfTitleExcl"] = ex_list



# getting amount of exclamation points in title of testing data

all_titles = pd.read_csv("test_2.csv", usecols=["title"])

ex_list = []

for i in range(len(test_data)):

    testing_row = all_titles.loc[i]

    for sentence in testing_row:

        count=0

        for letter in sentence:

            if(letter == "!"):

                count=count+1

    ex_list.append(count)

test_data["numberOfTitleExcl"] = ex_list
# getting the like to dislike ratio from the train data

ratio = pd.read_csv("train.csv", usecols=["likeCount","dislikeCount"])

like_to_dislike = ratio["likeCount"] / (ratio["dislikeCount"])

train_data["like-to-dislike"] = np.log(like_to_dislike).replace([np.inf, -np.inf,np.nan],0)



# getting the like to dislike ratio from the test data

ratio = pd.read_csv("test_2.csv", usecols=["likeCount","dislikeCount"])

like_to_dislike = ratio["likeCount"] / (ratio["dislikeCount"])

test_data["like-to-dislike"] = np.log(like_to_dislike).replace([np.inf, -np.inf,np.nan],0)
# splitting data into X and y

y_train = train_data["class"]



X_train = train_data.drop("class", axis=1)

X_test = test_data.drop("ID", axis=1)
# fit random forest model to training data

model = RandomForestClassifier(n_estimators=200)

model.fit(X_train, y_train)
# make predictions for test data

y_pred = model.predict(X_test)
# storing result in csv file

test_data["class"] = y_pred

test_data["class"] = test_data["class"].map(lambda x: "True" if x==1 else "False")

result = test_data[["ID","class"]]

result.to_csv("submission.csv", index=False)