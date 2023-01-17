# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Date from: https://www.kaggle.com/vaibhavsxn/reddit-comments-labeled-data



import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier, Perceptron

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        fname = os.path.join(dirname, filename)



print("Done")

# print(fname)

# Any results you write to the current directory are saved as output.
# Copy/Paste here to run locally. Must adjust fname and aquire database



# ToDo: Experiment with different classifiers

# HashingVectorizer

# More: https://elitedatascience.com/machine-learning-algorithms



import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier, Perceptron

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.model_selection import train_test_split



#############

# Functions #

#############



def strings_to_vector(strings):

    str_vectors = vectorizer.transform(strings)

    #predict = clf_svm.predict(str_vectors)

    predict = model_pac.predict(str_vectors)

    return predict





def predict_cat_from_strings(strings):

    vectors = strings_to_vector(strings)

    names = []

    for vector in vectors:

        names.append(dic_tag_to_name[vector])

    return names





#############

# Load File #

#############



# overriding with local file

#fname = "Reddit_10k.pkl"



# Load dataframe

# We may need to do this line by line because of size

df_load = pd.read_pickle(fname)





################

# Prepare Data #

################



# Reduce size because of operational time causing problems

sample_size = len(df_load) # try 1000 to start

df = df_load[:sample_size].copy()

print("Copy created.")



# Create a dictionary mapping each Tag category to a Number

dic_tag = {}

dic_tag_to_name = {}

place = 0

for tag in df.Tag:

    if tag not in dic_tag:

        dic_tag[tag] = place

        dic_tag_to_name[place] = tag

        place += 1



# create a list called cls which holds full range (of y)

cls = [i for i in range(len(dic_tag))]



# Create a nuw column called Tag_ID

df["Tag_ID"] = [dic_tag[a] for a in df.Tag]



# Split Data

training, val = train_test_split(df, test_size=0.33, random_state=42)



# Assign domain & range

train_x = training.body

train_y = training.Tag_ID

val_x = val.body

val_y = val.Tag_ID





###################

# Vectorize Words #

###################





# Bag of Words

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



vectorizer = TfidfVectorizer()



vectorizer.fit(train_x)

train_x_vectors = vectorizer.transform(train_x)

val_x_vectors = vectorizer.transform(val_x)

print("Done with vectorization.")





##############

# Model Data #

##############



# clf model

#clf_svm = svm.SVC(kernel='linear')

#clf_svm.fit(train_x_vectors, train_y)



#train_y_predictions = clf_svm.predict(train_x_vectors)

#val_y_predictions = clf_svm.predict(val_x_vectors)





model_pac = PassiveAggressiveClassifier()

model_pac.partial_fit(train_x_vectors, train_y, classes=cls)

train_y_predictions = model_pac.predict(train_x_vectors)

val_y_predictions = model_pac.predict(val_x_vectors)



# random forest

#reddit_model = RandomForestRegressor(random_state = 1)

#reddit_model.fit(train_x_vectors, train_y)

#train_y_predictions = reddit_model.predict(train_x_vectors)

#val_y_predictions = reddit_model.predict(val_x_vectors)



print("Done with Predictions.")





##############

# Rate Data #

##############



# F1 Scores

from sklearn.metrics import f1_score

from sklearn.metrics import mean_absolute_error



# train_mae = mean_absolute_error( train_y_predictions, train_y)

# val_mae = mean_absolute_error( val_y_predictions, val_y)

# print("MAE: ", train_mae)



# Show errors

print("\nRandom Guess Error:" + str(100 - 100/len(dic_tag)))



# Calculate error in training data

err_train = 0.0

for i, item in enumerate(train_y):

    if item != train_y_predictions[i]:

        err_train += 1



err_train = err_train / len(train_y) * 100

print("Training Error: " + str(err_train))



# Calculate error in validation data

err_val = 0.0

for i, item in enumerate(val_y):

    #        print(train_y_predictions[i])

    #        print(item)

    if item != val_y_predictions[i]:

        err_val += 1



err_val = err_val / len(val_y) * 100

print("Validation Error: " + str(err_val))



# Record Data

data = {"SampleSize": [sample_size],

        "TestError": [err_train],

        "ValidationError": [err_val]}

saves_df = pd.DataFrame(data)

saves_df.to_csv("David_results.csv", mode="a", index=False)



# ToDo: make sure f1 score has meaning in this context



# labels = [keys for keys in dic_tag ]

# f1_val = f1_score(val_y, clf_svm.predict(val_x_vectors), average=None, labels = labels)

# f1_score(test_y, clf_log.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEUTRAL, Sentiment.NEGATIVE])

# print("f1 score: " + str(f1_val))





#############

# Play time #

#############





test_text = ["I love politics", "Apple or Samsung", "Knock Knock", "PIzza is deliicous", "Donald trump is an idiot",

             "Never have I ever been so humiliated"]

# test1 = clf_svm.predict(vectorizer.transform(test_text))

test1 = predict_cat_from_strings(test_text)

print("\n\nHere are the predictions:\n")

for (i, item) in enumerate(test1):

    print(test_text[i][:20] + "  -->   " + item)

# Play Around and have fun

test_text = ["I love politics",

             "Apple or Samsung",

             "Knock Knock",

             "PIzza is deliicous",

             "Donald trump is an idiot",

             "Never have I ever been so humiliated"]

# test1 = clf_svm.predict(vectorizer.transform(test_text))

test1 = predict_cat_from_strings(test_text)

print("\n\nHere are the predictions:\n")

for (i, item) in enumerate(test1):

    print(test_text[i][:20] + "  -->   " + item)