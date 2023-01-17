# general imports

import pandas as pd

import numpy as np

import math

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn import metrics

import time



# time based features

from datetime import datetime

from dateutil.parser import parse



# sentiment analysis

from textblob import TextBlob



# models

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
def classAssignment(x):

    if x == True:

        return 1

    else:

        return 0



def sentimentPolarity(x):

    blob = TextBlob(x)

    return blob.sentiment.polarity



def sentimentSubjectivity(x):

    blob = TextBlob(x)

    return blob.sentiment.subjectivity



def combineColumns(row):

    concat_columns = ""

    for column in row:

        words = column.split()

        for item in words:

            concat_columns += " " + item

    return concat_columns



def avgCommentPolarity(row):

    concat_comments = combineColumns(row)

    return sentimentPolarity(concat_comments)



def avgCommentSubjectivity(row):

    concat_comments = combineColumns(row)

    return sentimentSubjectivity(concat_comments)



def avgCommentLength(row):

    concat_comments = combineColumns(row)

    words = concat_comments.split()

    return len(words)



def capitalizedWordRatio(row):

    concat_columns = combineColumns(row)

    capital_count = 0

    total_count = 0

    words = concat_columns.split()

    

    # check for division by 0

    if len(words) == 0:

        return 0

    

    for word in words:

        total_count += 1

        if word[0].isalpha() and word == word.upper():

            capital_count += 1

    return capital_count / total_count



def dayPart(time):

    hour = parse(time).hour

    

    if hour < 7:

        return "EM"

    elif hour < 12:

        return "LM"

    elif hour < 17:

        return "EE"

    elif hour < 20:

        return "LE"

    elif hour < 22:

        return "PT"

    else:

        return "LN"

        

# read in data to df

#train_data_raw = pd.read_csv("../input/clickbait-thumbnail-detection/train.csv") 

#prediction_data_raw = pd.read_csv("../input/clickbait-thumbnail-detection/test_1.csv")



# update to test_2.csv data

train_data_raw = pd.read_csv("../input/clickbait-thumbnail-detection/train.csv") 

prediction_data_raw = pd.read_csv("../input/clickbait-thumbnail-detection/test_2.csv")



# create a dummy class column for test so the dfs are identical

prediction_data_raw["class"] = 0



train_data_raw.head(1)
# ~ 90 seconds

train_data = pd.DataFrame()

prediction_data = pd.DataFrame()



df_new = [train_data, prediction_data]

df_raw = [train_data_raw, prediction_data_raw]



user_comment_list = ["user_comment_" + str(x+1) for x in range(0,10)]



# perform the same transformations on both datasets

start_time = time.time()

for i in range(0,2):

    # bring in ready features

    df_new[i] = df_raw[i][["ID", "viewCount", "likeCount","dislikeCount","commentCount", "class"]]

    

    # turns class into 1 or 0

    df_new[i].loc[:,"class"] = df_raw[i]["class"].apply(classAssignment)

    

    # title length feature

    df_new[i].loc[:,"title_len"] = df_raw[i]["title"].apply(lambda x: len(x))

    

    # description length feature

    df_new[i].loc[:,"description_len"] = df_raw[i]["description"].apply(lambda x: len(x))

    

    # like dislike ratio feature

    df_new[i].loc[:,"like_dislike_ratio"] = (df_raw[i]["likeCount"] / df_raw[i]["dislikeCount"]).replace([np.inf, -np.inf], 100)

    

    # comment count / view count

    df_new[i].loc[:,"comment_view_ratio"] = (df_raw[i]["commentCount"] / df_raw[i]["viewCount"]).replace([np.inf, -np.inf], 0)

    

    # comment count / like ratio

    df_new[i].loc[:,"comment_like_ratio"] = (df_raw[i]["commentCount"] / df_raw[i]["likeCount"]).replace([np.inf, -np.inf], 0)

    

    # comment count / dislike ratio

    #df_new[i].loc[:,"comment_dislike_ratio"] = (df_raw[i]["commentCount"] / df_raw[i]["dislikeCount"]).replace([np.inf, -np.inf], 0)

    

    # log view count

    df_new[i].loc[:,"log_view_count"] = df_raw[i]["viewCount"].apply(lambda x: math.log2(x)).replace([np.inf, -np.inf], -1)

    

    # title polarity

    df_new[i].loc[:,"title_polarity"] = df_raw[i]["title"].apply(lambda x: sentimentPolarity(x))

    

    # title subjectivity

    df_new[i].loc[:,"title_subjectivity"] = df_raw[i]["title"].apply(lambda x: sentimentSubjectivity(x))

    

    # description polarity

    df_new[i].loc[:,"description_polarity"] = df_raw[i]["description"].apply(lambda x: sentimentPolarity(x))

    

    # description subjectivity

    df_new[i].loc[:,"description_subjectivity"] = df_raw[i]["description"].apply(lambda x: sentimentSubjectivity(x))

    

    # average comment length

    df_new[i].loc[:,"avg_comment_length"] = df_raw[i][user_comment_list].apply(avgCommentLength, axis=1)

    

    # average comment polarity

    df_new[i].loc[:,"avg_comment_polarity"] = df_raw[i][user_comment_list].apply(avgCommentPolarity, axis=1)

    

    # average comment subjectivity

    df_new[i].loc[:,"avg_comment_subjectivity"] = df_raw[i][user_comment_list].apply(avgCommentSubjectivity, axis=1)

    

    # title capitalization ratio

    df_new[i].loc[:,"title_capitalization_ratio"] = df_raw[i]["title"].apply(lambda x: capitalizedWordRatio(x))

    

    # description capitalization ratio

    df_new[i].loc[:,"description_capitalization_ratio"] = df_raw[i]["description"].apply(lambda x: capitalizedWordRatio(x))

    

    # comments capitalization ratio

    df_new[i].loc[:,"comment_capitalization_ratio"] = df_raw[i][user_comment_list].apply(capitalizedWordRatio, axis=1)

    

    # day part dummy creation

    df_new[i].loc[:,"day_part"] = df_raw[i]["timestamp"].apply(dayPart)

    df_new[i] = df_new[i].drop("day_part", axis=1).merge(pd.get_dummies(df_new[i]["day_part"], prefix="day_part"), 

                                                        right_index = True, left_index=True)

    

    df_new[i] = df_new[i].fillna(method="ffill")

    

    print("Part", i, "created. Time: ", round(time.time() - start_time, 2))

    

    

# save output into original dfs   

train_data = df_new[0]

prediction_data = df_new[1]



# check that it worked

train_data.head()
# check for any bad values that can cause an error in the next step

bad_indices = np.where(np.isinf(train_data.drop(columns=["ID","class"], axis=1)))

if len(bad_indices[0])== 0:

    print("No bad values")

else:

    print(bad_indices)
# split the training data into X and y

X = train_data.drop(columns=["ID","class"], axis=1)

y = train_data[["class"]]



# scale the training data

sc = StandardScaler()

X_scaled = sc.fit_transform(X)



# split to test and train groups

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)



# print shape as a check

X_train.shape, y_train.shape, X_test.shape, y_test.shape
#model = SVC(kernel='rbf',C=3000,gamma=0.3, probability=True) # -> 91%

#model = DecisionTreeClassifier() # -> 90.5%

#model = LogisticRegression() # -> 81%

model = XGBClassifier() # -> 90.6%



# fit the model to the training data

model.fit(X_train, np.array(y_train))



# print score

metrics.fbeta_score(model.predict(X_test), np.array(y_test), beta=0.5)
# scale the prediction data for input into the model

scaled_prediction_data = sc.transform(prediction_data.drop(columns=["ID","class"], axis=1))



# use the model to make predictions

Y_pred = model.predict(scaled_prediction_data)



# assign the class for the prediction data

prediction_data["class"] = Y_pred

prediction_data["class"] = prediction_data["class"].map(lambda x: "True" if x==1 else "False")



# subset the relevant columns

result = prediction_data[["ID","class"]]



# print to csv

result.to_csv("submission.csv", index=False)



# check the result to make sure its in the right format

result.head(10)