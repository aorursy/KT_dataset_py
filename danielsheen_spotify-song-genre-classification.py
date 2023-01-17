from pandas import read_csv

import pandas as pd

import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

from matplotlib import pyplot

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

import numpy as np
df_2000 = pd.read_csv("../input/spotify-top-2000s-mega-dataset/Spotify-2000.csv")

df_top10s = pd.read_csv("../input/top-spotify-songs-from-20102019-by-year/top10s.csv", engine='python') # the engine needs to be changed otherwise UTF-8 error occurs

df_2000.head()
df_top10s.head()
df_2000.info()
df_top10s.info()
len(df_2000["Top Genre"].unique()), len(df_top10s["top genre"].unique())
df_2000["Top Genre"].value_counts(), df_top10s["top genre"].value_counts()
df_top10s.info()
df_2000.drop(columns = ['Index', 'Title', 'Artist', 'Year'], inplace = True)

df_top10s.drop(columns = ['Unnamed: 0', 'title', 'artist', 'year'], inplace = True)
df_top10s.columns = df_2000.columns # setting column names as each other

df = df_2000.append(df_top10s, ignore_index = True)
attributes = df.columns[1:]

for attribute in attributes:

    temp = df[attribute]

    for instance in range(len(temp)):

        if(type(temp[instance]) == str):

            df[attribute][instance] = float(temp[instance].replace(',',''))

# check data types using df.dtype
# first extracting the genre columns

# getting rid of white spaces and turning it all into lower cases

genre = (df["Top Genre"].str.strip()).str.lower()
# function to split the genre column

def genre_splitter(genre):

    result = genre.copy()

    result = result.str.split(" ",1)

    for i in range(len(result)):

        if (len(result[i]) > 1):

            result[i] = [result[i][1]]

    return result.str.join('')
# loop until the genre cannot be split any further

genre_m1 = genre.copy()

while(max((genre_m1.str.split(" ", 1)).str.len()) > 1):

    genre_m1 = genre_splitter(genre_m1)
len(genre_m1.unique())
genre_m1.value_counts()
unique = genre_m1.unique()

to_remove = [] 



# genres that have a single instance only will be placed within the to_remove array

for genre in unique:

    if genre_m1.value_counts()[genre] < 20: # 10 was arbitrarily chosen

        to_remove += [genre]

len(to_remove)
df['Top Genre'] = genre_m1

df
df.set_index(["Top Genre"],drop = False, inplace = True)

for name in to_remove:

    type(name)

    df.drop(index = str(name), inplace = True)

    
df["Top Genre"].value_counts()
train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 42)

# training set

X_train = train_set.values[:,1:]

y_train = train_set.values[:,0]



# test set

X_test = test_set.values[:,1:]

y_test = test_set.values[:,0]
from sklearn.preprocessing import StandardScaler



standard_scaler = StandardScaler().fit(X_train)



# Standard Scaler

X_train_ST = standard_scaler.transform(X_train)

X_test_ST = standard_scaler.transform(X_test)
# obtaining all unique classes

unique = np.unique(y_train)
from sklearn.preprocessing import label_binarize

from sklearn.preprocessing import LabelEncoder

# 1 hot encoding

y_test_1hot = label_binarize(y_test, classes = unique)

y_train_1hot = label_binarize(y_train, classes = unique)



# labelling

y_test_label = LabelEncoder()
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import SGDClassifier

from sklearn.multiclass import OneVsOneClassifier



models = []

models += [['Naive Bayes', GaussianNB()]]

models += [['SGD', OneVsOneClassifier(SGDClassifier())]]

models += [['Logistic', LogisticRegression(multi_class = 'ovr')]]

rand_forest = RandomForestClassifier(random_state = 42, min_samples_split = 5)
result_ST =[]

kfold = StratifiedKFold(n_splits = 10, random_state = 1, shuffle = True)



# Random Forest has to be done separately since it takes in one hot encoded labels instead

RF_cross_val_score = cross_val_score(rand_forest, X_train_ST, y_train_1hot, cv = 10, scoring = 'accuracy')

print('%s: %f (%f)' % ('Random Forest', RF_cross_val_score.mean(), RF_cross_val_score.std()))



for name, model in models:

    cv_score = cross_val_score(model, X_train_ST, y_train, cv = kfold, scoring = 'accuracy')

    result_ST.append(cv_score)

    print('%s: %f (%f)' % (name,cv_score.mean(), cv_score.std()))
from sklearn.metrics import precision_score, recall_score



result_precision_recall = []



# same reasoning as before for Random Forest

y_temp_randforest = cross_val_predict(rand_forest, X_train_ST, y_train_1hot, cv = 10)

result_precision_recall += [['Random Forest', precision_score(y_train_1hot, y_temp_randforest, average = "micro"), 

                            recall_score(y_train_1hot, y_temp_randforest, average = "micro")]]



print('%s| %s: %f, %s (%f)' % ('Random Forest', 'Precision Score: ', precision_score(y_train_1hot, y_temp_randforest, average = "micro"), 

                           'Recall Score: ', recall_score(y_train_1hot, y_temp_randforest, average = "micro")))



for name, model in models:

    y_pred = cross_val_predict(model, X_train_ST, y_train, cv = kfold)

    precision = precision_score(y_train, y_pred, average = "micro")

    recall = recall_score(y_train, y_pred, average = "micro")

    # storing the precision and recall values

    result_precision_recall += [[name , precision, recall]]

    print('%s| %s: %f, %s (%f)' % (name, 'Precision Score: ', precision, 'Recall Score: ', recall))

from sklearn.metrics import f1_score



for name, precision, recall in result_precision_recall:

    print("%s: %f" % (name, 2 * (precision * recall) / (precision + recall)))
# training the models

model_method1 = LogisticRegression(multi_class = 'ovr').fit(X_train_ST, y_train)



# getting predictions

predictions_method1 = model_method1.predict(X_test_ST)
from sklearn.metrics import confusion_matrix

print(f1_score(y_test, predictions_method1, labels = unique, average = 'micro' ))
# recalling our original dataframe

df = df_2000.append(df_top10s, ignore_index = True)

attributes = df.columns[1:]

for attribute in attributes:

    temp = df[attribute]

    for instance in range(len(temp)):

        if(type(temp[instance]) == str):

            df[attribute][instance] = float(temp[instance].replace(',',''))

# check data types using the following code

# df.dtypes
genre = df['Top Genre'].str.split(" ")

genre