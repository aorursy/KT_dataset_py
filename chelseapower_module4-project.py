# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#import warnings filter

from warnings import simplefilter

# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)
#Load the sentiment_values.csv - The smallest dataset

df = pd.read_csv('../input/nowplayingrs/sentiment_values.csv')



#Look at size of the dataset

df.shape
#Look at the columns and initial rows of the dataset

df.head()
#Rename hashtag column

df.rename(columns = {'hashtag':'ss_score'}, inplace = True)



#Reset index

df.reset_index(inplace=True)



#Rename columns

df.rename(columns = {'level_0':'hashtag','level_1':'vader_score','level_2':'afinn_score','level_3':'ol_score'}, inplace = True)



#Show dataset to confirm changes

df.head()
#Set columns

df.columns=['hashtag','vader_score','afinn_score','ol_score','ss_score','vader_min','vader_max','vader_sum','vader_avg','afinn_min',

            'afinn_max','afinn_sum','afinn_avg','ol_min','ol_max','ol_sum','ol_avg','ss_min','ss_max','ss_sum','ss_avg']
df = df.drop(['vader_min','vader_max','vader_sum','afinn_min','afinn_max','afinn_sum','ol_min','ol_max','ol_sum','ss_min','ss_max','ss_sum'], axis=1)

df.head()
#Show how many (sum) unique values are in the hashtag column

len(df['hashtag'].unique().tolist())
df.info()

#There all score columns are missing values
#Fill in missing vader_score with vader_avg score, if available

df['vader_score'] = df.apply(

    lambda row: row['vader_avg'] if np.isnan(row['vader_score']) else row['vader_score'],

    axis=1

)
df.info()

#vader_score didn't change
#Fill in missing afinn_score with afinn_avg score, if available

df['afinn_score'] = df.apply(

    lambda row: row['afinn_avg'] if np.isnan(row['afinn_score']) else row['afinn_score'],

    axis=1

)
df.info()

#afinn_score increased from 3867 to 4532
#Fill in missing ol_score with ol_avg score, if available

df['ol_score'] = df.apply(

    lambda row: row['ol_avg'] if np.isnan(row['ol_score']) else row['ol_score'],

    axis=1

)
df.info()

#ol_score increased from 3867 to 4831
#Fill in missing ss_score with ss_avg score, if available

df['ss_score'] = df.apply(

    lambda row: row['ss_avg'] if np.isnan(row['ss_score']) else row['ss_score'],

    axis=1

)
df.info()

#ss_score increased from 3867 to 4471
#Remove all of the unnecessary scores - ol_score has the highest amount of ratings per hashtag

df1 = df.drop(['vader_score','afinn_score','ss_score','vader_avg','afinn_avg','ol_avg','ss_avg'], axis=1)

df1.head()
df1 = df1.dropna(axis = 0, how ='any') 

df1.info()
df1.sort_index(by='hashtag', ascending=[False])
#Rename column

df1.rename(columns = {'ol_score':'sentiment_score'}, inplace = True)

df1.head()
#Show top 10 hashtags with the largest sentiment_score

x = df1.nlargest(10, 'sentiment_score', keep='all')

x
#Look at dataset by sentiment score counts

count = df1.groupby(['sentiment_score']).count() 

print(count)
#Load second dataset

df2 = pd.read_csv('../input/nowplayingrs/user_track_hashtag_timestamp.csv')



#Look at size of the dataset

df2.shape
#Look at the columns and initial rows of the dataset

df2.head()
#Check for null values

df2.apply(lambda x: x.isnull().sum())
#Drop null rows

df2.dropna(subset=['hashtag'], inplace=True)

df2.apply(lambda x: x.isnull().sum())
# Get the count of the track_id

counts = df2['track_id'].value_counts()



# Select the items where the track_id count is less than 50 and remove them

df2 = df2[~df2['track_id'].isin(counts[counts < 50].index)]



# Show info

df2.info()
df2.user_id.value_counts()
df2.track_id.value_counts()
#Merge CSV files into a single file based on hashtag

df_sentiment = pd.merge(df1, df2, on="hashtag", how='inner')

df_sentiment.head()
#Confirm null values in new dataframe

df_sentiment.apply(lambda x: x.isnull().sum())
df_sentiment.shape
df_sentiment.hashtag.value_counts().head(10)
#Load third dataset and limit it to only load 22 columns

df3 = pd.read_csv('../input/nowplayingrs/context_content_features.csv', usecols=range(0, 22))



#Look at size of the dataset

df3.shape
# Get the count of the track_id

counts = df3['track_id'].value_counts()



# Select the items where the track_id count is less than 50 and remove them

df3 = df3[~df3['track_id'].isin(counts[counts < 50].index)]



# Show info

df3.info()
#Drop unnecessary columns before merging with df_sentiment dataframe

df3 = df3.drop(['coordinates','id','place','geo'], axis=1)

df3.head()
#Drop all null value rows

df3 = df3.dropna()



#Convert mode to Int64

df3['mode'] = df3['mode'].astype('Int64')



df3.shape
#Limit dataset to only en (English) language

df3 = df3.loc[~((df3['lang'] != 'en')),:]

df3.info()
#Confirm change by looking at the unique values in the lang column

df3.lang.unique()
#Merge df_sentiment and df3 CSV files into new CSV file based on track_id, created_at and user_id

df4 = df_sentiment.merge(df3, on=['track_id','created_at','user_id'], how='inner')

df4.head()
#Convert hashtag info string

df4['hashtag'] = df4['hashtag'].astype(str)



#Convert user_id info string

df4['user_id'] = df4['user_id'].astype(str)



#Show changes

df4.info()
#Create new column sentiment that will be the predictor based on the sentiment_score values

df4['sentiment'] = np.where(df4['sentiment_score']>= 0.01, 1, 0)

df4.head()
#Drop all null value rows

df4 = df4.dropna()



#Drop unnecessary columns lang and created_at

df4 = df4.drop(['hashtag','created_at','artist_id','tweet_lang','lang'], axis=1)



df4.head()
#Look at unique values for time_zone column

df4.time_zone.unique()
#Make all USA Timezone values consistent

df4['time_zone'].replace('Eastern Time (US & Canada)', 'Eastern Time',inplace=True)

df4['time_zone'].replace('Central Time (US & Canada)', 'Central Time',inplace=True)

df4['time_zone'].replace('Pacific Time (US & Canada)', 'Pacific Time',inplace=True)

df4['time_zone'].replace('Mountain Time (US & Canada)', 'Mountain Time',inplace=True)

df4['time_zone'].replace('Alaska', 'Alaska Time',inplace=True)

df4['time_zone'].replace('Hawaii', 'Hawaii Time',inplace=True)

df4['time_zone'].replace('Arizona', 'Mountain Time',inplace=True)

df4['time_zone'].replace('America/Chicago', 'Central Time',inplace=True)

df4['time_zone'].replace('America/New_York', 'Eastern Time',inplace=True)

df4['time_zone'].replace('America/Los_Angeles', 'Pacific Time',inplace=True)

df4['time_zone'].replace('America/Denver', 'Mountain Time',inplace=True)

df4['time_zone'].replace('America/Detroit', 'Eastern Time',inplace=True)
#Limit dataset to only USA time zones

df4 = df4.loc[~((df4['time_zone'] != 'Eastern Time') & (df4['time_zone'] != 'Central Time') & (df4['time_zone'] != 'Pacific Time') & 

               (df4['time_zone'] != 'Mountain Time') & (df4['time_zone'] != 'Alaska Time') & (df4['time_zone'] != 'Hawaii Time')),:]

df4.info()
# Reorder columns

df_mvp = df4[['sentiment','sentiment_score','user_id','track_id','time_zone','instrumentalness',

              'liveness','speechiness','danceability','valence','loudness','tempo','acousticness','energy','mode','key']]

df_mvp.info()
# Create new dataset with only user_id, track_id and time_zone and the category variables

df_timezone = df_mvp.drop(['user_id','track_id','sentiment','sentiment_score','instrumentalness','liveness','speechiness',

                             'danceability','valence','loudness','tempo','acousticness','energy','mode','key'], axis=1)

df_timezone.head()
from numpy import array

from numpy import argmax

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder



# One Hot Encode the category data set

one_hot_df = pd.get_dummies(df_timezone)

one_hot_df.head()
# Rename the columns

one_hot_df.columns = ['tz_Alaska_Time','tz_Central_Time','tz_Eastern_Time','tz_Hawaii_Time','tz_Mountain_Time','tz_Pacific_Time']

one_hot_df.head()
#Drop time_zone from MVP dataset

df_mvp = df_mvp.drop(["time_zone"], axis=1)
#Concatenate one hot encoded dataframe

df_mvp = pd.concat([df_mvp,one_hot_df], axis=1)

df_mvp.head()
df_mvp.to_csv('module4_cleaned.csv',index=False)
#Look at value counts of the predictor variable sentiment

df_mvp.sentiment.value_counts()
# Visualize the predictor variable

import seaborn as sns

import matplotlib.pyplot as plt



sns.countplot(x='sentiment', data=df_mvp, palette='hls')

plt.show()
# Create continuous dataset and look at distributions of data

df_mvp_lin = df_mvp.drop(['user_id','track_id','sentiment','mode','tz_Alaska_Time','tz_Central_Time','tz_Eastern_Time',

                          'tz_Hawaii_Time','tz_Mountain_Time','tz_Pacific_Time'], axis=1)

df_mvp_lin.hist(figsize = [30, 20]);
#Create coorelation heatmap and check for multicolinarity

from matplotlib import pyplot as plt

import seaborn as sns



correlation = df_mvp_lin.corr()

plt.figure(figsize=(14, 12))

heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
# Define X and y

y = df_mvp['sentiment']

X = df_mvp.drop('sentiment', axis = 1)
# Normalizing the data prior to fitting the model.

x_feats = ['sentiment_score','instrumentalness','liveness','speechiness','danceability',

           'loudness','tempo','acousticness','energy','mode','key','valence','tz_Alaska_Time',

           'tz_Central_Time','tz_Eastern_Time','tz_Hawaii_Time','tz_Mountain_Time','tz_Pacific_Time']



X = pd.get_dummies(df_mvp[x_feats], drop_first=False)

y = df_mvp.sentiment

X.head()
from sklearn.model_selection import train_test_split



# Splitting the data into train and test sets (automatically uses stratified sampling by labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression(fit_intercept = True, C=1e12)

model_log = logreg.fit(X_train, y_train)

model_log
print(y_train.value_counts())

print(y_test.value_counts())
#Predict against test set using Sigmoid function

import time

start = time.time()



y_hat_test = logreg.predict(X_test)

y_hat_train = logreg.predict(X_train)



end = time.time()

print("Execution time:", end - start)
y_hat_test = logreg.predict_proba(X_test)

y_hat_test[0]
import time

start = time.time()



logreg.predict_proba(X_train)



end = time.time()

print("Execution time:", end - start)
# How may times was the classifier correct for the training set?

logreg.score(X_train, y_train)
# How may times was the classifier correct for the test set?

logreg.score(X_test, y_test)
# Function to calculate the precision

def precision(y_hat, y):

    y_y_hat = list(zip(y, y_hat))

    tp = sum([1 for i in y_y_hat if i[0]==1 and i[1]==1])

    fp = sum([1 for i in y_y_hat if i[0]==0 and i[1]==1])

    return tp/float(tp+fp)
# Function to calculate the recall

def recall(y_hat, y):

    y_y_hat = list(zip(y, y_hat))

    tp = sum([1 for i in y_y_hat if i[0]==1 and i[1]==1])

    fn = sum([1 for i in y_y_hat if i[0]==1 and i[1]==0])

    return tp/float(tp+fn)
# Function to calculate the accuracy

def accuracy(y_hat, y):

    y_y_hat = list(zip(y, y_hat))

    tp = sum([1 for i in y_y_hat if i[0]==1 and i[1]==1])

    tn = sum([1 for i in y_y_hat if i[0]==0 and i[1]==0])

    return (tp+tn)/float(len(y_hat))
# Calculate the precision, recall and accuracy of the classifier.

y_hat_test = logreg.predict(X_test)

y_hat_train = logreg.predict(X_train)



print('Training Precision: ', precision(y_hat_train, y_train))

print('Testing Precision: ', precision(y_hat_test, y_test))

print('\n')



print('Training Recall: ', recall(y_hat_train, y_train))

print('Testing Recall: ', recall(y_hat_test, y_test))

print('\n')



print('Training Accuracy: ', accuracy(y_hat_train, y_train))

print('Testing Accuracy: ', accuracy(y_hat_test, y_test))
# concatenate our training data back together

training_data = pd.concat([X_train, y_train], axis=1)
# separate minority and majority classes

not_safe = training_data[training_data.sentiment==0]

safe = training_data[training_data.sentiment==1]
from sklearn.utils import resample

# upsample minority

not_safe_upsampled = resample(not_safe, 

                              replace=True, # sample with replacement

                              n_samples=len(safe), # match number in majority class

                              random_state=42) # reproducible results
# combine majority and upsampled minority

upsampled = pd.concat([safe, not_safe_upsampled])
# check new class counts

print(upsampled.sentiment.value_counts())
sns.countplot(x='sentiment', data=upsampled, palette='hls')

plt.show()
X_train = upsampled.drop('sentiment', axis=1)

y_train = upsampled.sentiment
# Scaling X using StandardScaler

from sklearn.preprocessing import StandardScaler



# Only fit training data to avoid data leakage

scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=list(X.columns))

X_test = pd.DataFrame(scaler.transform(X_test), columns=list(X.columns))

X_train.head()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



logreg2 = LogisticRegression(fit_intercept = True, C=1e12)

model_log = logreg2.fit(X_train, y_train)

model_log
print(y_train.value_counts())

print(y_test.value_counts())
# Predict against test set using Sigmoid function

import time

start = time.time()



y_hat_test = logreg2.predict(X_test)

y_hat_train = logreg2.predict(X_train)



end = time.time()

print("Execution time:", end - start)
y_hat_test = logreg2.predict_proba(X_test)

y_hat_test[0]
import time

start = time.time()



logreg2.predict_proba(X_train)



end = time.time()

print("Execution time:", end - start)
#Train score

logreg2.score(X_train, y_train)
#Test score

logreg2.score(X_test, y_test)
logreg2.coef_[0]
for feature, weight in zip(X.columns, logreg2.coef_[0]):

    print("{} has a weight of : {}".format(feature, weight))
#Create a Confusion Matrix

from sklearn.metrics import confusion_matrix



cnf_matrix = confusion_matrix(y_train, y_hat_train)

print('Confusion Matrix:\n',cnf_matrix)
#Plot the Confusion Matrix

import itertools

import matplotlib.pyplot as plt

%matplotlib inline



plt.imshow(cnf_matrix,  cmap=plt.cm.Blues) #Create the basic matrix



#Add title and axis labels

plt.title('Confusion Matrix')

plt.ylabel('True label')

plt.xlabel('Predicted label')



#Add appropriate axis scales

class_names = set(y) #Get class labels to add to matrix

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names, rotation=45)

plt.yticks(tick_marks, class_names)



#Add Labels to each cell

thresh = cnf_matrix.max() / 2. #Used for text coloring below

#Here we iterate through the confusion matrix and append labels to our visualization

for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):

        plt.text(j, i, cnf_matrix[i, j],

                 horizontalalignment="center",

                 color="white" if cnf_matrix[i, j] > thresh else "black")



#Add a side bar legend showing colors

plt.colorbar()
#conf_matrix function

def conf_matrix(y_true, y_pred):

    cm = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

    

    for ind, label in enumerate(y_true):

        pred = y_pred[ind]

        if label == 1:

            # CASE: TP 

            if label == pred:

                cm['TP'] += 1

            # CASE: FN

            else:

                cm['FN'] += 1

        else:

            # CASE: TN

            if label == pred:

                cm['TN'] += 1

            # CASE: FP

            else:

                cm['FP'] += 1

    return cm
#Set variable

model_confusion_matrix = conf_matrix(y_train, y_hat_train)
#Precision function

def precision(confusion_matrix):

    return confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FP'])
#Recall function

def recall(confusion_matrix):

    return confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FN'])
#Accuracy function

def accuracy(confusion_matrix):

    return (confusion_matrix['TP'] + confusion_matrix['TN']) / sum(confusion_matrix.values())
#f1 score

def f1(confusion_matrix):

    precision_score = precision(confusion_matrix)

    recall_score = recall(confusion_matrix)

    numerator = precision_score * recall_score

    denominator = precision_score + recall_score

    return 2 * (numerator / denominator)
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score



preds = [y_hat_train]



for ind, i in enumerate(preds):

    print('-'*40)

    print('Model Metrics:'.format(ind + 1))

    print('Precision: {}'.format(precision_score(y_train, i)))

    print('Recall: {}'.format(recall_score(y_train, i)))

    print('Accuracy: {}'.format(accuracy_score(y_train, i)))

    print('F1-Score: {}'.format(f1_score(y_train, i)))
from sklearn.metrics import classification_report



for ind, i in enumerate(preds):

    print('-'*40)

    print("Model Classification Report:".format(ind + 1))

    print(classification_report(y_train, i))
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

import matplotlib.pyplot as plt



num = 20

train_err = []

test_err = []

for i in range(num):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    logreg2.fit(X_train, y_train)

    y_hat_train = logreg2.predict(X_train)

    y_hat_test = logreg2.predict(X_test)

    train_err.append(mean_squared_error(y_train, y_hat_train))

    test_err.append(mean_squared_error(y_test, y_hat_test))

plt.scatter(list(range(num)), train_err, label='Training Error')

plt.scatter(list(range(num)), test_err, label='Testing Error')

plt.legend();
#K-Fold Cross Validation

from sklearn.model_selection import cross_val_score



cv_5_results = np.mean(cross_val_score(logreg2, X, y, cv=5, scoring="accuracy"))

cv_10_results = np.mean(cross_val_score(logreg2, X, y, cv=10, scoring="accuracy"))

cv_20_results = np.mean(cross_val_score(logreg2, X, y, cv=20, scoring="accuracy"))



print(cv_5_results)

print(cv_10_results)

print(cv_20_results)
# Load libraries

from numpy import loadtxt

from keras.models import Sequential

from keras.layers import Dense

from keras.callbacks import ModelCheckpoint



# split into input (X) and output (y) variables

X = pd.get_dummies(df_mvp[x_feats], drop_first=False)

y = df_mvp.sentiment
# define the keras model

model = Sequential()

model.add(Dense(12, input_dim=18, activation='relu'))

model.add(Dense(19, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
# compile the keras model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Set callback functions to early stop training and save the best model so far

checkpoint = [ModelCheckpoint(filepath='models.hdf5', save_best_only=True, monitor='val_loss')]
# fit the keras model on the dataset

model.fit(X, y, epochs=10, callbacks=checkpoint, batch_size=100)
# evaluate the keras model

_, accuracy = model.evaluate(X, y)

print('Accuracy: %.2f' % (accuracy*100))
# split into input (X) and output (y) variables

X = pd.get_dummies(X_train, drop_first=False) #X_train = upsampled.drop('sentiment', axis=1)

y = y_train #y_train = upsampled.sentiment
# define the keras model

model2 = Sequential()

model2.add(Dense(12, input_dim=18, activation='relu'))

model2.add(Dense(8, activation='relu'))

model2.add(Dense(1, activation='sigmoid'))
# compile the keras model

model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Set callback functions to early stop training and save the best model so far

checkpoint = [ModelCheckpoint(filepath='models.hdf5', save_best_only=True, monitor='val_loss')]
# fit the keras model on the dataset

model2.fit(X, y, epochs=10, callbacks=checkpoint, batch_size=100)
# evaluate the keras model

_, accuracy = model2.evaluate(X, y)

print('Accuracy: %.2f' % (accuracy*100))