import numpy as np

import pylab as pl

import pandas as pd

import matplotlib.pyplot as plt 

%matplotlib inline

import seaborn as sns

from sklearn.utils import shuffle

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.model_selection import cross_val_score, GridSearchCV

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/predict-who-is-more-influential-in-a-social-network/train.csv")

test = pd.read_csv("../input/predict-who-is-more-influential-in-a-social-network/test.csv")
train.info()

train[0:10]
test.info()

test[0:10]
print("Any missing sample in training set:",train.isnull().values.any())

print("Any missing sample in test set:",test.isnull().values.any(), "\n")
#Frequency distribution of classes"

train_outcome = pd.crosstab(index=train["Choice"],  # Make a crosstab

                              columns="count")      # Name the count column



train_outcome
#Select feature column names and target variable we are going to use for training

features=['A_follower_count','A_listed_count','A_mentions_received','A_retweets_received','A_posts','A_network_feature_1','A_network_feature_2','A_network_feature_3','B_follower_count','B_following_count','B_listed_count','B_mentions_received','B_retweets_received','B_posts','B_network_feature_1','B_network_feature_2','B_network_feature_3']

target = 'Choice'
#This is input which our classifier will use as an input.

train[features].head(10)
#Display first 10 target variables

train[target].head(10).values
from sklearn.ensemble import RandomForestClassifier



# We define the model

rfcla = RandomForestClassifier(n_estimators=100,random_state=168,n_jobs=-1)



# We train model

rfcla.fit(train[features],train[target])



#Make predictions using the features from the test data set

predictions = rfcla .predict(test[features])



#Display our predictions

predictions
#Create a  DataFrame

submission = pd.DataFrame({"Id": list(range(1,len(predictions)+1)),

                         "Choice":predictions})



#Visualize the first 5 rows

submission.head()
#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

filename = 'submission.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)