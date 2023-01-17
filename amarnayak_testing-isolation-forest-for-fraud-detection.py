import pandas as pd

import numpy as np

cc =  pd.read_csv("../input/creditcard.csv")
cc.columns
cc_train= cc.drop('Class', 1)
from sklearn.ensemble import IsolationForest

clf = IsolationForest(n_estimators=100, max_samples=200)
#Train the model with the data.

clf.fit(cc_train)
# The Anomaly scores are calclated for each observation and stored in 'scores_pred'

scores_pred = clf.decision_function(cc_train)
#verify the length of scores and number of obersvations.

print(len(scores_pred))

print(len(cc))
# scores_pred is added to the cc dataframe 

cc['scores']= scores_pred
#I oberved an conflict with the name 'class'. Therefore, I have changed the name from class to category

cc= cc.rename(columns={'Class': 'Category'})
# Based on (Liu and Ting, 2008), anomalous observation is scored close to 1 

# and non anamolous observations are scored close to zero. 

# I have written a simple loop that will count the number of observation that has score more than 0.5 and is actually anomalous.

counter =0

for n in range(len(cc)):

    if (cc['Category'][n]== 1 and cc['scores'][n] >=0.5):

        counter= counter+1

print (counter)
# For convinience, divide the dataframe cc based on two labels. 

avg_count_0 = cc.loc[cc.Category==0]    #Data frame with normal observation

avg_count_1 = cc.loc[cc.Category==1]    #Data frame with anomalous observation
#Plot the combined distribution of the scores 

%matplotlib inline

import matplotlib.mlab as mlab

import matplotlib.pyplot as plt

%pylab.inline

normal = plt.hist(avg_count_0.scores, 50,)

plt.xlabel('Score distribution')

plt.ylabel('Frequency')

plt.title("Distribution of isoforest score for normal observation")

plt.show()
#Plot the combined distribution of the scores 

%matplotlib inline

import matplotlib.mlab as mlab

import matplotlib.pyplot as plt

%pylab.inline

normal = plt.hist(avg_count_1.scores, 50,)

plt.xlabel('Score distribution')

plt.ylabel('Frequency')

plt.title("Distribution of isoforest score for anomalous observation")

plt.show()