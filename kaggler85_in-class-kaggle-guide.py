#This notebook is meant to guide those who do not know how to go about it.

#You are free to modify this code and try different models.

#You are further advised to use cross validation in your notebooks

#Also remember to 'commit'

#The score on the Leaderboard is 0.54766 so please don't make it your final submission. Keep pushing!
import pandas as pd



#Import training and testing data

train = pd.read_csv('../input/csm6420-workshop/train.csv')

test = pd.read_csv('../input/csm6420-workshop/test.csv',index_col=0)

print(train.head())

print(test.head())

X = train.drop('Class', axis=1)

y = train['Class']

print (y)
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

model = GaussianNB()

y_pred = model.fit(X_train, y_train).predict(X_test)

print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
#view Smple Submission file

sample = pd.read_csv('../input/csm6420-workshop/sampleSubmission.csv')

print(sample.head())
#Make predications on test data

y_pred = model.predict(test.values)

results = pd.DataFrame()

results["TestId"] = test.index.values

results["PredictedScore"]= y_pred



#Export result

results.to_csv("submission.csv", index=False)
#View Results Dataframe

print (results)