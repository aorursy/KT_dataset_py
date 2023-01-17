import pandas as pd

import matplotlib.pyplot as plt

train = pd.read_csv("../input/ucfai-dsg-fa19-default/train.csv")

test = pd.read_csv("../input/ucfai-dsg-fa19-default/test.csv")

ID_test = test['id']
train['GOOD_STANDING'].value_counts()
# So there are 9x as many good loans as bad (naturally, any reputable lender would avoid bad loans)

# This is problomatic, because most models will notice that most features are associated with good loans

# Therefore, they will most likely just predict all good loans. Why is this a problem?



# The score for this comp is an AUC ROC metric. In an oversimplified sense, this score is based on both

# how precise your positives are AND your negatives

# If you guess on either of them, you should expect the lowest score (0.5)



# There are almost 1 million examples, it is safe to undersample

# Undersampling is basically where we only use a subset of the training data so that our good loans/bad loans are equal

# The simple solution to this is just to randomly choose good loans to use until we are equal to bad loans

# Here is how we are going to undersample

import numpy as np



# Give me the -length - of the subset of -train- made up of entries with GOOD_STANDING == 0 

# In otherwords, how many bad loans are there?

bad_standing_len = len(train[train["GOOD_STANDING"] == 0])



# Give me the index of the subset of train where good_standing == 1 

# In otherwords, give me the index of all the good loans

good_standing_index = train[train['GOOD_STANDING'] == 1].index



# Randomly choose indices of good loans equal to the number of bad loans

random_index = np.random.choice(good_standing_index, bad_standing_len, replace=False)



# Give me the index of all the bad loans in train

bad_standing_index = train[train['GOOD_STANDING'] == 0].index



# Concatonate the indices of bad loans, and our randomly sampled good loans

under_sample_index = np.concatenate([bad_standing_index, random_index])



# Create a new pandas dataframe made only of these indices 

under_sample = train.loc[under_sample_index]



# Make sure it works, and make this undersampled dataframe our train

train['GOOD_STANDING'].value_counts()

under_sample['GOOD_STANDING'].value_counts()

train = under_sample
# As we did in Titanic, lets concatonate train and test

train.head()

train_len = len(train)

dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
# We are going to predict only using the grade that lending tree provided

dataset.head()

dataset = dataset[["grade", "GOOD_STANDING"]]
# Turn all the grades into One-Hot dummy variables

dataset.head()

dataset = pd.get_dummies(dataset, columns = ["grade"], prefix="grade")

dataset.head()
# Separate train and test

train = dataset[:train_len]

test = dataset[train_len:]

# Drop the good standing from test (which should all be empty)

test.drop(labels=["GOOD_STANDING"],axis = 1,inplace=True)



# Make sure they are ints

train["GOOD_STANDING"] = train["GOOD_STANDING"].astype(int)



Y_train = train["GOOD_STANDING"]



X_train = train.drop(labels = ["GOOD_STANDING"],axis = 1)
from sklearn.ensemble import RandomForestClassifier
# Let's jus tuse a basic random forest

RF = RandomForestClassifier()

RF.fit(X_train, Y_train)
test_standing = pd.Series(RF.predict(test), name="GOOD_STANDING")



results = pd.concat([ID_test,test_standing],axis=1)



results.to_csv("GradePrediction.csv",index=False)