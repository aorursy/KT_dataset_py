# Set up libraries
import pandas as pd
import numpy as np
# import data
test = pd.read_csv("../input/TestData.csv",low_memory=False)
train = pd.read_csv("../input/TrainData.csv",low_memory=False)
# basic data prep for train
train_prep = train.copy()
train_prep['log__donations_and_bequests'] = train_prep["donations_and_bequests"].apply(lambda x: np.log(1+x))
# this successfully executes, even though there are negative values - weird.
# basic data prep for test
test_prep = test.copy()
test_prep['log__previous__donations_and_bequests'] = test_prep["previous__donations_and_bequests"].apply(lambda x: np.log(1+x))

# find the average log donation bequest in train
train_mean = train_prep['log__donations_and_bequests'].mean()
train_mean
# Define our common sense estimator
def naive_prediction(row):
    # Takes a training set, and either returns the previous fundraising value, or the average of all fundraising values
    if (np.all(pd.notnull(row['log__previous__donations_and_bequests']))):
        return row['log__previous__donations_and_bequests']
    else:
        return train_mean
# Apply our estimator and check the resulting numbers
test_prep['log__donations_and_bequests'] = test_prep.apply(naive_prediction,axis=1)
test_prep[['log__donations_and_bequests','log__previous__donations_and_bequests']].head(10)
# Create your submission file
test_prep[['id','log__donations_and_bequests']].to_csv('ExampleSubmission.csv',index=False)