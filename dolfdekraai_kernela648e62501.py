# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import matplotlib.pyplot as plt

import pandas as pd



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import auc,matthews_corrcoef,make_scorer

from sklearn.svm import SVC
# Read the train and test datasets

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv",index_col=0)
def count_aa(seq,

             aa_order=["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"],

             normalize_count=True):

    """

    This function counts the occurrence of characters defined in 'aa_order'. The parameter 

    'normalize_count' defines if we normalize (divide by string length) the occurrence of

    a character.

    

    The function returns a pandas series for each character defined in 'aa_order' in the 

    specified order of this list.

    """

    

    # Do we need to normalize?

    if normalize_count:

        # List containing the character counts - normalized

        counted_aa = [seq.count(aa)/len(seq) for aa in aa_order]

    else:

        # List containing the character counts

        counted_aa = [seq.count(aa) for aa in aa_order]

    

    # Return a pandas Series, with the 'aa_order' as names

    return pd.Series(counted_aa,index=aa_order)



def extract_rolling_features(seq,lib={}, lib_name="feat", num_aas=5):

    """

    This function first converts the sequence to a numeric vector; as described in the

    provided dict library. Then it divides the vector into n splits (= seq length) with

    a specific amount (= num aas) looking ahead. For each split we take the sum and

    for these summed values of each split we return the maximum in a pandas series.

    """

    # Get the vector with numbers according to lib (so: seq to --> numeric vector of lib)

    feature_seqs = pd.Series([lib[aa] for aa in seq if aa in lib.keys()])

    # Return maximum window according to lib

    return pd.Series(feature_seqs.rolling(num_aas).sum().max(),index=["%s_%s" % (lib_name,num_aas)])



lib = {

        "A" : 1.800,

        "R" : -4.500,

        "N" : -3.500,

        "D" : -3.500,

        "C" : 2.500,

        "E" : -3.500,

        "Q" : -3.500,

        "G" : -0.400,

        "H" : -3.200,

        "I" : 4.500,

        "L" : 3.800,

        "K" : -3.900,

        "M" : 1.900,

        "F" : 2.800,

        "P" : -1.600,

        "S" : -0.800,

        "T" : -0.700,

        "W" : -0.900,

        "Y" : -1.300,

        "V" : 4.200

}



# We might need to scale features, for now this is not performed

X = []

X.append(train["sequence"].apply(extract_rolling_features,lib=lib,lib_name="hydrophob_rolling"))

X.append(train["sequence"].apply(count_aa))

X = pd.concat(X,axis=1)



y = train["target"]



# We might need to scale features, for now this is not performed

X_test = []

X_test.append(test["sequence"].apply(extract_rolling_features,lib=lib,lib_name="hydrophob_rolling"))

X_test.append(test["sequence"].apply(count_aa))

X_test = pd.concat(X_test,axis=1)
print("\nThe head of the X-matrix (training set):\n")

print(X.head(5))

print("\nThe shape of the X-matrix (training set):\n")

print(X.shape)

print("\n\n\nThe head of the y-vector:\n")

print(y.head(5))

print("\nThe shape of the y-vector (training set):\n")

print(y.shape)
print("\nThe head of the X-matrix (test set):\n")

print(X_test.head(5))
# Define some hyperparameters we are going to test

param_dist = {  

        "C": [0.01,0.1,1.0,10.0,50.0,100.0,150.0],

        "solver" : ["lbfgs"]

    }



# Use MCC scoring

mcc = make_scorer(matthews_corrcoef)



# Make a handle for our model (initialization or our model without fitted parameters)

logreg_model_handle = LogisticRegression()



# Define a CV strategy

cv = StratifiedKFold(n_splits=10,

                     shuffle=True,

                     random_state=42)



# Define how we are going to fit our model parameters using the CV, hyperparameters and what evaluation metric

grid_search = GridSearchCV(logreg_model_handle,

                           param_grid=param_dist,

                           verbose=0,

                           scoring=mcc,

                           n_jobs=1,

                           refit=True,

                           cv=cv)



# Fit the model parameters using the earlier defined search

random_search_res = grid_search.fit(X,y)



print("Best performance: %s" % (random_search_res.best_score_))

print("With the hyperparameters: %s" % (random_search_res.best_params_))
lr_model = random_search_res.best_estimator_
# Predict probabilities of belonging to either classes 

predictions_test = lr_model.predict_proba(X_test)[:,1]



# Create a dataframe with predictions

# Only problem is that we need to convert it to integers. This is now done by rounding and converting to integers.

# In future it might be interesting to determine the optimal threshold for the probability and assigning either of

# the two classes.

predictions_test_df = pd.DataFrame({"index":X_test.index,"target":list(map(int,map(round,predictions_test)))})



# Write predictions to a file

predictions_test_df.to_csv("predictions.csv",index=False)