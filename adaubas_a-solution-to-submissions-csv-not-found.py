import numpy as np

import pandas as pd



# First part of your notebook

 

# Part2



# Part 3



# Part 4

def prob(): # Your very complex model to do predictions

    return .5



# Part 5

sf = pd.read_csv('../input/deepfake-detection-challenge/sample_submission.csv', index_col='filename')

sf["label"] = prob()

sf["label"].fillna(0.5, inplace = True)

sf.reset_index(inplace=True)

sf[['filename','label']].to_csv('submission.csv', index=False)



# Part 6

print("Shape {}\nmin {}\nmax {}\nNA {}".format(sf.shape

                , sf["label"].min(), sf["label"].max(), sf["label"].isna().sum()))

sf.head()
from sklearn.metrics import log_loss



y = np.concatenate([np.ones(2000), np.zeros(2000)], axis=0)



# If you submit a submission file with those values

for i in [0, 0.05, 0.1, .15, .2, .25, .3, .35, .4, .45, .5, 1]:

    y_pred = np.full(4000, i)

    print("{:.2f} : {:.5f}".format(i, log_loss(y, y_pred)))

    

# Then you will have those public log loss :
# with small differences 

y = np.concatenate([np.ones(2010), np.zeros(1990)], axis=0)



# the log loss is not the same (except for 0.5)

for i in [0, 0.05, 0.1, .15, .2, .25, .3, .35, .4, .45, .5, 1]:

    y_pred = np.full(4000, i)

    print("{:.2f} : {:.5f}".format(i, log_loss(y, y_pred)))
import numpy as np

import pandas as pd



# read the sample submission file

_temp = pd.read_csv('../input/deepfake-detection-challenge/sample_submission.csv', index_col='filename')

_cte = len(_temp)



# Create a submssion file with i as a constant prediction

def submission_to_find_bug(i, verbose=False):

    ts = _temp.copy()

    y_pred = np.full(_cte, i)

    ts["label"] = y_pred

    ts.reset_index(inplace=True)

    ts[['filename','label']].to_csv('submission.csv', index=False)

    if verbose:

        print("Debug with value {}".format(i))

        print(ts.head(3))

        print(ts.tail(3))



# First part of the notebook

# ...

# Create the submission file with 0 as predcition for all videos 

submission_to_find_bug(0)



# Part2

# ...

# Create the submission file with 0.05 as predcition for all videos 

submission_to_find_bug(0.05)



# Part 3

# ...

submission_to_find_bug(0.1)



# Part 4

def prob(): # Imagine your very complex model making your predictions

    submission_to_find_bug(0.15)

    return .5

submission_to_find_bug(0.2, verbose=True)



# Part 5

sf = pd.read_csv('../input/deepfake-detection-challenge/sample_submission.csv', index_col='filename')

submission_to_find_bug(0.25)

sf["label"] = prob()

sf["label"].fillna(0.5, inplace = True)

sf.reset_index(inplace=True)

sf[['filename','label']].to_csv('submission.csv', index=False)



# Part 6

print("Shape {}\nmin {}\nmax {}\nNA {}".format(sf.shape

                , sf["label"].min(), sf["label"].max(), sf["label"].isna().sum()))

sf.head()