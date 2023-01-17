import pandas as pd

import numpy as np
def boolToExtreme(b):

    """Service function for predToExtremeValues"""

    if b:

        return np.float128(1 - np.float128(1e-15))

    return np.float128(1e-15)



def predToExtremeValues(predictions):

    """Get the predictions in the form required for submission

    as a pandas DataFrame and edit the probability to be exactly

    the extreme values 10^15 and 1-10^15"""

    

    predictions = predictions.apply(lambda el: [p==el.max() for p in el], axis=1)

    predictions = predictions.applymap(boolToExtreme)

    return predictions
def howManyWrong(logloss, N):

    """Returns the estimate number of wrong predictions based on the

    Spooky Author Challenge evaluations system. logloss is the score,

    N is the number of observations"""

    

    log1 = np.float128(np.log(np.float128(1 - 1e-15)))

    log2 = np.float128(np.log(np.float128(1e-15)))

    

    w = N * ((log1 - logloss) / (log2 - log1))

    return w

# For test purposes let's use sklearn's log_loss evaluator

from sklearn.metrics import log_loss



dataset = pd.read_csv('../input/train.csv')



def yToInt(author):

    if author == "EAP":

        return 0

    elif author == "HPL":

        return 1

    else:

        return 2



# True labels from the training set

y_true = [yToInt(a) for a in dataset.author]



# Fake predictions for test purposes (always predict EAP)

fakePred = pd.DataFrame([["id1212", 1, 0, 0]]*len(y_true), columns=("id", "EAP", "HPL", "MWS"))



# Fake prediction values: leave out the id column

fpv = fakePred[["EAP", "HPL", "MWS"]]



# Count the actual number of wrong predictions

aw = 0

for i in range(len(y_true)):

    if not fpv.loc[i][y_true[i]] == 1:

        aw += 1



print("Actual wrong predictions:", aw)



# logloss estimate

logloss = log_loss(y_true, fpv.as_matrix(), eps=1e-15)

print("Logloss:", logloss)

print("Calculated number of wrong predictions:", howManyWrong(logloss, len(y_true)))

print("Accuracy:", (len(y_true)-howManyWrong(logloss, len(y_true)))/len(y_true))
# On one submission I had a score of 0.44097

# Submitting the same predictions in the "extremized" form

# I got the following logloss value

logloss = 3.93369



# Number of predictions in the submission:

N = 8392



w = howManyWrong(logloss, N)

print("Calculated number of wrong predictions:", w)

print("Accuracy:", (N-w)/N)
