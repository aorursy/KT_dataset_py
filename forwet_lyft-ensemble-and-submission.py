# IMPORTS

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os
# Load Submission files

submission_1 = pd.read_csv("../input/submission-v3/submission_1000_iter.csv")

submission_2 = pd.read_csv("../input/submission-peter/submission_peter.csv")

submission_3 = pd.read_csv("../input/score-by-confidence/submission_new.csv")
# Utility Scripts



def load_submission(mode="multi"):

    """Returns submission file for corresponding mode viz multi-mode and single mode"""

    if mode=="multi":

        return pd.read_csv("../input/lyft-motion-prediction-autonomous-vehicles/multi_mode_sample_submission.csv")

    else:

        return pd.read_csv("../input/lyft-motion-prediction-autonomous-vehicles/single_mode_sample_submission.csv")



def generate_submission(base_submission, coefs, submissions=None):

    """Updates base_submission to the new predictions"""

    

    # Getting Columns

    cols = list(base_submission.columns)

    

    # Getting Coefficients and column indices

    # for predictions of different models

    coords = []

    coff = cols[2:5]

    cords1 = cols[5: 105]

    coords.append(cols[5: 105])

    coords.append(cols[105: 205])

    coords.append(cols[205:305])



    assert submissions != None , "At least one submission file should be provided"

    

    # Updating columns for corresponding submission file.

    # Note that even if you provide more than three submission files,

    # only the first three would be selected as size of coords is 3.

    print("Ensembling predictions...")

    for cord, each in zip(coords, submissions):

        base_submission[cord] = each[cords1]

    print("Ensembling done...")

    

    # Updating the coefs 

    assert len(coefs) == 3, "Length of coefs must be exactly three"

    assert sum(coefs) == 1, "Sum of coefs must be exactly one"

    

    base_submission[coff] = coefs

    

    return base_submission
sample_submission = load_submission(mode="multi")



submission = generate_submission(sample_submission, [0.2 ,0.1, 0.7], 

                                 [submission_1, submission_2, submission_3])
submission.head()
submission.to_csv("submission.csv", index=False, float_format="%.6g")