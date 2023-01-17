import numpy as np

import pandas as pd
ftgru = pd.read_csv('../input/leakyfasttext/leaky_submission.csv') 

ftlstm = pd.read_csv('../input/fasttextglove/fin.csv')

ggru = pd.read_csv('../input/fasttextglove/submission_GloveGRU.csv') 

glstm = pd.read_csv('../input/fasttextglove/submission_GloveLSTM.csv')

# The value of an ensemble is (a) the individual scores of the models and

# (b) their correlation with one another. We want multiple individually high

# scoring models that all have low correlations. Based on this analysis, it

# looks like these kernels have relatively low correlations and will blend to a

# much higher score.

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

for label in labels:

    print(label)

    print(np.corrcoef([ftgru[label].rank(pct=True), ftlstm[label].rank(pct=True), ggru[label].rank(pct=True), glstm[label].rank(pct=True)]))

submission = pd.DataFrame()

submission['id'] = ggru['id']

for label in labels:

    submission[label] = ftgru[label].rank(pct=True) * 0.25 + ftlstm[label].rank(pct=True) * 0.25 + ggru[label].rank(pct=True) * 0.25 + glstm[label].rank(pct=True) * 0.25



submission.to_csv('submission.csv', index=False)