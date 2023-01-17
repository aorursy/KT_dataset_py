import pandas as pd

import numpy  as np
sample = pd.read_csv('../input/lish-moa/sample_submission.csv')

sample.iloc[:,1:] = 0.0

sample.to_csv('submission.csv',index=False)
# read in 

scored     = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

sample     = pd.read_csv('../input/lish-moa/sample_submission.csv')



# calculate

predictions = []

for target_name in list(scored)[1:]:

    rate = float(sum(scored[target_name])) / len(scored)

    predictions.append(rate)

predictions = np.array( [predictions] * len(sample) )



# write out

sample.iloc[:,1:] = predictions

sample.to_csv('submission.csv',index=False)