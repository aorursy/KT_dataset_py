import pandas as pd

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
sample = pd.read_csv('../input/google-quest-challenge/sample_submission.csv')

sample
submit = sample

submit
submit.to_csv('submission.csv', index=False)