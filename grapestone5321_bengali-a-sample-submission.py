import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')

df.to_csv('submission.csv', index=False)
df.head(10)