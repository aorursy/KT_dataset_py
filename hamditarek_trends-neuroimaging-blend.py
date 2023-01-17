import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt
submission1 = pd.read_csv('/kaggle/input/trends-neuroimaging-blending/submission_rapids_ensemble.csv')

submission2 = pd.read_csv('/kaggle/input/trends-neuroimaging-blending/submission (62).csv')

#submission3 = pd.read_csv('../input/tpuinference-super-fast-xlmroberta/submission (47).csv')
submission1
sns.set()

plt.hist(submission1['Predicted'],bins=100)

plt.show()
sns.set()

plt.hist(submission2['Predicted'],bins=100)

plt.show()
submission1['Predicted'] = submission1['Predicted']*0.4 + submission2['Predicted']*0.6
submission1.to_csv('submission.csv', index=False)