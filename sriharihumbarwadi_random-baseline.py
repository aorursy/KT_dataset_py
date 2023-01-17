# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sample_submission = pd.read_csv("/kaggle/input/deepfake-detection-challenge/sample_submission.csv")

sample_submission.head()
sample_submission['label'] = np.random.uniform(low=0.6, high=1, size=sample_submission['label'].values.shape)

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head()