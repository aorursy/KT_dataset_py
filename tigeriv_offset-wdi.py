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
# I made a mistake so I need to fix it

# 42, 85, 128

# 43

submission = pd.read_csv("../input/bad-submission/submission.csv")

submission = submission.to_numpy()

for i in range(len(submission) - 1):

    if i % 43 != 42:

        submission[i, [1, 2]] = np.copy(submission[i+1, [1, 2]])

    else:

        final_row = np.copy(submission[i, [1, 2]]) + (submission[i-1, [1, 2]] - submission[i-2, [1, 2]])

        submission[i, [1, 2]] = final_row

my_columns = ["ForecastId", "ConfirmedCases", "Fatalities"]

df = pd.DataFrame(submission, columns=my_columns)

df = df.astype({"ForecastId": int, "ConfirmedCases": float, "Fatalities": float})

df.to_csv('submission.csv', index=False)