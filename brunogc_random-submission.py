# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
sample_submission = pd.read_csv("../input/sampleSubmission.csv")
sample_submission.head()
random_nums = np.random.random(size=len(sample_submission))
random_nums
sample_submission["Exited"] = random_nums
sample_submission.to_csv("my_first_submission.csv", index=False)