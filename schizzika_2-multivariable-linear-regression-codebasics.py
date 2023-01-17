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
import pandas as pd

import numpy as np

from sklearn import linear_model

import math

from word2number import w2n
df = pd.read_csv("/kaggle/input/hiring/hiring.csv")

df.head()
df.experience.fillna("zero", inplace = True)

df.experience = df.experience.apply(w2n.word_to_num)
df.head()
mean_test_score = df["test_score(out of 10)"].mean()
df["test_score(out of 10)"].fillna(mean_test_score, inplace = True)
df
reg = linear_model.LinearRegression()

reg.fit(df[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']], df['salary($)'])
reg.coef_
reg.intercept_
(2827.63404314 * 2) + (1912.93803053 * 9) + (2196.9753141  * 6) + 17237.330313727172
reg.predict([[2, 9, 6]])
reg.predict([[2, 9, 6]])