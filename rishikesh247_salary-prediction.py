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

df = pd.read_csv("../input/salaries.csv")

df.head()
df.isnull().sum()
inputs = df.drop('salary_more_then_100k',axis='columns')

target = df['salary_more_then_100k']

from sklearn.preprocessing import LabelEncoder



le_company = LabelEncoder()

le_job = LabelEncoder()

le_degree = LabelEncoder()



inputs['n_company'] = le_company.fit_transform(inputs['company'])

inputs['n_job'] = le_job.fit_transform(inputs['job'])

inputs['n_degree'] = le_degree.fit_transform(inputs['degree'])

inputs.head()
inputs.drop(['company','job','degree'], axis='columns', inplace=True)

from sklearn import tree



model = tree.DecisionTreeClassifier(criterion='entropy')

model.fit(inputs, target)
model.score(inputs,target)
model.predict([[2,1,0]])
model.predict([[2,1,1]])
