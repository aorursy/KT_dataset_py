# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')

df
df.isnull().sum()
x = df.drop(columns=['diagnosis'])

y = df['diagnosis']

print(x)

print(y)
from sklearn import preprocessing 

label_encoder = preprocessing.LabelEncoder()  

x= x.apply(label_encoder.fit_transform)

print(x)

y= label_encoder.fit_transform(y)

print(y)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0).fit(x, y)

clf.score(x, y)