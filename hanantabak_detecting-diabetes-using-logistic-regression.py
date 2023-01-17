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
dataset_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_data_upload.csv"

df = pd.read_csv(dataset_path)
df.head()
df.isna().sum()
from sklearn.preprocessing import LabelEncoder

cat_cols = ['Gender','Polyuria','Polydipsia','sudden weight loss','weakness','Polyphagia','Genital thrush','visual blurring','Itching','Irritability','delayed healing','partial paresis','muscle stiffness','Alopecia','Obesity']

encoder = LabelEncoder()

encoded = df[cat_cols].apply(encoder.fit_transform)

encoded = encoded.join(df['class'])

encoded.head()
train_df = encoded.sample(frac = 0.8, random_state=0)

test_df = encoded.drop(train_df.index)
train_labels = train_df.pop('class')

test_labels = test_df.pop('class')
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

model = LogisticRegression()

model.fit(train_df,train_labels)

prediction = model.predict(test_df)

print(model.score(test_df,test_labels))
