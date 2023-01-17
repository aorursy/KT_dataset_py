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
dataset = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
dataset.head()
dataset.info()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset['diagnosis'] = le.fit_transform(dataset['diagnosis']) 
dataset.head(20)
corr = dataset.corr()
corr['diagnosis'].sort_values(ascending=False)
import seaborn as sns
sns.heatmap(corr, cmap='coolwarm')
x = dataset.drop(['id', 'Unnamed: 32', 'diagnosis' ], axis=1)
y = dataset['diagnosis']
from sklearn.model_selection import train_test_split
xtr,xts, ytr, yts = train_test_split(x, y, test_size=0.3, random_state=101)
print(xtr.shape)
print(xts.shape)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=5000)
model.fit(xtr, ytr)
ypr = model.predict(xts)
model_score = model.score(xts, yts)
format(model_score,".2f")