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
data = pd.read_csv('../input/creditcardfraud/creditcard.csv')
data.head()
data.corr()
data.Class.value_counts().plot(kind='bar')
data.Class.value_counts()
data.Class.isnull().value_counts()
na = [col for col in data.columns if data[col].isna().any()]
data.hist(figsize=(20,20))
import seaborn as sns
import matplotlib.pyplot as plt
figure = plt.figure(figsize=(20,20))
sns.heatmap(data.corr())
figure.show()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
x=data.drop('Class',axis=1)
y=data.Class
train_x,valid_x,train_y,valid_y=train_test_split(x,y,test_size=0.2,random_state=1)
model_1 = RandomForestClassifier(n_estimators=50)
model_1.fit(train_x,train_y)
prediction = model_1.predict(valid_x)
print("The accuracy score of the model is {}".format(accuracy_score(prediction,valid_y)))
