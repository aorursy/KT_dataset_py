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
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
%matplotlib inline
cardio = pd.read_csv('/kaggle/input/cardiogoodfitness/CardioGoodFitness.csv')
cardio.head(5)
# Exploratory data analysis
sns.countplot(x='Product', hue = 'Gender', data = cardio)
sns.heatmap(cardio.corr())
pd.crosstab(cardio['Product'],cardio['MaritalStatus'])
pd.pivot_table(cardio,'Income',index=['Product','Gender'],columns=['MaritalStatus'])
sns.distplot(cardio['Miles'])
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
cardio.Gender = labelencoder.fit_transform(cardio.Gender)
cardio.MaritalStatus = labelencoder.fit_transform(cardio.MaritalStatus)
cardio.describe()
def age(x):
    if x >= 40:
        return 40
    elif x >= 30:
        return 30
    elif x >= 20:
        return 20
    else:
        return 18

cardio.Age = cardio.Age.apply(age)
cardio.Product = labelencoder.fit_transform(cardio.Product)
cardio.head(5)
features = cardio.drop(columns=["Miles"])
target = cardio.Miles

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)
predicted_values = linear_model.predict(x_test)

#Test accuracy score

score=linear_model.score(x_test,y_test)
score*100
