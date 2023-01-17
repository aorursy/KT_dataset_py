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
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
df.head()
df.shape
# Check for null values
df.isnull().sum()
#Drop the column with all missing values (na, NAN, NaN)
df = df.dropna(axis=1)
df.shape
#Check for diagnosis either 'B' or 'M'
df['diagnosis'].value_counts()
#Visualize this count 
sns.countplot(df['diagnosis'],label="Count")
df.dtypes
#Encoding categorical data values 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df.iloc[:,1]= le.fit_transform(df.iloc[:,1].values)
print(le.fit_transform(df.iloc[:,1].values))
# Check correlation
df.corr()
plt.figure(figsize=(20,20))  
sns.heatmap(df.corr(), annot=True)
x = df.iloc[:,2:31].values
y = df.iloc[:,1].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
#Feature Scaling to bring all features to the same level of magnitude
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
#Using Logistic Regression 
from sklearn.linear_model import LogisticRegression
log = LogisticRegression(random_state = 0)
log.fit(x_train, y_train)
pred = log.predict(x_test)
print(log.score(x_train,y_train))
from sklearn.metrics import classification_report
print(classification_report(y_test, pred))
