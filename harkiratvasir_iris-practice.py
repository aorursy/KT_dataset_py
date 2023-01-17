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
df = pd.read_csv("../input/iris-dataset/iris.data.csv",names =['SepalLength','SepalWidth','PetalLength','Petalwidth','Species'])
df.head()
df.info()
df.describe()
df.isnull().sum()
import seaborn as sns
sns.pairplot(df,hue ='Species')
import matplotlib.pyplot as plt

fig,axes = plt.subplots(2,2,sharex = True,sharey = True,figsize = (10,10))

sns.boxplot(x=df['Species'],y=df['SepalLength'],ax = axes[0,0])

sns.boxplot(x=df['Species'],y=df['SepalWidth'],ax = axes[0,1])

sns.boxplot(x=df['Species'],y=df['PetalLength'],ax = axes[1,0])

sns.boxplot(x=df['Species'],y=df['Petalwidth'],ax = axes[1,1])
df['Species'].value_counts()
X = df.iloc[:,0:4].values

y= df.iloc[:,-1].values
from sklearn.preprocessing import LabelEncoder

labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(max_iter = 10000)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))