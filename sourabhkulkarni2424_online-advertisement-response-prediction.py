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

import sklearn

import matplotlib.pyplot as plt

import seaborn as sn

%matplotlib inline
data= pd.read_csv('../input/testtehiuh/advertising.csv')
data.head(4)
sn.heatmap(data.isnull(),cbar=False,yticklabels=False)
data['Age'].plot.hist(bins=20,rwidth= 0.8)
sn.jointplot(x='Age',y='Area Income',data=data)
sn.jointplot(x='Age', y='Daily Time Spent on Site', data = data, color='Green')
data.head(5)
data.drop(["Ad Topic Line","Timestamp","City","Country"],axis=1,inplace=True)

data.head(2)
x = data.drop("Clicked on Ad",axis=1)

y = data["Clicked on Ad"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train,y_train)
y_predict=model.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test,y_predict))

print(classification_report(y_test,y_predict))
a = model.predict([[80,35,8000,200,1]])



if a ==1 : 

    print("Person will click on Advertisement")

    

else:

    print("Person will not click on Advertisement")

a = model.predict([[70,28,70000,200,0]])



if a ==1 : 

    print("Person will click on Advertisement")

    

else:

    print("Person will not click on Advertisement")