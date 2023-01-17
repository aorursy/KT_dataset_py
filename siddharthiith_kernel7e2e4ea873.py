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
irisData=pd.read_csv("/kaggle/input/iris/Iris.csv")
irisData.head()
irisData.info()
irisData.describe()
irisData['Species'].value_counts()
import seaborn as sns

sns.set_palette('husl')
data=irisData.drop('Id',axis=1)
y=data['Species']
y.head()
x=data.drop('Species',axis=1)
x
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
len(X_train)
len(X_test)
import math
def calcEuclideanDistance(a,b,features):

    

    d=0

    for x in range(features):

        d += pow((a-b),2)

    return math.sqrt(d)

        

    
from xgboost import XGBClassifier
model =XGBClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(y_pred)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: %.2f%%" % (accuracy * 100.0))