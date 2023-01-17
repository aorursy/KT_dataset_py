# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt





from sklearn.linear_model import LogisticRegression # SKLearn





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv")
data.head()
data.tail()
data.info()
data.drop(["Date","Location","WindGustDir","WindDir9am","WindDir3pm","RainTomorrow","RISK_MM"],axis=1,inplace=True)

data.fillna(data.median(),inplace=True)

data.head()
data.RainToday = [1 if i == 'Yes' else 0 for i in data.RainToday]

data.head()
x = data.drop("RainToday",axis=1).values

x = (x - np.min(x)) / (np.max(x) - np.min(x))
y = data.RainToday.values # We've created our x axis so we only need to create y axis



from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=75)
print(f"""

x_train's shape is {x_train.shape} 

x_test's shape is {x_test.shape}

y_train's shape is {y_train.shape}

y_test's shape is  {y_test.shape}

""")
y_train = y_train.reshape(-1,1)

y_test = y_test.reshape(-1,1)
print(f"""

x_train's shape is {x_train.shape} 

x_test's shape is {x_test.shape}

y_train's shape is {y_train.shape}

y_test's shape is  {y_test.shape}

""")
import warnings as wrn

wrn.filterwarnings('ignore')



lr_model = LogisticRegression()

lr_model.fit(x_train,y_train)
print("Our model's score is ",lr_model.score(x_test,y_test))