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

import plotly.express as px

df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')

df.head()
df.drop('Serial No.', inplace = True, axis = 1) #Dropping the Serial No. column
df.shape #Checking the shape of the dataframe
df.describe()
df.isnull().sum() #Checking the Null Values
for i in df.columns:

  fig = px.histogram(df, x = i)

  fig.show()
sns.pairplot(df)

plt.show()
plt.figure(figsize=(12,8))

sns.heatmap(df.corr(), annot=True)

plt.show()
df.columns
x = df.drop(columns = ['Chance of Admit '])

y = df['Chance of Admit '] #Target Variable
print(x.shape, y.shape) 
#Converting x & y into NumPy Arrays



x = np.array(x)

y = np.array(y)

y = y.reshape(-1,1)

y.shape
#Scaling the Data



from sklearn.preprocessing import StandardScaler, MinMaxScaler



scaler = StandardScaler()

minmax = MinMaxScaler()



x = scaler.fit_transform(x)

y = scaler.fit_transform(y)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, accuracy_score



lr_model = LinearRegression()



lr_model.fit(x_train, y_train)
accuracy_lr = lr_model.score(x_test, y_test)

print(accuracy_lr)
from sklearn.tree import DecisionTreeRegressor



dr_model = DecisionTreeRegressor() #Instantiate an object 



dr_model.fit(x_train, y_train)
accuracy_dr = dr_model.score(x_test, y_test)

accuracy_dr
from sklearn.ensemble import RandomForestRegressor



rf_model = RandomForestRegressor(n_estimators= 100, max_depth=25) #Instantiate an object 



#Try Experimenting with this n_estimators and max_depth parameters



rf_model.fit(x_train, y_train)
accuracy_rf = rf_model.score(x_test, y_test)

accuracy_rf