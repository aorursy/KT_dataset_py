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
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab
%matplotlib inline
df=pd.read_csv('../input/standard-metropolitican-area-dataset/Standard Metropolitan Areas Data.csv')
df.head()
df.info()
df.isnull().sum()
df.mean(axis=0)
df.max(axis=0)
df.min(axis=0)
plt.scatter(df.percent_senior, df.crime_rate)

plt.title('Plot of Crime Rate vs Percent Senior') # Adding a title to the plot
plt.xlabel("Percent Senior") # Adding the label for the horizontal axis
plt.ylabel("Crime Rate") # Adding the label for the vertical axis
plt.show()
plt.plot(df.work_force, df.income, '--ro')  # ro = red circles
plt.xlabel("Work Force") 
plt.ylabel("Income")
plt.show()

plt.plot(df.work_force, df.income, color="r", label = 'labor') 
plt.plot(df.physicians, df.income, label='graduates') 

# Adding a legend
plt.legend()

plt.show()
plt.subplot(1,2,1)  
plt.plot(df.work_force, df.income, "go")
plt.title("Income vs Work Force")


plt.subplot(1,2,2).label_outer()

plt.plot(df.hospital_beds, df.income, "r^")
plt.title("Income vs Hospital Beds")

plt.suptitle("Sub Plots") 
plt.show()
plt.hist(df.income)
plt.show()
plt.title("Histogram")
plt.xlabel("Percentage of Senior Citizens")
plt.ylabel("Frequency")

plt.hist(df.percent_senior)
plt.show()
plt.bar(df.region, df.crime_rate, color="green")

plt.title("Bar Graph")
plt.xlabel("Region")
plt.ylabel("Crime Rate")
plt.show()
features = ['land_area', 'percent_city', 'percent_senior', 'physicians','hospital_beds', 'graduates', 'work_force', 'income', 'region']
X = df[features]
y = df.crime_rate 
y.head()
X.describe().transpose()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=25)
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth=3,random_state=10)

model.fit(X_train,y_train)
from sklearn.metrics import mean_squared_error,mean_absolute_error
print("Mean Squared Error: ",mean_squared_error(y_test,model.predict(X_test)))
print("Root Mean Squared Error: ",np.sqrt(mean_squared_error(y_test,model.predict(X_test))))
print("Mean Absolute Error: ",mean_absolute_error(y_test,model.predict(X_test)))