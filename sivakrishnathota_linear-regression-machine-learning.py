import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import datasets

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Diabetes dataset from skleran 

df_diabetes = datasets.load_diabetes()
print("Features matrix shape",df_diabetes.data.shape)

print("Target vector shape",df_diabetes.target.shape)

print("Feature column name(Independent columns)",df_diabetes.feature_names)
x_train,x_test,y_train,y_test = train_test_split(df_diabetes.data,df_diabetes.target,

                                                 test_size=0.2,random_state=42)
model=LinearRegression()

model.fit(x_train,y_train)

model.score(x_test,y_test)
print("Model co-efficient values", model.coef_)

print("Y- Interecept",model.intercept_)
print("Model predection using test data",model.predict(x_test))
y_pred=model.predict(x_test)

plt.plot(y_test,y_pred,'*')

x=np.linspace(0,330,100)

y=x

plt.plot(x,y)

plt.show()