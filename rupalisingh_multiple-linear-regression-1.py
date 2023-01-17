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

import seaborn







df = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")





# Missing Data

value = 0.0

df["salary"] = df["salary"].fillna(value)



# seperating into X and Y Variable

df_X = df.iloc[:,[2,4]]

df_Y = df.iloc[:, 12]





# Splitting into train test data



from sklearn.model_selection import train_test_split



X_train, X_test, Y_train,  Y_test = train_test_split(df_X, df_Y, test_size= 0.2, random_state= 0)







#fitting linear regression model into the dataset



from sklearn.linear_model import LinearRegression



linear = LinearRegression()

linear.fit(X_train, Y_train)



# predicting the results



Y_pred = linear.predict(X_test)





# finding intercept and regressor coefficient

linear.intercept_













linear.coef_