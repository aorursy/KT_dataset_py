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
from sklearn.model_selection import train_test_split



# To preprocess all our data. We can add to this later

def preprocess(df):

    df = df[['Age','Fare']] # We're only interested in age and fare

    

    df = df.dropna() # Drop rows with no data

    

    x_train, x_test, y_train, y_test = train_test_split(df['Age'], df['Fare'], test_size=0.33, random_state=42) # Our testing set is 1/3 of the original, and we set a random seed so it is the same every time

    

    return x_train, x_test, y_train, y_test



train_df = pd.read_csv("/kaggle/input/titanic/train.csv")



x_train, x_test, y_train, y_test = preprocess(train_df)
import matplotlib.pyplot as plt



plt.scatter(x_train, y_train)
from sklearn.linear_model import LinearRegression



def reshape(data):

    return np.array(data).reshape(-1, 1) # Make it a numpy array and change it to be from a 1D list ([1, 2, 3, 4, 5]) to a 2D list ([[1], [2], [3], [4], [5]]) which sklearn expects



x = reshape(x_train)

y = reshape(y_train)



reg = LinearRegression() # Make the regression object

reg.fit(x, y)

print("Our m: ", reg.coef_)

print("Our b: ", reg.intercept_)
x_test = reshape(x_test)

y_test = reshape(x_test)



reg.score(x_test, y_test) # Returns the R^2. The closer it is to 1, the better