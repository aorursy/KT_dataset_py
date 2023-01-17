 # This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

%matplotlib inline
##create a multiplication table in a dataframe format



a = [5]*100                 ## create an array of 100 with value of 5

b = list(range(1,101))      ## another array of 100 with values 1 to 100

data = {"A":a,"B":b}        ## create a dictionary from the a and b values

df = pd.DataFrame(data)     ## convert the dictionary into a dataframe

df["C"] =df["A"]*df["B"]  ## create another column 'C' which holds the product of 'A and 'B' columns
df.head()  # check the dataframe structure
print(df.info())
print(df.describe())   #basic statistics
#Prepare the data for training



X = df[["A", "B"]]     

y= df["C"]                 # y is the target variable
# split the X and y dataframe into training and testing data



X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
lm = LinearRegression()    #create a linear regresson model

lm.fit(X_train,y_train)    #train the model using the training data
prediction = lm.predict(X_test)    #predict
plt.scatter(prediction, y_test) #compare the predicted values with the actual values. 

plt.xlabel("prediction")

plt.ylabel("y_test")

plt.title("Actual vs Prediction")



# A linear graph indicates a perfect prediciton
sns.distplot((y_test-prediction))
from sklearn import metrics

from sklearn.metrics import r2_score
mae = metrics.mean_absolute_error(y_test, prediction)  #mean absolute error

mae
mse=metrics.mean_squared_error(y_test, prediction)  #mean squared error

mse
rms = np.sqrt(mse)           #root mean squared error

rms
lm.coef_  # print the co-efficinets of the model
lm.intercept_
r2_score(y_test, prediction)   