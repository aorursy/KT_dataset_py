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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

import statsmodels.api as sm

sns.set()
data  = pd.read_csv('/kaggle/input/diabetes-dataset/diabetes2.csv')
data.describe(include = "all")
y = data["Outcome"]

x1 = data.drop(["Outcome"], axis = 1)

x_train,x_test,y_train,y_test = train_test_split(x1,y , test_size = 0.3, random_state = 101)

x_train_const = sm.add_constant(x_train)

reg_train = sm.Logit(y_train,x_train_const)

result_train = reg_train.fit()

result_train.summary()
def confusion_matrix(data,actual_values,model):

        

        # Confusion matrix 

        

        # Parameters

        # ----------

        # data: data frame or array

            # data is a data frame formatted in the same way as your input data (without the actual values)

            # e.g. const, var1, var2, etc. Order is very important!

        # actual_values: data frame or array

            # These are the actual values from the test_data

            # In the case of a logistic regression, it should be a single column with 0s and 1s

            

        # model: a LogitResults object

            # this is the variable where you have the fitted model 

            # e.g. results_log in this course

        # ----------

        

        #Predict the values using the Logit model

        pred_values = model.predict(data)

        # Specify the bins 

        bins=np.array([0,0.5,1])

        # Create a histogram, where if values are between 0 and 0.5 tell will be considered 0

        # if they are between 0.5 and 1, they will be considered 1

        cm = np.histogram2d(actual_values, pred_values, bins=bins)[0]

        # Calculate the accuracy

        accuracy = (cm[0,0]+cm[1,1])/cm.sum()

        # Return the confusion matrix and 

        return cm, accuracy
confusion_matrix(x_train_const,y_train,result_train)
x_test_const = sm.add_constant(x_test)

reg_test = sm.Logit(y_test,x_test_const)

result_test = reg_test.fit()

result_test.summary()
confusion_matrix(x_test_const,y_test,result_test)
