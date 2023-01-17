# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns

#sns.set_style('whitegrid')

#sns.color_palette('pastel')

%matplotlib inline

#from sklearn.metrics import roc_curve

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics

dataset = pd.read_csv("/kaggle/input/transfusion/transfusion-1.data.csv")
#taking a look a what the data set looks like 

dataset.head()
# Renaming last column to 'Prediction' for brevity 

dataset.rename(

    columns={'whether he/she donated blood in March 2007': 'Prediction'},

    inplace=True

)



# Print out the first 2 rows

dataset.head(2)
#method for describing the dataset 

dataset.describe()

#trying to learn more about what I have in my dataset. I'm paying close attention to the Recency and Monetary columns 

#because I think they are important data points that can help with predicting whether a donor will donate again. 

 
#creating pairplot 

sns.pairplot(dataset)

#defining X and y values for splitting and training the module. This is also for dividing the data into attributes and labels

X = dataset[['Recency (months)', 'Frequency (times)', 'Monetary (c.c. blood)', 'Time (months)']].values

y = dataset['Prediction'].values

#splitting the dataset into a training set and a testing set. This should be splitting the data into 80% training and 20% testing. 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#training the model

regressor = LinearRegression()  

regressor.fit(X_train, y_train)
#this code will do a prediction test 

y_pred = regressor.predict(X_test)
#this is creating a dataframe to compare the actual results and the predicted results 

prediction_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

#prints the first 25 columns of the compared columns (Prediction column)

prediction_df1 = df.head(25)

#printing the prediction comparison 

prediction_df1
#plotting the prediction_df1 

prediction_df1.plot(kind='bar',figsize=(10,8))

#styling the grid

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

#printing the graph

plt.show()