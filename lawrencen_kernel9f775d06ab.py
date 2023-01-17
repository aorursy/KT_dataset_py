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
# Done after the submission Data, hence answers already known and looked at. 



import pandas as pd

import numpy as np

import tensorflow as tf



# fixing the random seed for reproducibility

np.random.seed(50)

tf.random.set_seed(50)



# Saving the filepath of 'Phishing.csv' saved in the GitHub repository

filepath = "https://raw.githubusercontent.com/sayakpaul/Manning-Phishing-Websites-Detection/master/Phishing.csv"



# Loading the .csv into a DataFrame

myData = pd.read_csv(filepath)
myData.head(10)
myData.sample(12)
myData.columns
#Columns and raws 

myData.info
#Number of records and features 

myData.shape
myData['Result'].unique()
#T attributed notes during the revision

myData.T

#Column headers clear 

myData.head(5).T
myData.describe()
resultsData=myData['Result'].unique()
resultsData
myResultsData=myData['Result'].value_counts()
myResultsDataAsArray=np.array(myResultsData)
myResultsDataAsArray
myResultsDataAsArray[1]
data = np.array([['','Class','Num_Observations'],

                ['0',resultsData[0],myResultsDataAsArray[0]],

                ['1',resultsData[1],myResultsDataAsArray[1]]])

                

print(pd.DataFrame(data=data[1:,1:],

                  index=data[1:,0],

                  columns=data[0,1:]))
import matplotlib.pyplot as plt

# aggreate the data 

height = [myResultsDataAsArray[0],myResultsDataAsArray[1]]

bars = (resultsData[0], resultsData[1])

y_pos = np.arange(len(bars))

 

# Create bars

plt.bar(y_pos, height)

 

# Create names on the x-axis

plt.xticks(y_pos, bars)

 

# Show graphic

plt.show()
myData.describe().T
myData['Result']=myData['Result'].replace([-1],0)
myData.head(10).T
# Saving the filepath of 'Phishing.csv' saved in the GitHub repository

filepath = "https://raw.githubusercontent.com/sayakpaul/Manning-Phishing-Websites-Detection/master/Phishing.csv"



# Loading the .csv into a DataFrame

myData = pd.read_csv(filepath, header =None)

myData.T
(myData[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]]==0).sum()
y=myData[30]
y
y=myData['Result'].replace([-1],0)
y
x=myData.drop(['Result'],axis=1)
x
x.T
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.8,random_state=1)
len(y_train)
len(x_train)
len(x_test)