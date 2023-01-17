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
#Q1)-LOad a data set of your choice,display the first 11 rows,display a row of that dataset

#having missing values, and replace missing values with Nan



# importing pandas package 

import pandas as pd 

	

# making data frame from csv file 

data = pd.read_csv("../input/ipl-data-set/matches.csv") 



# Printing the first 11 rows of 

# the data frame for visualization 

data[0:11] 

# will replace  Nan value in dataframe with value -99   

data.fillna("Nan") 

#Q3)-Take two NumPy array of your choice, find the common items between the arrays, 

#and remove the matching items but only from one array such that they exist in the second one.



import numpy as np

array1 = np.array([90, 600, 240, 470, 690])

print("Array1: ",array1)

array2 = [90, 30, 470]

print("Array2: ",array2)

print("Common values between two arrays:")

print(np.intersect1d(array1, array2))

#Q2Given the score of CSK, KKR, DC, and MI such that no two teams has the same score,

# chalk out an appropriate graph for the best display of the scores.

# Also, highlight the team having the highest score in the graph.



from matplotlib import pyplot as plt

teams=['CSK','KKR','DC','MI']

scores=['220','180','190','200']

plt.pie(scores,labels=teams,startangle=0,shadow=True,explode=(0.5,0,0,0),autopct='%1.1f%%')

#as the score of CSK is highest so I have put the value of explode to 0.5 for CSk