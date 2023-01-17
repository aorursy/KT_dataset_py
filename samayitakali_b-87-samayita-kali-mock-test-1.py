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
#Write a Pandas program to select the specified columns and rows from a given data frame.



import pandas as pd

import numpy as np



df  = pd.DataFrame({"Roll No.": [70,71,72,73,74,75,76,77,78,79,80],"Marks2": [45,3,np.nan,23,50,36,41,np.nan,32,4,np.nan],

        "Marks1": [20,37,43,50,47,29,7,31,11,49,32],"Remarks":['P', 'P','P','P','P','P','F','P','F','P','P']},

        index=[1,2,3,4,5,6,7,8,9,10,11])



print("Presenting the 1st,3rd,5th,6th,10th row along with 1st, 2nd and 3rd column")

s=df.iloc[[0,2,4,5,9],[0,1,2]]

print(s)
#Store height of 50 students in inches. Now while the data was being recorded manually there has been some typing mistake and 

#therefore height of 2 students have been recorded as 172 inch and 2 students have been recorded as 12 inch. 

#Graphically plot and show how you can seggregate correct data from abnormal data.



import numpy as np

import random 

import matplotlib.pyplot as plt



height = np.array([])

for i in range(50):

    height = np.append(height , [random.randint(52,90)])

height[10] = 172

height[48] = 172

height[4] = 12

height[27] = 12

plt.boxplot(height)

plt.show()

#Write a Python program to get the number of observations, missing values and nan values.



import pandas as pd

import numpy as np



data = pd.read_csv("../input/titanic/train_and_test2.csv")

print("Number of observations in the dataset: ")

print(data.info())



print("\n\nNaN values in the dataset:")

count = data.isnull().sum()

count1= data.isnull().sum().sum()

print(count)

print("Number of NaN values in the dataset: ")

print(count1)



print("\n\nNumber of missing values in the dataset: ")

miss1=data[data.isnull()].sum()

print(miss1)

print(data[data.isnull()].sum().sum())




