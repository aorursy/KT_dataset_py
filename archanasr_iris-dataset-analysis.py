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
'''Reading the Iris CSV data'''

irisdataframe = pd.read_csv("../input/Iris.csv")

         
'''Checking the rows and columns in the dataset'''

irisdataframe.shape
'''Looking at the column headers'''

irisdataframe.columns
'''Retriving first 5 rows of the dataset'''

irisdataframe.head()
'''Retriving last 5 rows of the dataset'''

irisdataframe.tail()
'''Picking a single column of the dataframe (Series)'''

irisdataframe.SepalLengthCm
'''Adding new column to the dataset - NewTestColumn'''

new_col = irisdataframe.PetalLengthCm + irisdataframe.PetalWidthCm

irisdataframe['NewTestColumn'] = new_col
'''Verifying the newly added column'''

irisdataframe.columns
'''Printing the full dataset with the newly added column'''

irisdataframe
'''Checking the type of the object'''

type(irisdataframe)
'''Checking the type of the object - A column type in dataframe'''

type(irisdataframe.Id)
'''Sorting  a column in the dataframe'''

irisdataframe.PetalLengthCm.sort_values()
'''Filtering the dataset based on multiple condition'''

irisdataframe[(irisdataframe.PetalLengthCm >= 4.0) | (irisdataframe.PetalWidthCm >= 3.0)]
'''Finding mean of all numeric columns'''

irisdataframe.mean()
'''Finding mean of all numeric columns grouped by iris species'''

irisdataframe.groupby('Species').mean()
'''using matplotlib'''

%matplotlib inline
'''Plotting the above functionality as a graph using matplotlib'''

irisdataframe.groupby('Species')['SepalLengthCm', 'SepalWidthCm','PetalLengthCm', 'PetalWidthCm'].mean().plot(kind="bar")
'''Dataset'''

irisdataframe
'''Comparing Sepal Length & Width each iris species wise'''

irisdataframe.groupby('Species')['SepalLengthCm', 'SepalWidthCm'].mean().plot(kind="bar")
'''Comparing Petal Length & Width each iris species wise'''

irisdataframe.groupby('Species')['PetalLengthCm', 'PetalWidthCm'].mean().plot(kind="bar")
# Count of each value in the column - SepalLengthCm

irisdataframe.SepalLengthCm.value_counts()
# Scatter plot - SepalLength Vs SepalWidth

irisdataframe.groupby("Species").plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm", color="red", )
# Scatter plot - PetalLength Vs PetalWidth

irisdataframe.plot(kind="scatter", x="PetalLengthCm", y="PetalWidthCm")