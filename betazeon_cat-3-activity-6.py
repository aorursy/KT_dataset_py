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


# Group number= 18
# Group members
# 18SCSE1140015	ASHWANI TRIPATHI   
# 18SCSE1140038	Saurabh Kumar   
# 18SCSE1140075	Divyanshu Kumar Singh  
# Activity =6
# Answer some questions
# 1. Read the dataset.
# 2. Describe the dataset.
# 3. Find mean,median and mode of columns.
# 4. Find the distribution of columns with numerical data. Evaluate if they are normal or not.
# 5. Draw different types of graphs to represent information hidden in the dataset.
# 6. Find columns which are correlated.
# 7. Find columns which are not correlated.
# 8. Compare different columns of dataset
# 9. Is Any supervised machine learning possible ? if yes explain.


 # 1. Read the dataset.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv('/kaggle/input/human-resources-data-set/HRDataset_v13.csv')
df
# 2. Describe the dataset. 
df.describe(include = 'all')
# 3. Find mean,median and mode of columns.
print("\n----------- Calculate Mean -----------\n")
print(df.mean())
 
print("\n----------- Calculate Median -----------\n")
print(df.median())
 
print("\n----------- Calculate Mode -----------\n")
print(df.mode())
# 4. Find the distribution of columns with numerical data. Evaluate if they are normal or not.
# Numeric Dataset and these features are normal
numeric = list(df._get_numeric_data().columns)
numeric
# 4. Find the distribution of columns with numerical data. Evaluate if they are normal or not.
# Ordinal columns
# columns which are not normal are ordinal
categorical = list(set(df.columns) - set(df._get_numeric_data().columns))
categorical
# 5. Draw different types of graphs to represent information hidden in the dataset.
plt.rcParams['figure.figsize']=(10,6)

# 5. Draw different types of graphs to represent information hidden in the dataset.
df.plot.hist()
# 5. Draw different types of graphs to represent information hidden in the dataset.
df.plot.bar()

# 5. Draw different types of graphs to represent information hidden in the dataset.
plt.plot(df.EmpID,label='EmpID')
plt.plot(df.MarriedID,label='MarriedID')
plt.plot(df.DeptID,label='DeptID')
plt.legend()
# 5. Draw different types of graphs to represent information hidden in the dataset.
plt.hist(df.MarriedID)
plt.xlabel('Married')
plt.ylabel('total number of Employee')
# 6. Find columns which are correlated.
# 7. Find columns which are not correlated.
df.corr()
# 8. Compare different columns of dataset
comparison_column = np.where(df["EmpSatisfaction"] == df["SpecialProjectsCount"], True, False)
df["equal"] = comparison_column
df
# 9. Is Any supervised machine learning possible ? if yes explain.

# No supervised machine learning is not possible because
# Supervised learning is where you have input variables (x) and an output variable (Y) and you use an algorithm to learn the 
# mapping function from the input to  the output.

# Y = f(X)

# The goal is to approximate the mapping function so well that when you have new input data (x) that you can predict the 
# output variables (Y) for that data.

# It is called supervised learning because the process of an algorithm learning from the training dataset can be thought 
# of as a teacher supervising the learning process.
 
# Some popular examples of supervised machine learning algorithms are:

# Linear regression for regression problems.
# Random forest for classification and regression problems.
# Support vector machines for classification problems