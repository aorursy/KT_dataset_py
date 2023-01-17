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
# Group number= 2
# Group members
# Anurag singh   18SCSE1010238
# AYUSH MEHROTRA 18SCSE1140042 
# DEVANSHI GUPTA 18SCSE1050033
# Activity =2
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
df=pd.read_csv('/kaggle/input/student-performance-data-set/student-por.csv')
df
# 2. Describe the dataset. 
df.describe(include = 'all')
# 3. Find mean,median and mode of columns.
df.mean()
# 3. Find mean,median and mode of columns. 
df.median()
# 3. Find mean,median and mode of columns. 
df.mode ()
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
plt.plot(df.age,label='age')
plt.plot(df.sex,label='sex')
plt.plot(df.Medu,label='Medu')
plt.legend()
# 5. Draw different types of graphs to represent information hidden in the dataset.
plt.hist(df.age)
plt.xlabel('age')
plt.ylabel('total number of student')
# 5. Draw different types of graphs to represent information hidden in the dataset.
female=df.loc[df['sex']=='F'].count()[0]
male=df.loc[df['sex']=='M'].count()[0]
labels=['Female','Male']
plt.pie([female,male],labels=labels,autopct='%.2f%%')

# 6. Find columns which are correlated.
# 7. Find columns which are not correlated.
df.corr()
# 6. Find columns which are correlated.
# As you see in the above table the positive and negative colerations are given 
# age is showing positive retaion of 1 with age age is showing 0.112805 positive relation with gooutand similary all the other 
# coloum are showing coleration with other for colerated  we consider only the positive value 
# 7. Find columns which are not correlated.
# As you see in the above table the positive and negative colerations are given 
# age is showing negative retaion of  -0.107832 with Medu age is showing -0.020559 positive relation with famrel similary all the other 
# coloum are showing coleration with other for not colerated we consider only the negative values value 
# 8. Compare different columns of dataset
comparison_column = np.where(df["G1"] == df["G2"], True, False)
df["equal"] = comparison_column
df
# 9. Is Any supervised machine learning possible ? if yes explain.

# yes supervised machine learning possible because
# Supervised learning is where you have input variables (x) and an output variable (Y) and you use an algorithm to learn the 
# mapping function from the input to  the output.

# Y = f(X)

# The goal is to approximate the mapping function so well that when you have new input data (x) that you can predict the 
# output variables (Y) for that data.

# It is called supervised learning because the process of an algorithm learning from the training dataset can be thought 
# of as a teacher supervising the learning process. We know the correct answers, the algorithm iteratively makes 
# predictions on the training data and is corrected by the teacher. Learning stops when the algorithm achieves an acceptable
# level of performance
 
# Some popular examples of supervised machine learning algorithms are:

# Linear regression for regression problems.
# Random forest for classification and regression problems.
# Support vector machines for classification problems
