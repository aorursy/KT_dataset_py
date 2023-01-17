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
1.

import numpy as np

import pandas as pd



exam_data  = {'name': ['Virat', 'Sourav', 'Naresh', 'Joy', 'Erik', 'Gareth', 'Emilia', 'Kobe', 'Ramesh', 'Suresh'],

        'marks': [15, 30, 17.5, 19, np.nan, 25, np.nan, 2, 28,16],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualified': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']



df = pd.DataFrame(exam_data , index=labels)

print("Select specific columns and rows:")

print(df.iloc[[1, 2, 4, 6], [2, 3]])
2.

import matplotlib.pyplot as plt

height=[70,65,67,80,65,62,71,73,64,72,75,67,63,82,78,78,68,69,72,172,86,84,12,72,172,12,76,63,65,66,58,70,76,80,64,67,75,74,76,76,79,65,68,81,76,78,66,78,69,83]

plt.boxplot(height)

plt.show()
3.

import pandas as pd 

# Read a dataset with missing values

flights = pd.read_csv("../input/titanic/train_and_test2.csv")

  # Select the rows that have at least one missing value

flights[flights.isnull().any(axis=1)].head()