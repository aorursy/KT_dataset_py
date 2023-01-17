# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Q1 - Load the file trees.csv and print the median of Height accurate to at 

# least 1 decimal point. Enter the output in the answer box.

raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 

        'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'], 

        'age': [42, 52, 36, 24, 73], 

        'preTestScore': [4, 24, 31, 2, 3],

        'postTestScore': [25, 94, 57, 62, 70]}

df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'preTestScore', 'postTestScore'])

df

df.to_csv('submission.csv', index=False)

# Q2 - Load the file trees.csv and print the average Volume 

# accurate to at least 1 decimal point. Enter the output in the answer box. 
# Q3 - Using titanic.csv, print ONLY the standard deviation of age. 

# Enter your answer in the box (accurate to first decimal place)

# Q4 - Using trees.csv, print the difference between the median and the first quartile. 

# Enter your answer in the box (accurate to first decimal place)
# Q5 - In file rock.csv, print the Correlation Coefficient between area and shape. 

# What is the correlation between these two parameters?