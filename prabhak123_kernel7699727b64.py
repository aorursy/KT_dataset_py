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
# Q1-print in the order first name, last name and customer code.

input_str = 'Chowdhury_Nishith_001'

print(input_str[10:17])

print(input_str[0:9])

print(input_str[18:21])



#Q2-Remove SPSS from input_list=['SAS', 'R', 'PYTHON', 'SPSS'] and add 'SPARK' in its place

input_list=['SAS', 'R', 'PYTHON', 'SPSS']

input_list.pop(-1)

print(input_list)

input_list.append('SPARK')

print(input_list)

#Q3-Convert a string input_str = 'I love Data Science & Python' to a list by splitting it on ‘&’. 

input_str = 'I love Data Science & Python' 

input_str1=input_str.split('&')

print(input_str1)



#Q4-Convert a list ['Pythons syntax is easy to learn', 'Pythons syntax is very clear'] to a string using ‘&’.

list=['Pythons syntax is easy to learn', 'Pythons syntax is very clear'] 

str=' & '.join(list)

print(str)
#Q5. Extract Python from a nested list input_list =  [['SAS','R'],['Tableau','SQL'],['Python','Java']]

input_list =  [['SAS','R'],['Tableau','SQL'],['Python','Java']]

print(input_list[2][0])




