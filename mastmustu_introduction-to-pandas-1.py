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
df = pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')
df.head() # to see 5 records by default
df.head(10)
df.tail() # last 5 records
# data set info 



df.info()
df.describe()
# If you just want columns name  



df.columns
df.to_excel('IRIS_with_index.xls' ) # writing as excel with index 
df.to_excel('IRIS_with_no_index.xls', index = False ) # # writing as excel without index 
# Lets us see the difference 



df_1 = pd.read_excel('IRIS_with_index.xls')

df_1.head()
# Lets us see the difference 



df_2 = pd.read_excel('IRIS_with_no_index.xls')

df_2.head()
# to ways to solve this problem 







df_3 = pd.read_excel('IRIS_with_index.xls' , index_col = [0])

df_3.head()
x = list(range(1,101,2))

print(x)



series_x = pd.Series(x)  # Series is 1 dimension 

print(series_x)
x =np.arange(0,100,2)

print(x)



series_x = pd.Series(x)  # Series is 1 dimension 

print(series_x)


# dictionary with list object in values 

details = { 

'Name' : ['Ankit', 'Aishwarya', 'Shaurya', 'Shivangi'], 

'Age' : [23, 21, 22, 21], 

'University' : ['BHU', 'JNU', 'DU', 'BHU'], 

} 



# creating a Dataframe object 

df = pd.DataFrame(details) 



df 
