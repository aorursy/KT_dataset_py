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
import pandas as pd 

import numpy as np

from sklearn.datasets import load_boston  #importing important libraries along with boston dataset to work on from sklearn library
bs = load_boston()

X = bs.data 
Y = bs.target
print(X)

print(Y)
df = pd.DataFrame(X)
correlation_matrix = df.corr().abs() #printing the correlation matrix that will depict linear dependancy between columns.

print(correlation_matrix)
u_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape),k=1).astype(np.bool)) #selecting the upper triangle of the matrix because the lower triangle is just a mirror image of upper triangle
print(u_tri)  #the result will replace the lower triangle values with NAN along with the correlation value=1 because it is correlation of a column with itself, so there's no redundancy to seek

col_drop = [column for column in u_tri.columns if any(u_tri[column] > 0.85)] #selecting the columns that have correlation value>0.85 and need to be dropped

print(col_drop)
df1 = df.drop(df.columns[col_drop], axis=1)

print(df1)  #In this manner we drop the redundant/highly correlated column, while preseving one of each kind and saving valuable data