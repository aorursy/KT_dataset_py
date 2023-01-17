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
#Q.1 Write a Pandas program to get the powers of an array values element-wise.

import pandas as pd

df = pd.DataFrame({"a":[2,3,5,7,11,13,17],"b":[19,23,29,31,37,41,42],"c":[21,26,34,38,48,54,59]});

print(df)
#Q.2 Write Pandas program to create and display a DataFramefrom a specified dictionary data which has 

#the index labels.



#Q.3 Write a panda sprogram to get the first three rows of a given DataFrame.

import pandas as pd

df = pd.read_csv("../input/prediction-of-asteroid-diameter/Asteroid.csv");

row = df[0:3];

print(row)
#Q.4 Write a pandas program to select the specified columns and rows from a given DataFrame.

import pandas as pd

df = pd.read_csv("../input/prediction-of-asteroid-diameter/Asteroid.csv");

print(df.iloc[1:10,3:5])
#Q.5 write a pandas program to select the rows where the score is mising.

import pandas as pd

df = pd.read_csv("../input/prediction-of-asteroid-diameter/Asteroid.csv");

print(df[df[:].isnull()])