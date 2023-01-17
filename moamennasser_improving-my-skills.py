import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OneHotEncoder # creating new columns with object's unique indexs
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
DataDir = '/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv'
campus_data = pd.read_csv(DataDir)
print("successfuly uploading the data.")
campus_data.describe()

campus_data.head()
s = (campus_data.dtypes == "object")
object_cols = list(s[s].index)

print('categorical columns:')
print(object_cols)
print("unic objects in the salary: ", campus_data['salary'].unique())
drop_data = campus_data.select_dtypes(exclude=['object'])
drop_data.head()
#this comman dropped the columns which contains objects.
# dropping the rows whitch contains "NaN" in salary column

index_names = campus_data[campus_data['salary'] == 'NaN'].index
campus_data.drop(index_names, inplace = True)

campus_data.head()

