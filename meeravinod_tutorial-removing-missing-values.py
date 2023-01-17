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
df=pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.head()

df.shape
df.info()
df.describe()
df.replace(0,np.NaN,inplace=True)
#inplace=true so that the change would be permanent

df.head()
#ok,the NaNs have turned up, so that's done :)
#lets get the sum of null population in all columns
df.isnull().sum()
#to drop all rows with NaN values
df.dropna(axis=0)
df.dropna(axis=1)
#column wise dropping. We see that all except 2 columns contained null values.
#created a random number data-frame with 4 columns and 100 rows.
a=np.random.seed(2)
new = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))
new.shape
new.head()
new.describe()
new.iloc[3:5,0]=np.nan
new.iloc[15:20,1]=np.nan

new.iloc[2:3,2]=np.nan


new.iloc[3:6,3]=np.nan
new.head()
#filling NANss column wise with the means in the respective columns.
new.fillna(new.mean())
#similarly with median
new.fillna(new.median())
# new.fillna(new.mode()[0])
fill_mode = lambda col: col.fillna(col.mode()[0])
new.apply(fill_mode, axis=0)