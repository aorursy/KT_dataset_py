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
#1.) Read data into pandas data frame 
import pandas as pd
df=pd.read_csv("/kaggle/input/all-space-missions-from-1957/Space_Corrected.csv")
df

#2.) Summary of Dataset
import pandas as pd
df=pd.read_csv("/kaggle/input/all-space-missions-from-1957/Space_Corrected.csv")
df.info()

#3.)If any missing value is there findout how many?
import pandas as pd
df=pd.read_csv("/kaggle/input/all-space-missions-from-1957/Space_Corrected.csv")
df1=df.notnull()
print(df1)
count=df.isna().sum()
print("\n\nTotal number of missing values\n\n\n",count)

#4.)Sorting
import pandas as pd
df=pd.read_csv("/kaggle/input/all-space-missions-from-1957/Space_Corrected.csv")
col=input("Enter Column name: ")
df1=df.sort_values(by=col)
print("\n\nAfter Sorting by column",col,"\n\n")
df1