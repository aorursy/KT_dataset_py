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
df=pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df.head()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df.info()
sns.countplot(df['gender'],hue=df['status'])
sns.scatterplot(y=df['ssc_p'],x=df['sl_no'],hue=df['status'])
sns.countplot(df['ssc_b'],hue=df['status'])
sns.scatterplot(y=df['hsc_p'],x=df['sl_no'],hue=df['status'])
sns.countplot(df['hsc_b'],hue=df['status'])
sns.countplot(df['hsc_s'],hue=df['status'])
sns.scatterplot(y=df['degree_p'],x=df['sl_no'],hue=df['status'])
sns.countplot(df['degree_t'],hue=df['status'])
sns.countplot(df['workex'],hue=df['status'])
sns.scatterplot(y=df['etest_p'],x=df['sl_no'],hue=df['status'])
sns.countplot(df['specialisation'],hue=df['status'])
sns.scatterplot(y=df['mba_p'],x=df['sl_no'],hue=df['status'])
df[df['salary']==df['salary'].max()]
df[df['gender']=='M']['salary'].mean()
df[df['gender']=='F']['salary'].mean()
