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
df = pd.read_csv("../input/cholera-dataset/data.csv")
df
df['Number of reported cases of cholera'].iloc[1059]='35'
df['Number of reported cases of cholera']=df['Number of reported cases of cholera'].fillna(0)
df['Number of reported cases of cholera']=df['Number of reported cases of cholera'].apply(int)
df
df2=df[(df['Year']<=2016)&(df['Year']>=2007)]
df2
df2[df2['Number of reported cases of cholera']==0]['Country']