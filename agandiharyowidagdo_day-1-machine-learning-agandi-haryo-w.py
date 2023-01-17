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
df = pd.read_csv('/kaggle/input/titanic/train.csv')
df.head(5)
df.isna().sum()
total_male = sum(df.Sex == 'male')
total_female = sum(df.Sex == 'female')
total_sex = total_male + total_female

ratio_male = (total_male/total_sex) * 100
ratio_female = (total_female/total_sex) * 100
print(f'ratio male is {ratio_male} and female is {ratio_female}')
df.Cabin.value_counts()
#df.Cabin.nunique()
df_parch = df[df['Parch'] >= 2]
df_parch.Parch.count()
df_alone = (df['SibSp'] == 0) & (df['Parch'] == 0)
df[df_alone]['Name']


