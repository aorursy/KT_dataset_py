# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv('../input/titanic/train.csv')
sns.heatmap(df_train.isnull())
sns.set_style('whitegrid')
sns.countplot(x = 'Survived' , data = df_train , hue = 'Pclass')
sns.distplot(df_train['Age'].dropna() , kde = False)
df_train['Fare'].hist(bins = 20 , figsize = (10,4))
sns.boxplot(x = 'Pclass' , y = 'Age' , data = df_train)
def agecal(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2 :
            return 29
        else:
            return 24
    else:
        return Age
df_train['Age'] = df_train[['Age','Pclass']].apply(agecal,axis = 1)
sns.heatmap(df_train.isnull())
