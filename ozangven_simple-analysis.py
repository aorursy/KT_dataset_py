# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/finance-data/Finance_data.csv")
#Specific and simple graph

#Correlation btw saving objectives and gender

sns.countplot(x="What are your savings objectives?",hue="gender",data=df)
#Boxplot of correlation btw age and equity market

sns.boxplot(x="age", y="Equity_Market", data=df)
#Embracing and simple

#Correlation of gender between other parameters

sns.pairplot(df, hue="gender")
sns.jointplot(x="age", y="Equity_Market", data=df, kind="kde", color="y", height=7)