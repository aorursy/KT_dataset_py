# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib as plt

import seaborn as sns 

%matplotlib inline
data_nba = pd.read_csv("../input/salary/NBA_season1718_salary.csv",index_col=0)
data_nba.head()
data_nba.isnull().sum()
data_nba[data_nba['season17_18']==data_nba['season17_18'].max()]
data_nba[data_nba['season17_18']==data_nba['season17_18'].min()]
gkk = data_nba.groupby(['Tm','Player'])
gkk.first()
gk = data_nba.groupby('Tm')
gk.first()
gk.get_group('GSW')
data_nba['Tm'].value_counts().head(10).plot.bar()


data_nba['season17_18'].value_counts().head(20).plot.bar()