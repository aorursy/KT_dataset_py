# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
games = pd.read_csv("../input/ign.csv")
games.head()
games.isnull().sum()
games.drop(['Unnamed: 0','url'],axis=1,inplace=True)
games.head()
games.isnull().sum()
games['genre'].value_counts()[0:15].plot(kind='pie',autopct='%1.1f%%',shadow=True,explode=[0.2,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
plt.title('distribution of genre (games)')
games.groupby('release_month')['genre'].count().plot(color='r')
games.groupby('release_day')['genre'].count().plot(color='g')
games["score"].describe()
import sklearn as sk
games["score"].isnull().sum()
games["score"].hist()
