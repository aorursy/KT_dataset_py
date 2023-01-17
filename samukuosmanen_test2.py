# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import Series,DataFrame
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

%matplotlib inline

# get training & test csv files as a DataFrame
train_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test_df    = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

# preview the data
train_df.head()

test_df.head()
train_df.info()
import seaborn as sns

# plot
sns.factorplot('Pclass','Survived', data=train_df,size=3,aspect=2)
sns.factorplot('Survived',col='Pclass', data=train_df,size=3,aspect=1,kind='count')

#sns.factorplot('Sex','Survived', data=train_df,size=3,aspect=2)
#sns.factorplot('Age','Survived', data=train_df,size=3,aspect=2)
sns.factorplot('e_deck','Survived', data=train_df,size=3,aspect=2)


train_df['e_deck']=train_df['Cabin'].str[:1]
train_df['e_deck'].fillna('X',inplace=True)

sns.factorplot('e_deck','Survived', data=train_df,size=3,aspect=2)