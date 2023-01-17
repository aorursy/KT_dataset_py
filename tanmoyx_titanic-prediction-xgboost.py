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
# Ignoring warning messages
import warnings
warnings.filterwarnings('ignore')

# Importing the basic required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
# Reading the input data and preview
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
# Checking train data stats
print('** DATA SHAPE **')
print(train.shape)
print('\n** DATA INFO **')
print(train.info())
print('\n** DATA DESCRIPTION **')
train.describe()
test.shape
train.head()
# Mapping Sex column to 0 an 1 instead of male and female.
# 0 -> Male
# 1 -> Female
train['Sex'] = train['Sex'].map({'male': 0, 'female':1}).astype(int)
cat_features = train[['Sex', 'Pclass', 'Embarked']].columns
for feature in cat_features:
    sns.barplot(y = "Survived", x = feature, data = train)
    plt.title(feature +" Vs Survived")
    plt.show()
