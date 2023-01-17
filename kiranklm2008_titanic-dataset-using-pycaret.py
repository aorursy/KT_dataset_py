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
import numpy as np
import pandas as pd
import seaborn as sns
train = pd.read_csv('../input/titanicdataset-traincsv/train.csv')
train.count()
train.info()
train.isnull().sum()
train.head()
sns.heatmap(train.isnull(), yticklabels=False)
sns.set_style("whitegrid")
sns.countplot(x='Survived', data=train)
sns.set_style("whitegrid")
sns.countplot(x='Survived',hue='Sex', data=train )
sns.boxplot(x='Pclass', y='Age', data=train)
###Dropping variables not important to the model

train = train.drop(['Name', 'PassengerId', 'Ticket'], axis = 1)
train.head()
# Importing module and initializing setup
!pip install pycaret
from pycaret.classification import *
clf1 = setup(data = train, target = 'Survived')
##Comparing models

compare_models()
###Final Model 
Logistic_Regression = create_model('lr')
### Hyper Tuning of Model
tuned_lr =tune_model(Logistic_Regression)
