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
print("HELLO, I LIKE PYTHON PROGRAMMING LANGUAGE.");
##########################################################
# 1. IMPORT ALL PACKAGES
##########################################################
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
##########################################################
# 2. LOAD TITANIC DATASET AND VIEW SOME ROWS
##########################################################
# Load CSV data format
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
get_test_class = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

# View only 5 rows data
train.head(5)
##########################################################
# 3. DISPLAY INFO ABOUT EACH COLUMN
##########################################################
# Summary of the training dataset
train.info()

##########################################################
# 4. COUNT GENDER IN DATASET
##########################################################
sns.factorplot('Sex',data=train,kind='count')
##########################################################
# 5. COUNT SURVIVED AND DECEASED IN DATASET
##########################################################
sns.factorplot('Survived', data=train, kind='count')
##########################################################
# 6. COUNT GENDER VS SURVIVAL IN DATASET
##########################################################
sns.factorplot('Sex', kind='count', data=train, hue='Survived')
##########################################################
# 7. COUNT PCLASS IN DATASET
##########################################################
sns.factorplot('Pclass', data=train, kind='count')
##########################################################
# 8. COUNT PCLASS VS GENDER IN DATASET
##########################################################
sns.factorplot('Pclass', data=train, hue='Sex', kind='count')

##########################################################
# 9. COUNT PCLASS VS SURVIVAL IN DATASET
##########################################################
sns.factorplot('Pclass', data=train, hue='Survived', kind='count')