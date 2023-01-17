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
import pandas as pd

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn
# Preparatory part of the code

test = pd.read_csv('../input/titanic/test.csv') # load test dataset

test['Boy'] = (test.Name.str.split().str[1] == 'Master.').astype('int')

submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pd.Series(dtype='int32')})



# Three lines of code for LB = 0.78947 (not less 80% teams - Titanic Top 20%) 

# Reasoning the statements see below (EDA)

test['Survived'] = [1 if (x == 'female') else 0 for x in test['Sex']]     # Statement 1

test.loc[(test.Boy == 1), 'Survived'] = 1                                 # Statement 2

test.loc[((test.Pclass == 3) & (test.Embarked == 'S')), 'Survived'] = 0   # Statement 3



# Saving the result

submission.Survived = test.Survived

submission.to_csv('submission_S_Boy_Sex.csv', index=False)