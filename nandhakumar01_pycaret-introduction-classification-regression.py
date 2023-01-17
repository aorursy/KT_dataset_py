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
!pip install pycaret

import numpy as np # linear algebra

import pandas as pd # data processing 

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

sub = pd.read_csv("../input/titanic/gender_submission.csv")
from pycaret.classification import *
train.head()
train.info()
clf1 = setup(data = train, 

             target = 'Survived',

             numeric_imputation = 'mean',

             categorical_features = ['Sex','Embarked'], 

             ignore_features = ['Name','Ticket','Cabin'],

             silent = True)
compare_models()