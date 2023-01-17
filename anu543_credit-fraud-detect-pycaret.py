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
%matplotlib inline

import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')
df.head(5)
df.isna().sum()
df['Class'].value_counts()
sns.countplot(df['Class'])
from sklearn.utils import resample
import imblearn
from imblearn.over_sampling import SMOTE

! pip install pycaret

from pycaret.classification import *
clf1 = setup(data =df ,target= 'Class')
compare_models()
