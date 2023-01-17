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

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.utils import shuffle

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold

from sklearn.feature_selection import SelectFromModel



from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score



from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression



pd.set_option('display.max_columns', 100)
trainset = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/train.csv')

testset = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/test.csv')
trainset.head()
# how many rows and columns

print("Train dataset (rows, cols): ", trainset.shape, "\nTest dataset (rows, cols):", testset.shape)
print("Columns in train not in test dataset:", set(trainset.columns)-set(testset.columns))
# uses code from https://www.kaggle.com/bertcarremans/data-preparation-exploration ( see references )

data = []

for feature in trainset.columns:

    # Defining the role

    if feature == 'target':

        use = 'target'

    elif feature == 'id':

        use = 'id'

    else:

        use = 'input'

        

    # Defining the type

    if 'bin' in feature or feature == 'target':

        type = 'binary'

    elif 'cat' in feature or feature == 'id':

        type = 'categorical'

    elif trainset[feature].dtype == float or isinstance(trainset[feature].dtype, float):

        type = 'real'

    elif trainset[feature].dtype == int:

        type = 'integer'

        

    # Initialize preserve to True for all variables except for id

    preserve = True

    if feature == 'id':

        preserve = False

        

    # Defining the data type

    dtype = trainset[feature].dtype

    

    category = 'none'

    # Defining the category

    if 'ind' in feature:

        category = 'individual'

    elif 'reg' in feature:

        category = 'registration'

    elif 'car' in feature:

        category = 'car'

    elif 'calc' in feature:

        category = 'calculated'

        

    # Creating a Dict that contains all the metadata for the variable

    feature_dictionary = {

        'varname': feature,

        'use': use,

        'type': type,

        'preserve': preserve,

        'dtype': dtype,

        'category': category

    }

    data.append(feature_dictionary)

    

metadata = pd.DataFrame(data, columns=['varname', 'use', 'type', 'preserve', 'dtype', 'category'])

metadata.set_index('varname', inplace=True)

metadata
# all categorical values : 

metadata[(metadata.type == 'categorical') & (metadata.preserve)].index
# count all features, "category"

pd.DataFrame({'count' : metadata.groupby(['category'])['category'].size()}).reset_index()
# count all features, "use" and "type"

# type (nominal, interval, ordinal, binary)

pd.DataFrame({'count' : metadata.groupby(['use', 'type'])['use'].size()}).reset_index()