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
import pandas as pd

import warnings

warnings.filterwarnings('ignore')



df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

pd.set_option('display.width', 1000)

pd.set_option('display.max_columns', 20)

print(df.head())

print(df.describe())





sta_map = {'Not Placed': 0, 'Placed': 1}

df['status'] = df['status'].map(sta_map)



wx_map = {'No': 0, 'Yes': 1}

df['workex'] = df['workex'].map(wx_map)



print(df.columns)

df = df.drop(['sl_no', 'gender', 'ssc_b', 'hsc_b', 'hsc_s',

               'degree_t', 'specialisation',

               'mba_p', 'salary'], axis=1)



print(df.head())



# Splitting the data into input and output variable

array = df.values

X = array[:, 0:5]

Y = array[:, 5]



# Splitting the data into training and testing sets

from sklearn.model_selection import KFold

kfold = KFold(n_splits=10, random_state=0)



from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()



from sklearn.model_selection import cross_val_score

results = cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')

print('Accuracy of the model :', round(results.mean()*100, 2))