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
import seaborn as sns

from matplotlib import pyplot as plt



from sklearn.decomposition import PCA

from sklearn.datasets import load_iris



from sklearn.preprocessing import StandardScaler



pd.set_option('display.max_columns', None)
train_df = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/train.csv', index_col=0)

test_df = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/test.csv', index_col=0)
train_df=train_df.replace('?', np.nan)

test_df=test_df.replace('?', np.nan)
train_df['price']=train_df['price'].fillna(train_df['price'].median())

train_df['normalized-losses']=train_df['normalized-losses'].fillna(train_df['normalized-losses'].median())



test_df['price']=test_df['price'].fillna(test_df['price'].median())

test_df['normalized-losses']=test_df['normalized-losses'].fillna(test_df['normalized-losses'].median())
train_df=train_df.fillna(0)

test_df=test_df.fillna(0)
train_df=pd.get_dummies(train_df, columns=['make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'engine-type', 'num-of-cylinders', 'fuel-system'])

display(train_df)
test_df=pd.get_dummies(test_df, columns=['make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'engine-type', 'num-of-cylinders', 'fuel-system'])

display(test_df)
#train_df = train_df[['num-of-doors_two','body-style_hatchback', 'body-style_convertible', 'make_mitsubishi', 'make_porsche', 'make_saab', 'engine-location_rear', 'normalized-losses', 'price', 'symboling']]

#test_df = test_df[['num-of-doors_two','body-style_hatchback', 'body-style_convertible', 'make_mitsubishi', 'make_porsche', 'make_saab', 'engine-location_rear', 'normalized-losses', 'price']]
del train_df['wheel-base']

del train_df['length']

del train_df['width']

del train_df['height']

del train_df['engine-size']

del train_df['curb-weight']

del train_df['bore']

del train_df['stroke']

del train_df['compression-ratio']

del train_df['horsepower']

del train_df['peak-rpm']
del test_df['wheel-base']

del test_df['length']

del test_df['width']

del test_df['height']

del test_df['engine-size']

del test_df['curb-weight']

del test_df['bore']

del test_df['stroke']

del test_df['compression-ratio']

del test_df['horsepower']

del test_df['peak-rpm']
print(list(filter(lambda x: x not in test_df.columns, train_df.columns)))
del train_df['make_chevrolet'] 

del train_df['make_jaguar']

del train_df['make_mercury']

del train_df['num-of-doors_0']

del train_df['engine-type_dohcv']

del train_df['num-of-cylinders_eight']

del train_df['num-of-cylinders_three']

del train_df['num-of-cylinders_twelve']
test_df['fuel-system_mfi']=0

test_df['fuel-system_spfi']=0
X_train = train_df.drop('symboling', axis=1).to_numpy()

y_train = train_df['symboling'].to_numpy()

X_test = test_df.to_numpy()
#from sklearn.svm import SVC

#model = SVC()

#model.fit(X_train, y_train)
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()

model.fit(X_train, y_train)
model.score(X_train, y_train)
p_test = model.predict(X_test)

p_test
submit_df = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/sampleSubmission.csv', index_col=0)

submit_df['symboling'] = p_test

submit_df
submit_df.to_csv('submission01.csv')