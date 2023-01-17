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
train_data=pd.read_csv('/kaggle/input/forest-cover-type-kernels-only/train.csv')

test_data=pd.read_csv('/kaggle/input/forest-cover-type-kernels-only/test.csv')

train_data.head(5)
print("Training point in data  = %i " % train_data.shape[0])

print(" Number of features  = %i " % train_data.shape[1])
train_data.describe()
corr_in_data = train_data.corr()
corr_in_data
train_data.drop(['Id'], inplace = True, axis = 1 )

train_data.drop(['Soil_Type15' , "Soil_Type7"], inplace = True, axis = 1 )

test_data.drop(['Soil_Type15' , "Soil_Type7"], inplace = True, axis = 1 )



from sklearn.model_selection import train_test_split

X = train_data.drop(['Cover_Type'], axis = 1)

Y = train_data['Cover_Type']

print( Y.head() )



from tpot import TPOTClassifier

X_train, X_test, y_train, y_test = train_test_split( X.values, Y.values, test_size=0.05, random_state=42 )

tpot = TPOTClassifier(verbosity=2, generations=10, max_eval_time_mins=0.04, population_size=40)

tpot.fit(X_train, y_train)

tpot.score(X_test,y_test)
from sklearn.metrics import classification_report

prediction = tpot.predict(X_test)

print (classification_report(y_test, prediction))


