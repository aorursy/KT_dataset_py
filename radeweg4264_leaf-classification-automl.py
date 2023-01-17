# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/leaf-classification/train.csv')

test = pd.read_csv('../input/leaf-classification/test.csv')

train.head(5)

test.head(5)
X = train.iloc[:,2:]

y = train.iloc[:,1]



from tpot import TPOTClassifier

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

tpot = TPOTClassifier(verbosity=2, generations=10, max_eval_time_mins=0.04, population_size=40)

tpot.fit(X_train, y_train)







    

    

    
from sklearn.metrics import classification_report

prediction = tpot.predict(X_val)

print (classification_report(y_val, prediction))