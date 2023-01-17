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
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')



train.head()
train['Survived'].value_counts(normalize=True)
def wrangle(df):

    df = df.drop('Name', axis = 1)

    df = df.drop('Ticket', axis = 1)

    df['Cabin'] = df['Cabin'].fillna('Unknown')

    return df



target = 'Survived'

train = wrangle(train)

test = wrangle(test)

X_train = train.drop(target, axis=1)

y_train = train[target]



X_train.describe(exclude='number')
X_train.describe()
import category_encoders as ce

from sklearn.impute import SimpleImputer

from sklearn.pipeline import make_pipeline

from xgboost import XGBClassifier



pipeline = make_pipeline(

    ce.OrdinalEncoder(), 

    SimpleImputer(strategy='most_frequent')  

)



X_train_encoded = pipeline.fit_transform(X_train, y_train)

X_test_encoded = pipeline.fit_transform(test)
X_train_encoded = pd.DataFrame(X_train_encoded, columns = X_train.columns)
X_test_encoded = pd.DataFrame(X_test_encoded, columns = test.columns)
classifier = XGBClassifier(n_jobs=-1)



classifier.fit(X_train_encoded, y_train)
final_preds = classifier.predict(X_test_encoded)
final_preds = pd.DataFrame(final_preds, columns=['Survived'])
final_preds['PassengerId'] = test['PassengerId']
os.chdir(r'/kaggle/working')



submission = final_preds.to_csv('final_preds.csv', index=False)
