#Team Members: Shivam Vashi



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
from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn import tree



train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

train['label'] = 1



test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test['label'] = 0



concat_df = pd.concat([train , test])

features_df = pd.get_dummies(concat_df)



train = features_df[features_df['label'] == 1]

test = features_df[features_df['label'] == 0]



train = train.drop('label', axis=1)

test = test.drop('label', axis=1)





imputer = SimpleImputer()



imputed_train = pd.DataFrame(imputer.fit_transform(train))

imputed_train.columns = train.columns



model = Pipeline([('decisionTree', tree.DecisionTreeClassifier(max_depth = 400, max_features = 22))])







X_train = imputed_train.loc[:, imputed_train.columns != 'SalePrice']

y_train = imputed_train['SalePrice']

model.fit(X_train, y_train)
imputed_test = pd.DataFrame(imputer.fit_transform(test))



X_test = imputed_test

y_pred = model.predict(X_test)

print(y_pred)
submission = pd.DataFrame({'Id': test.Id, 'SalePrice': y_pred})

submission.to_csv('Shivam_Vashi_DT.csv', index=False)