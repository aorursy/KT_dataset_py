# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

sample_submission_df = pd.read_csv('../input/sample_submission.csv')
# Check the shape of input DataFrame

print(train_df.shape)

print(test_df.shape)

print(sample_submission_df.shape)
# Check correlation

train_df_corr = train_df.corr()

print(train_df_corr)
#import some necessary librairies



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt  # Matlab-style plotting

import seaborn as sns
#Correlation map to see how features are correlated with SalePrice

corrmat = train_df.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True)
train_df_corr_sr = train_df_corr['SalePrice']

print(train_df_corr_sr[train_df_corr_sr.values >= 0.5])
# Target Value

train_sale_price_df = train_df['SalePrice']



# Object Variable (correlation rate is higher than 50%)

train_df_2 = train_df[['OverallQual', 'YearBuilt', 'YearRemodAdd', 

                      'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 

                      'TotRmsAbvGrd', 'GarageCars', 'GarageArea']]



# Object Variable (correlation rate is higher than 70%)

train_df_3 = train_df[['OverallQual', 'GrLivArea']]



train_df_2.head()
print(train_df_2.isnull().all())
# 機械学習ライブラリ scikit-learn

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



X = train_df_2.as_matrix()

y = train_sale_price_df.as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2, random_state=42) 

rfc = RandomForestClassifier(n_estimators=5, random_state=2)

rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
count = 0

for i in range(len(rfc_pred)):

    if np.abs(1 - (rfc_pred[i]/y_test[i])) > 0.1:

        count += 1



print(count / rfc_pred.size * 100)
# predicting with train_df_3

X = train_df_3.as_matrix()

y = train_sale_price_df.as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2, random_state=42) 
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
count = 0

for i in range(len(rfc_pred)):

    if np.abs(1 - (rfc_pred[i]/y_test[i])) > 0.1:

        count += 1



print(count / rfc_pred.size * 100)