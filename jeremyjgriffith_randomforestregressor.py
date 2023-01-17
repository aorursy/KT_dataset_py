# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib

sns.set_style('whitegrid')

%matplotlib inline



from sklearn.ensemble import RandomForestRegressor

from sklearn.cross_validation import cross_val_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Read Datasets

train_df   = pd.read_csv("../input/train.csv")

test_df    = pd.read_csv("../input/test.csv")



# preview the data

train_df.head()
train_df.info()

print('______________________')

test_df.info()
prices = pd.DataFrame({"price":train_df["SalePrice"], "log(price+1)":np.log1p(train_df["SalePrice"]) })

prices.hist()
# Transform the dependent variable 

train_df["SalePrice"] = np.log1p(train_df["SalePrice"])
all_data = pd.concat((train_df.loc[:, 'MSSubClass':'SaleCondition'],

                     test_df.loc[:, 'MSSubClass':'SaleCondition']))



all_data = pd.get_dummies(all_data)



all_data = all_data.fillna(all_data.mean())
X_train = all_data[:train_df.shape[0]]

X_test = all_data[train_df.shape[0]:]

Y_train = train_df.SalePrice
random_forest = RandomForestRegressor(n_estimators=500)



random_forest.fit(X_train, Y_train)



Y_pred = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)
from sklearn.metrics import make_scorer, mean_squared_error

scorer = make_scorer(mean_squared_error, False)

cv_score = np.sqrt(-cross_val_score(estimator=random_forest, X=X_train, y=Y_train, cv=15, scoring = scorer))





plt.figure(figsize=(10,5))

plt.bar(range(len(cv_score)), cv_score)

plt.title('Cross Validation Score')

plt.ylabel('RMSE')

plt.xlabel('Iteration')



plt.plot(range(len(cv_score) + 1), [cv_score.mean()] * (len(cv_score) + 1))

plt.tight_layout()
Y_pred_train = random_forest.predict(X_train)



pd.DataFrame({"predicted":Y_pred_train,"actual":Y_train })
Y_pred_transform = np.expm1(Y_pred)

Y_pred_transform
submission = pd.DataFrame({

        "Id": test_df["Id"],

        "SalePrice": Y_pred_transform

    })

submission

submission.to_csv('house_prices.csv', index=False)