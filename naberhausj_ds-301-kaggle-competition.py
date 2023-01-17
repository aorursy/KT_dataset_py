import numpy as np

import pandas as pd



train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')



train_df.info()
# Find the correlation between all features and the sale price

corrs = train_df.corr()['SalePrice']



# Find correlation that are significant enough to include in our model

significant_corrs = corrs[corrs > .5]

features = significant_corrs[significant_corrs.keys() != "SalePrice"].keys()
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression



x_train = train_df[features]

y_train = train_df['SalePrice']



model = Pipeline([

    ('std_scaler', StandardScaler()),

    ('imput', SimpleImputer(strategy='median')),

    ('logistic_regression', LogisticRegression(max_iter=1000))

])



model.fit(x_train, y_train)
test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



x_test = test_df[features]

y_pred = model.predict(x_test)
submission = pd.DataFrame({'Id': test_df.Id, 'SalePrice': y_pred})

submission.to_csv('logistic_regression.csv', index=False)