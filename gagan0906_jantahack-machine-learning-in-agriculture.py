!wget https://datahack-prod.s3.amazonaws.com/test_file/test_pFkWwen.csv

!wget https://datahack-prod.s3.amazonaws.com/train_file/train_yaOffsB.csv
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import cross_val_score



from catboost import CatBoostClassifier

from xgboost import XGBClassifier

train_df = pd.read_csv('train_yaOffsB.csv')

test_df = pd.read_csv('test_pFkWwen.csv')



train_df.drop('ID', inplace=True, axis=1)
train_df.dtypes
train_df.isna().sum()
sns.countplot(x='Crop_Damage', data=train_df)
train_df.boxplot(column=['Estimated_Insects_Count'], by='Crop_Damage', grid=False)
cols = ['Crop_Type','Soil_Type','Pesticide_Use_Category','Season']

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,8))

for idx, col_name in enumerate(cols):

    row = int(idx/2)

    col = int(idx%2)

    sns.countplot(x=col_name, hue='Crop_Damage', ax=ax[row, col], data=train_df)
pd.crosstab([train_df.Pesticide_Use_Category, train_df.Soil_Type], train_df.Crop_Type)
sns.scatterplot(train_df['Number_Weeks_Used'], train_df['Estimated_Insects_Count'])
train_df.corr()
train_df['Daily_avg'] = round(train_df['Number_Weeks_Used']/train_df['Number_Doses_Week'], 2)

train_df['Daily_avg'].fillna(value=0, inplace=True)



test_df['Daily_avg'] = round(test_df['Number_Weeks_Used']/test_df['Number_Doses_Week'], 2)

test_df['Daily_avg'].fillna(value=0, inplace=True)
y = train_df.pop('Crop_Damage')

X = train_df



test_ids = test_df.pop('ID')
X.fillna(value=-999, inplace=True)

test_df.fillna(value=-999, inplace=True)
catboost = CatBoostClassifier(iterations=2000, learning_rate=0.03, verbose=500)

catboost.fit(X, y)
sum(cross_val_score(catboost, X, y, cv=5))/5
xgboost = XGBClassifier()

xgboost.fit(X, y)

sum(cross_val_score(xgboost, X, y, cv=5))/5
catboost_preds = catboost.predict_proba(test_df)

xgboost_preds = xgboost.predict_proba(test_df)
final_res = []

for idx in range(len(xgboost_preds)):

    final_res.append(np.argmax((xgboost_preds[idx]*.4 + catboost_preds[idx]*.6)/2))



pd.DataFrame({'ID': test_ids, 'Crop_Damage':final_res}).to_csv('submission.csv', index=False)