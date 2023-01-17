import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn import linear_model



train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")



train_data = train_data.replace({'Neighborhood': 

                                 {'Blmngtn': 16, 'Blueste': 7, 'BrDale': 3, 'BrkSide': 5,

                                  'ClearCr': 18, 'CollgCr': 17, 'Crawfor': 19, 'Edwards': 6,

                                  'Gilbert': 14,'IDOTRR': 2, 'MeadowV': 1, 'Mitchel': 12,

                                  'NAmes': 10, 'NPkVill': 11, 'NWAmes': 15,'NoRidge': 24,

                                  'NridgHt': 25, 'OldTown': 4,'SWISU': 9, 'Sawyer': 8,

                                  'SawyerW': 13,'Somerst': 21, 'StoneBr': 23,'Timber': 22,

                                  'Veenker': 20}})

test_data = test_data.replace({'Neighborhood': 

                                 {'Blmngtn': 16, 'Blueste': 7, 'BrDale': 3, 'BrkSide': 5,

                                  'ClearCr': 18, 'CollgCr': 17, 'Crawfor': 19, 'Edwards': 6,

                                  'Gilbert': 14,'IDOTRR': 2, 'MeadowV': 1, 'Mitchel': 12,

                                  'NAmes': 10, 'NPkVill': 11, 'NWAmes': 15,'NoRidge': 24,

                                  'NridgHt': 25, 'OldTown': 4,'SWISU': 9, 'Sawyer': 8,

                                  'SawyerW': 13,'Somerst': 21, 'StoneBr': 23,'Timber': 22,

                                  'Veenker': 20}})



train_data = train_data.replace({'BldgType': 

                                 {'1Fam': 5, '2fmCon': 1, 'Duplex': 2, 

                                  'TwnhsE': 4,'Twnhs': 3}})

test_data = test_data.replace({'BldgType': 

                                 {'1Fam': 5, '2fmCon': 1, 'Duplex': 2, 

                                  'TwnhsE': 4,'Twnhs': 3}})

train_data = train_data.replace({'Condition1': 

                                 {'Norm': 9, 'Feedr': 3, 'PosN': 7, 

                                  'Artery': 1, 'RRAe': 2, 'RRNn': 6, 

                                  'RRAn': 4, 'PosA': 8, 'RRNe': 5}})

test_data = test_data.replace({'Condition1': 

                                 {'Norm': 9, 'Feedr': 3, 'PosN': 7, 

                                  'Artery': 1, 'RRAe': 2, 'RRNn': 6, 

                                  'RRAn': 4, 'PosA': 8, 'RRNe': 5}})

train_data = train_data.replace({'HouseStyle': 

                                 {'2Story': 9, '1Story': 8, '1.5Fin': 7, 

                                  '1.5Unf': 1,'SFoyer': 2,'SLvl': 6,

                                  '2.5Unf': 3,'2.5Fin': 5}})

test_data = test_data.replace({'HouseStyle': 

                                 {'2Story': 9, '1Story': 8, '1.5Fin': 7, 

                                  '1.5Unf': 1,'SFoyer': 2,'SLvl': 6,

                                  '2.5Unf': 3,'2.5Fin': 5}})





train_data = train_data[train_data['GarageArea'] < 1200]



train_data['enc_street'] = pd.get_dummies(train_data.Street, drop_first=True)

test_data['enc_street'] = pd.get_dummies(test_data.Street, drop_first=True)



def encode(x): return 1 if x == 'Partial' else 0

train_data['enc_condition'] = train_data.SaleCondition.apply(encode)

test_data['enc_condition'] = test_data.SaleCondition.apply(encode)



data = train_data.select_dtypes(include=[np.number]).interpolate().dropna()

X = data.drop(['SalePrice', 'Id'], axis=1)



y = np.log(train_data.SalePrice)

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.33,random_state=42)



lm= linear_model.LinearRegression()

model = lm.fit(XTrain, yTrain)

print("R^2 is: \n", model.score(XTest, yTest))

# print("迴歸係數:", lm.coef_)

# print("截距:", lm.intercept_ )

# pred_train = lm.predict(XTrain)

# print(pred_train)



feats = test_data.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()

predictions = model.predict(feats)

pred_test = np.exp(predictions)

print(pred_test)



submit = pd.read_csv("../input/sample_submission.csv")

submit['SalePrice'] = pred_test

submit.to_csv('submit.csv', index= False)