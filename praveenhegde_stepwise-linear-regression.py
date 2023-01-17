import pandas as pd

import numpy as np



from sklearn import linear_model

from sklearn import cross_validation

from sklearn import preprocessing



train_df = pd.read_csv('../input/train.csv', header=0)

test_df = pd.read_csv('../input/test.csv', header=0)

train_y = train_df['SalePrice']

test_id = test_df['Id']



train_df.drop('Id',axis=1,inplace=True)

train_df.drop('SalePrice',axis=1,inplace=True)

test_df.drop('Id',axis=1,inplace=True)



features = train_df.columns.values

consider = []

numerics = ['LotArea','LotFrontage','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath',

                'BsmtHalfBath','FullBath','HalfBath','Bedroom','Kitchen','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF',

                'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YrSold']

i=1

for col in features:

        print("Including "+str(i)+": "+str(col))

        i+=1

        consider.append(col)

        if col not in numerics:

                le = preprocessing.LabelEncoder()

                train_df[col] = train_df[col].fillna("Missing")

                train_df[col] = le.fit_transform(train_df[col])

        else:

                train_df[col] = train_df[col].fillna(0)



        train = train_df[consider]

        X = train.as_matrix()

        Y = train_y.as_matrix()



        log_regressor = linear_model.LinearRegression()

        score2 = cross_validation.cross_val_score(log_regressor,X,Y,cv=5)

        print ("lin_reg: mean="+str(np.mean(score2))+" std="+str(np.std(score2)))