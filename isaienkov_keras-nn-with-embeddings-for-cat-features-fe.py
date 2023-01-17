import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



np.random.seed(666)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

from keras.models import Model, load_model

from keras.layers import Input, Dropout, Dense, Embedding, SpatialDropout1D, concatenate, BatchNormalization, Flatten

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing import text, sequence

from keras.callbacks import Callback

from keras import backend as K

from keras.models import Model

from keras.losses import mean_squared_error as mse_loss

from keras import optimizers

from keras.optimizers import RMSprop, Adam

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import matplotlib.pyplot as plt
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test =  pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
categorical = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 

               'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 

               'CentralAir', 'Electrical', 'Functional', 'GarageType', 'SaleType',  'SaleCondition']

numerical = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'ExterQual', 'ExterCond', 'BsmtQual', 

             'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'HeatingQC', '1stFlrSF', 

             '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',

            'TotRmsAbvGrd', 'Fireplaces', 'FireplaceQu', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive',

            'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscVal', 'YrSold']
f = plt.figure(figsize=(19, 15))

plt.matshow(train.corr(), fignum=f.number)

plt.xticks(range(train.shape[1]), train.columns, fontsize=14, rotation=45)

plt.yticks(range(train.shape[1]), train.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)
train['MSZoning'] = train['MSZoning'].replace('C (all)', np.nan)

test['MSZoning'] = test['MSZoning'].replace('C (all)', np.nan)

train['Utilities'] = train['Utilities'].replace('NoSeWa', np.nan)

train['Condition2'] = train['Condition2'].replace('RRNn', np.nan)

train['Condition2'] = train['Condition2'].replace('Artery', np.nan)

train['Condition2'] = train['Condition2'].replace('PosN', np.nan)

train['Condition2'] = train['Condition2'].replace('PosA', np.nan)

train['Condition2'] = train['Condition2'].replace('RRAe', np.nan)

train['Condition2'] = train['Condition2'].replace('RRAn', np.nan)

test['Condition2'] = test['Condition2'].replace('RRNn', np.nan)

test['Condition2'] = test['Condition2'].replace('Artery', np.nan)

test['Condition2'] = test['Condition2'].replace('PosN', np.nan)

test['Condition2'] = test['Condition2'].replace('PosA', np.nan)

test['Condition2'] = test['Condition2'].replace('RRAe', np.nan)

test['Condition2'] = test['Condition2'].replace('RRAn', np.nan)

train['HouseStyle'] = train['HouseStyle'].replace('1.5Unf', np.nan)

train['HouseStyle'] = train['HouseStyle'].replace('2.5Unf', np.nan)

train['HouseStyle'] = train['HouseStyle'].replace('2.5Fin', np.nan)

test['HouseStyle'] = test['HouseStyle'].replace('1.5Unf', np.nan)

test['HouseStyle'] = test['HouseStyle'].replace('2.5Unf', np.nan)

test['HouseStyle'] = test['HouseStyle'].replace('2.5Fin', np.nan)

train['RoofMatl'] = train['RoofMatl'].replace('ClyTile', np.nan)

train['RoofMatl'] = train['RoofMatl'].replace('Metal', np.nan)

train['RoofMatl'] = train['RoofMatl'].replace('Roll', np.nan)

train['RoofMatl'] = train['RoofMatl'].replace('Membran', np.nan)

train['RoofMatl'] = train['RoofMatl'].replace('WdShngl', np.nan)

test['RoofMatl'] = test['RoofMatl'].replace('WdShngl', np.nan)

train['Exterior1st'] = train['Exterior1st'].replace('Stone', np.nan)

train['Exterior1st'] = train['Exterior1st'].replace('ImStucc', np.nan)

train['Exterior2nd'] = train['Exterior2nd'].replace('Other', np.nan)

train['Heating'] = train['Heating'].replace('Floor', np.nan)

train['Heating'] = train['Heating'].replace('OthW', np.nan)

train['Heating'] = train['Heating'].replace('Wall', np.nan)

test['Heating'] = test['Heating'].replace('Wall', np.nan)

train['Electrical'] = train['Electrical'].replace('FuseP', np.nan)

train['Electrical'] = train['Electrical'].replace('Mix', np.nan)

test['Electrical'] = test['Electrical'].replace('FuseP', np.nan)

train['Functional'] = train['Functional'].replace('Sev', np.nan)

test['Functional'] = test['Functional'].replace('Sev', np.nan)

train['SaleType'] = train['SaleType'].replace('Con', np.nan)

test['SaleType'] = test['SaleType'].replace('Con', np.nan)
quality = {

    'Ex': 5,

    'Gd': 4,

    'TA': 3,	

    'Fa': 2,	

    'Po': 1,

    'NA': 0

}



exposure = { 

    'Gd': 5,

    'Av' : 4,

    'Mn' : 3,	

    'No': 2,

    'NA': 0

}



garage = {

    'Fin' : 3,

    'RFn' : 2,	

    'Unf' : 1,

    'NA' : 0

}



pavedrive = {

       'Y': 1, 

       'P':	0.5,

       'N':	0,

}



fence = {

    'GdPrv': 5,

    'MnPrv': 4,

    'GdWo': 3,

    'MnWw': 2,

    'NA': 0

}
train['OveralRate'] = train['OverallQual'] * train['OverallCond']

test['OveralRate'] = test['OverallQual'] * test['OverallCond']

train['LastRemodeling'] = train[['YearBuilt', 'YearRemodAdd']].max(axis=1)

test['LastRemodeling'] = test[['YearBuilt', 'YearRemodAdd']].max(axis=1)

for item in quality:

    train['ExterQual'] = train['ExterQual'].replace(item, quality[item])

    test['ExterQual'] = test['ExterQual'].replace(item, quality[item])

    train['ExterCond'] = train['ExterCond'].replace(item, quality[item])

    test['ExterCond'] = test['ExterCond'].replace(item, quality[item])

train['ExterRate'] = (train['ExterQual'] + train['ExterCond'])/2

test['ExterRate'] = (test['ExterQual'] + test['ExterCond'])/2

for item in quality:

    train['BsmtQual'] = train['BsmtQual'].replace(item, quality[item])

    test['BsmtQual'] = test['BsmtQual'].replace(item, quality[item])

    train['BsmtCond'] = train['BsmtCond'].replace(item, quality[item])

    test['BsmtCond'] = test['BsmtCond'].replace(item, quality[item])

for item in exposure:

    train['BsmtExposure'] = train['BsmtExposure'].replace(item, exposure[item])

    test['BsmtExposure'] = test['BsmtExposure'].replace(item, exposure[item])

train['BsmtRate'] = (train['BsmtQual'] + train['BsmtCond'] + train['BsmtExposure'])/3

test['BsmtRate'] = (test['BsmtQual'] + test['BsmtCond'] + train['BsmtExposure'])/3

train['BsmtNumber'] = train['BsmtFinSF1'].apply(lambda x: min(x, 1)) + train['BsmtFinSF2'].apply(lambda x: min(x, 1))

test['BsmtNumber'] = test['BsmtFinSF1'].apply(lambda x: min(x, 1)) + test['BsmtFinSF2'].apply(lambda x: min(x, 1))

for item in quality:

    train['HeatingQC'] = train['HeatingQC'].replace(item, quality[item])

    test['HeatingQC'] = test['HeatingQC'].replace(item, quality[item])

    train['KitchenQual'] = train['KitchenQual'].replace(item, quality[item])

    test['KitchenQual'] = test['KitchenQual'].replace(item, quality[item])

    train['FireplaceQu'] = train['FireplaceQu'].replace(item, quality[item])

    test['FireplaceQu'] = test['FireplaceQu'].replace(item, quality[item])

for item in garage:

    train['GarageFinish'] = train['GarageFinish'].replace(item, garage[item])

    test['GarageFinish'] = test['GarageFinish'].replace(item, garage[item])

for item in quality:

    train['GarageQual'] = train['GarageQual'].replace(item, quality[item])

    test['GarageQual'] = test['GarageQual'].replace(item, quality[item])

    train['GarageCond'] = train['GarageCond'].replace(item, quality[item])

    test['GarageCond'] = test['GarageCond'].replace(item, quality[item])

train['GarageRate'] = (train['GarageFinish'] + train['GarageQual'] + train['GarageCond'])/3

test['GarageRate'] = (test['GarageFinish'] + test['GarageQual'] + test['GarageCond'])/3

for item in quality:

    train['PoolQC'] = train['PoolQC'].replace(item, quality[item])

    test['PoolQC'] = test['PoolQC'].replace(item, quality[item])

    train['Fence'] = train['Fence'].replace(item, quality[item])

    test['Fence'] = test['Fence'].replace(item, quality[item])

for item in pavedrive:

    train['PavedDrive'] = train['PavedDrive'].replace(item, pavedrive[item])

    test['PavedDrive'] = test['PavedDrive'].replace(item, pavedrive[item])

for item in fence:

    train['Fence'] = train['Fence'].replace(item, fence[item])

    test['Fence'] = test['Fence'].replace(item, fence[item])
categorical = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 

               'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation',

               'Heating', 'CentralAir', 'Electrical', 'Functional', 'GarageType', 'SaleType',  'SaleCondition']

numerical = ['LotFrontage', 'LotArea', 'OveralRate', 'LastRemodeling', 'MasVnrArea', 'ExterRate', 'BsmtRate', 'BsmtNumber', 'BsmtUnfSF', 'TotalBsmtSF', 

             'HeatingQC', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 

             'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Fireplaces', 'FireplaceQu', 'GarageYrBlt', 'GarageRate', 'GarageArea',  'PavedDrive',

             'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscVal']
features = categorical + numerical
target = train['SalePrice']

train = train[features]

test = test[features]
target = np.log1p(target)
from sklearn.preprocessing import LabelEncoder



for col in categorical:

    print(col)

    le = LabelEncoder()

    train[col] = train[col].astype(str)

    test[col] = test[col].astype(str)

    train[col] = le.fit_transform(train[col])

    test[col] = le.transform(test[col])

train = train.fillna(0)

test = test.fillna(0)
def model(dense_dim_1=64, dense_dim_2=32, dense_dim_3=32, dense_dim_4=16, 

dropout1=0.2, dropout2=0.1, dropout3=0.1, dropout4=0.1, lr=0.001):

    

    #Inputs

    input1 = Input(shape=[1], name="MSZoning")

    input2 = Input(shape=[1], name='Street') 

    input3 = Input(shape=[1], name='LotShape')

    input4 = Input(shape=[1], name='LandContour')

    input5 = Input(shape=[1], name='Utilities')

    input6 = Input(shape=[1], name='LotConfig')

    input7 = Input(shape=[1], name='LandSlope')

    input8 = Input(shape=[1], name='Neighborhood')

    input9 = Input(shape=[1], name= 'Condition1')

    input10 = Input(shape=[1], name= 'Condition2')

    input11 = Input(shape=[1], name='BldgType')

    input12 = Input(shape=[1], name='HouseStyle')

    input13 = Input(shape=[1], name='RoofStyle')

    input14 = Input(shape=[1], name='RoofMatl')

    input15 = Input(shape=[1], name='Exterior1st')

    input16 = Input(shape=[1], name='Exterior2nd')

    input17 = Input(shape=[1], name='MasVnrType')

    input18 = Input(shape=[1], name='Foundation')

    input19 = Input(shape=[1], name='Heating')

    input20 = Input(shape=[1], name='CentralAir')

    input21 = Input(shape=[1], name='Electrical')

    input22 = Input(shape=[1], name='Functional')

    input23 = Input(shape=[1], name='GarageType')

    input24 = Input(shape=[1], name='SaleType')

    input25 = Input(shape=[1], name='SaleCondition')

    input26 = Input(shape=[1], name='LotFrontage')

    input27 = Input(shape=[1], name='LotArea')

    input28 = Input(shape=[1], name='OveralRate')

    input29 = Input(shape=[1], name='LastRemodeling')

    input30 = Input(shape=[1], name='MasVnrArea')

    input31 = Input(shape=[1], name='ExterRate')

    input32 = Input(shape=[1], name='BsmtRate')

    input33 = Input(shape=[1], name='BsmtNumber')

    input34 = Input(shape=[1], name='BsmtUnfSF')

    input35 = Input(shape=[1], name='TotalBsmtSF')

    input36 = Input(shape=[1], name='HeatingQC')

    input37 = Input(shape=[1], name='1stFlrSF')

    input38 = Input(shape=[1], name='2ndFlrSF')

    input39 = Input(shape=[1], name='LowQualFinSF')

    input40 = Input(shape=[1], name='GrLivArea')

    input41 = Input(shape=[1], name='BsmtFullBath')

    input42 = Input(shape=[1], name='BsmtHalfBath')

    input43 = Input(shape=[1], name='FullBath')

    input44 = Input(shape=[1], name='HalfBath')

    input45 = Input(shape=[1], name='BedroomAbvGr')

    input46 = Input(shape=[1], name='KitchenAbvGr')

    input47 = Input(shape=[1], name='KitchenQual')

    input48 = Input(shape=[1], name='TotRmsAbvGrd')

    input49 = Input(shape=[1], name='Fireplaces')

    input50 = Input(shape=[1], name='FireplaceQu')

    input51 = Input(shape=[1], name='GarageYrBlt')

    input52 = Input(shape=[1], name='GarageRate')

    input53 = Input(shape=[1], name='GarageArea')

    input54 = Input(shape=[1], name='PavedDrive')

    input55 = Input(shape=[1], name='WoodDeckSF')

    input56 = Input(shape=[1], name='OpenPorchSF')

    input57 = Input(shape=[1], name='EnclosedPorch')

    input58 = Input(shape=[1], name='3SsnPorch')

    input59 = Input(shape=[1], name='ScreenPorch')

    input60 = Input(shape=[1], name='PoolArea')

    input61 = Input(shape=[1], name='PoolQC')

    input62 = Input(shape=[1], name='Fence')

    input63 = Input(shape=[1], name='MiscVal')



    #Embeddings layers

    emb1 = Embedding(5, 2)(input1)

    emb2 = Embedding(2, 1)(input2)

    emb3 = Embedding(4, 2)(input3)

    emb4 = Embedding(4, 2)(input4)

    emb5 = Embedding(2, 1)(input5)

    emb6 = Embedding(5, 2)(input6)

    emb7 = Embedding(3, 1)(input7)

    emb8 = Embedding(25, 12)(input8)

    emb9 = Embedding(9, 4)(input9)

    emb10 = Embedding(3, 1)(input10)

    emb11 = Embedding(5, 2)(input11)

    emb12 = Embedding(6, 3)(input12)

    emb13 = Embedding(6, 3)(input13)

    emb14 = Embedding(4, 2)(input14)

    emb15 = Embedding(14, 7)(input15)

    emb16 = Embedding(16, 8)(input16)

    emb17 = Embedding(5, 2)(input17)

    emb18 = Embedding(6, 3)(input18)

    emb19 = Embedding(4, 2)(input19)

    emb20 = Embedding(2, 1)(input20)

    emb21 = Embedding(4, 2)(input21)

    emb22 = Embedding(7, 3)(input22)

    emb23 = Embedding(7, 3)(input23)

    emb24 = Embedding(9, 4)(input24)

    emb25 = Embedding(6, 3)(input25)



    concat_emb = concatenate([

           Flatten() (emb1)

         , Flatten() (emb2)

         , Flatten() (emb3)

         , Flatten() (emb4)

         , Flatten() (emb5)

         , Flatten() (emb6)

         , Flatten() (emb7)

         , Flatten() (emb8)

         , Flatten() (emb9)

         , Flatten() (emb10)

         , Flatten() (emb11)

         , Flatten() (emb12)

         , Flatten() (emb13)

         , Flatten() (emb14)

         , Flatten() (emb15)

         , Flatten() (emb16)

         , Flatten() (emb17)

         , Flatten() (emb18)

         , Flatten() (emb19)

         , Flatten() (emb20)

         , Flatten() (emb21)

         , Flatten() (emb22)

         , Flatten() (emb23)

         , Flatten() (emb24)

         , Flatten() (emb25)

    ])

    

    categ = Dropout(dropout1)(Dense(dense_dim_1,activation='relu') (concat_emb))

    categ = BatchNormalization()(categ)

    categ = Dropout(dropout2)(Dense(dense_dim_2,activation='relu') (categ))

    

    #main layer

    main_l = concatenate([

          categ

        , input26

        , input27

        , input28

        , input29

        , input30

        , input31

        , input32

        , input33

        , input34

        , input35

        , input36

        , input37

        , input38

        , input39

        , input40

        , input41

        , input42

        , input43

        , input44

        , input45

        , input46

        , input47

        , input48

        , input49

        , input50

        , input51

        , input52

        , input53

        , input54

        , input55

        , input56

        , input57

        , input58

        , input59

        , input60

        , input61

        , input62

        , input63

    ])

    

    main_l = Dropout(dropout3)(Dense(dense_dim_3,activation='relu') (main_l))

    main_l = BatchNormalization()(main_l)

    main_l = Dropout(dropout4)(Dense(dense_dim_4,activation='relu') (main_l))

    

    #output

    output = Dense(1) (main_l)



    model = Model([ input1, 

                    input2,

                    input3,

                    input4, 

                    input5,  

                    input6, 

                    input7, 

                    input8,

                    input9,

                    input10,

                    input11, 

                    input12,

                    input13,

                    input14, 

                    input15,  

                    input16, 

                    input17, 

                    input18,

                    input19,

                    input20,

                    input21, 

                    input22,

                    input23,

                    input24, 

                    input25,  

                    input26, 

                    input27, 

                    input28,

                    input29,

                    input30,

                    input31, 

                    input32,

                    input33,

                    input34, 

                    input35,  

                    input36, 

                    input37, 

                    input38,

                    input39,

                    input40,

                    input41, 

                    input42,

                    input43,

                    input44, 

                    input45,  

                    input46, 

                    input47, 

                    input48,

                    input49,

                    input50,

                    input51, 

                    input52,

                    input53,

                    input54, 

                    input55,  

                    input56, 

                    input57, 

                    input58,

                    input59,

                    input60,

                    input61, 

                    input62,

                    input63], output)



    model.compile(optimizer = Adam(lr=lr),

                  loss= mse_loss,

                  metrics=[root_mean_squared_error])

    return model



def root_mean_squared_error(y_true, y_pred):

    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))
def get_keras_data(df, num_cols, cat_cols):

    cols = cat_cols + num_cols

    X = {col: np.array(df[col]) for col in cols}

    return X



def train_model(keras_model, X_t, y_train, batch_size, epochs, X_v, y_valid, fold, patience=3):

    early_stopping = EarlyStopping(patience=patience, verbose=1)

    model_checkpoint = ModelCheckpoint("model_" + str(fold) + ".hdf5",

                                       save_best_only=True, verbose=1, monitor='val_root_mean_squared_error', mode='min')



    hist = keras_model.fit(X_t, y_train, batch_size=batch_size, epochs=epochs,

                            validation_data=(X_v, y_valid), verbose=1,

                            callbacks=[early_stopping, model_checkpoint])



    keras_model = load_model("model_" + str(fold) + ".hdf5", custom_objects={'root_mean_squared_error': root_mean_squared_error})

    

    return keras_model
from sklearn.model_selection import KFold





oof = np.zeros(len(train))

batch_size = 32

epochs = 50

models = []



folds = 2

seed = 666



kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

fold_n = 0



for train_index, valid_index in kf.split(train):

    print('Fold:', fold_n)

    X_train, X_valid = train.iloc[train_index], train.iloc[valid_index]

    y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]

    X_t = get_keras_data(X_train, numerical, categorical)

    X_v = get_keras_data(X_valid, numerical, categorical)

    

    keras_model = model(dense_dim_1=100, dense_dim_2=50, dense_dim_3=50, dense_dim_4=20, 

                        dropout1=0.2, dropout2=0.1, dropout3=0.1, dropout4=0.1, lr=0.005)

    mod = train_model(keras_model, X_t, y_train, batch_size, epochs, X_v, y_valid, fold_n, patience=15)

    models.append(mod)

    print('*'* 50)

    fold_n += 1
for_prediction = get_keras_data(test, numerical, categorical)

res = (sum(np.expm1([model.predict(for_prediction)[:,0] for model in models])))/2
submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

submission['SalePrice'] = res

submission.loc[submission['SalePrice']<0, 'SalePrice'] = 0

submission.loc[submission['SalePrice']>10000000, 'SalePrice'] = 10000000

submission.to_csv('submission.csv', index=False, float_format='%.4f')

submission