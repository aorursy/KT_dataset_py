import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt



import tensorflow as tf



from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import RepeatedKFold

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, IsolationForest

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import ExtraTreeRegressor, DecisionTreeRegressor

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.metrics import mean_squared_log_error

from sklearn.svm import SVR

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor



import warnings



warnings.filterwarnings('ignore')
def preprocess_street(x):

    return x.strip()=='Pave'



def preprocess_centralair(x):

    return x.strip()=='Y'



def preprocess_alley(x):

    if pd.isna(x):

        return 0

    if x.strip()=='Grvl':

        return -1

    

    return 1



landslope_map = {'Gtl': 0, 'Mod':1, 'Sev': 2}



paveddrive_map = {'N': 0, 'P': 1, 'Y': 2}



lotshape_map = {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3}



landcontour_map = {'Low': 0, 'HLS': 1, 'Bnk': 2, 'Lvl': 3}



general_map = {np.nan: -1, 'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}



garagefinish_map = {np.nan: 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}



bsmtexposure_map = {np.nan: 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}



bsmtfintype_map = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, np.nan: 0}



functional_map = {'Typ': 8, 'Min1': 7, 'Min2': 6, 'Mod': 5, 'Maj1': 4, 'Maj2': 3, 'Sev': 2, 'Sal': 1}



def preprocess_test(test):

    df = test.copy()

    df['Street'] = df['Street'].apply(preprocess_street)

    df.drop('Utilities', axis=1, inplace=True)

    df['CentralAir'] = df['CentralAir'].apply(preprocess_centralair)

    df['Alley'] = df['Alley'].apply(preprocess_alley)

    df['LandSlope'] = df['LandSlope'].map(landslope_map)

    df['PavedDrive'] = df['PavedDrive'].map(paveddrive_map)

    df['LotShape'] = df['LotShape'].map(lotshape_map)

    df['FireplaceQu'] = df['FireplaceQu'].map(general_map)

    df['HeatingQC'] = df['HeatingQC'].map(general_map)

    df['BsmtCond'] = df['BsmtCond'].map(general_map)

    df['BsmtQual'] = df['BsmtQual'].map(general_map)

    df['BsmtExposure'] = df['BsmtExposure'].map(bsmtexposure_map)

    df['PoolQC'] = df['PoolQC'].map(general_map)

    df['GarageQual'] = df['GarageQual'].map(general_map)

    df['GarageCond'] = df['GarageCond'].map(general_map)

    df['GarageFinish'] = df['GarageFinish'].map(garagefinish_map)

    df['KitchenQual'] = df['KitchenQual'].map(general_map)

    df['ExterCond'] = df['ExterCond'].map(general_map)

    df['ExterQual'] = df['ExterQual'].map(general_map)

    df['LandContour'] = df['LandContour'].map(landcontour_map)

    df['BsmtFinType1'] = df['BsmtFinType1'].map(bsmtfintype_map)

    df['BsmtFinType2'] = df['BsmtFinType2'].map(bsmtfintype_map)

    df['Functional'] = df['Functional'].map(functional_map)

    df = pd.concat([df.drop('SaleCondition', axis=1), pd.get_dummies(df['SaleCondition'], prefix='SaleCondition')], axis=1)

    numerical_cols = []

    for col in df.columns:

        if col!='Id' and df[col].dtype!='object':

            numerical_cols.append(col)

            

    return np.array(df[numerical_cols].fillna(-999))





def evaluate_model(model, X_train, y_train, rkf, y_scaler):

    rmsles = []

    models = []

    for train_idx, val_idx in rkf.split(X_train):

        model.fit(X_train[train_idx], y_train[train_idx])

        y_pred = model.predict(X_train[val_idx])

        y_pred = y_scaler.inverse_transform(y_pred.reshape(-1,1)).clip(0, np.inf)

        y_true = y_scaler.inverse_transform(y_train[val_idx].reshape(-1,1))

        rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))

        rmsles.append(rmsle)

        models.append(model)

        

    print("RMSLE: {:.5f} +- {:.5f}".format(np.mean(rmsles), np.std(rmsles)))

    return models
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

sample = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
train.head()
train.info()
categocial_features = []

for col in train.columns:

    if train[col].dtype=='object':

        categocial_features.append((col, len(train[col].unique())))

        

sorted(categocial_features, key=lambda k: k[1])
train['Street'].value_counts()
train['Street'].isna().sum()
train['Street'] = train['Street'].apply(preprocess_street)
train['Utilities'].value_counts()
train['Utilities'].isna().sum()
train.drop('Utilities', axis=1, inplace=True)
train['CentralAir'].value_counts()
train['CentralAir'].isna().sum()
train['CentralAir'] = train['CentralAir'].apply(preprocess_centralair)
train['Alley'].value_counts()
train['Alley'].isna().sum()
sns.boxplot(x=train['Alley'].fillna('NaN'), y=train['SalePrice'])
train['Alley'] = train['Alley'].apply(preprocess_alley)
train['LandSlope'].value_counts()
train['LandSlope'] = train['LandSlope'].map(landslope_map)
train['PavedDrive'].value_counts()
train['PavedDrive'] = train['PavedDrive'].map(paveddrive_map)
train['LotShape'].value_counts()
train['LotShape'] = train['LotShape'].map(lotshape_map)
train['LandContour'].value_counts()
train['LandContour'] = train['LandContour'].map(landcontour_map)
train['ExterQual'].value_counts()
train['ExterQual'] = train['ExterQual'].map(general_map)
train['ExterCond'].value_counts()
train['ExterCond'] = train['ExterCond'].map(general_map)
train['KitchenQual'] = train['KitchenQual'].map(general_map)
train['GarageFinish'].value_counts()
train['GarageFinish'].isna().sum()
train['GarageFinish'] = train['GarageFinish'].map(garagefinish_map)
train['GarageCond'].value_counts()
train['GarageCond'] = train['GarageCond'].map(general_map)
train['GarageQual'].value_counts()
train['GarageQual'] = train['GarageQual'].map(general_map)
train['PoolQC'].value_counts()
train['PoolQC'] = train['PoolQC'].map(general_map)
train['BsmtQual'].value_counts()
train['BsmtQual'] = train['BsmtQual'].map(general_map)
train['BsmtCond'].value_counts()
train['BsmtCond'] = train['BsmtCond'].map(general_map)
train['BsmtExposure'].value_counts()
train['BsmtExposure'] = train['BsmtExposure'].map(bsmtexposure_map)
train['HeatingQC'].value_counts()
train['HeatingQC'] = train['HeatingQC'].map(general_map)
train['FireplaceQu'].value_counts()
train['FireplaceQu'] = train['FireplaceQu'].map(general_map)
train['SaleCondition'].value_counts()
train['SaleCondition'].isna().sum()
sns.boxplot(y='SalePrice', x='SaleCondition', data=train)
train = pd.concat([train.drop('SaleCondition', axis=1), pd.get_dummies(train['SaleCondition'], prefix='SaleCondition')], axis=1)
train['BsmtFinType1'] = train['BsmtFinType1'].map(bsmtfintype_map)

train['BsmtFinType2'] = train['BsmtFinType2'].map(bsmtfintype_map)
train['Functional'] = train['Functional'].map(functional_map)
import scipy.cluster.hierarchy as spc



corr = train.corr().values



pdist = spc.distance.pdist(corr)

linkage = spc.linkage(pdist, method='complete')

idx = spc.fcluster(linkage, 0.5 * pdist.max(), 'distance')



col_to_cluster = {}



for i,col in enumerate(train.corr().columns):

    

    col_to_cluster[col] = idx[i]
fig, ax = plt.subplots(figsize=[20,20])

sns.heatmap(train[sorted(col_to_cluster, key=lambda k: col_to_cluster[k])].corr(),

            annot=True, cbar=False, cmap='Blues', fmt='.1f')
numerical_cols = []

for col in train.columns:

    if col!='Id' and train[col].dtype!='object' and col!='SalePrice':

        numerical_cols.append(col)
X_df = train[numerical_cols].fillna(-999)
X_train = np.array(X_df)

X_test = preprocess_test(test)
X_scaler = StandardScaler()

X_train = X_scaler.fit_transform(X_train)

X_test = X_scaler.transform(X_test)
y_train = np.array(train['SalePrice'], ndmin=2).reshape(-1,1)
y_scaler = MinMaxScaler()

y_train = y_scaler.fit_transform(y_train)
rkf = RepeatedKFold(n_splits=6, n_repeats=5)
def get_model():

    nn_model = tf.keras.Sequential([

        tf.keras.layers.Dense(units=256, activation='relu'),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(0.1),

        tf.keras.layers.Dense(units=128, activation='relu'),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(0.1),

        tf.keras.layers.Dense(units=1, activation='sigmoid')

    ])



    nn_model.compile(optimizer='adam', loss='msle')

    return nn_model
reg_models = {

    'RF': RandomForestRegressor(n_estimators=250),

    'LGB': LGBMRegressor(n_estimators=200),

    'XGB': XGBRegressor(n_estimators=200, objective='reg:squarederror'),

    'ADA': AdaBoostRegressor(n_estimators=250),

    'KNN': KNeighborsRegressor(n_neighbors=7)

}
all_models = []

for model in reg_models:

    print(model)

    models = evaluate_model(reg_models[model], X_train, y_train, rkf, y_scaler)

    all_models += models
rmsles = []

nn_models = []

for train_idx, val_idx in rkf.split(X_train):

    nn_model = get_model()

    nn_model.fit(X_train[train_idx], y_train[train_idx], epochs=30, verbose=0)

    y_pred = nn_model.predict(X_train[val_idx])

    y_pred = y_scaler.inverse_transform(y_pred.reshape(-1,1)).clip(0, np.inf)

    y_true = y_scaler.inverse_transform(y_train[val_idx].reshape(-1,1))

    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))

    rmsles.append(rmsle)

    nn_models.append(nn_model)

    

print("RMSLE: {:.5f} +- {:.5f}".format(np.mean(rmsles), np.std(rmsles)))
all_models += nn_models
X_train_predictions = np.zeros(shape=(X_train.shape[0], len(all_models)))

for i, model in enumerate(all_models):

    X_train_predictions[:, i] = model.predict(X_train).reshape(-1)

    

X_test_predictions = np.zeros(shape=(X_test.shape[0], len(all_models)))

for i, model in enumerate(all_models):

    X_test_predictions[:, i] = model.predict(X_test).reshape(-1)
models = []

for train_idx, val_idx in rkf.split(X_train_predictions):

    meta_regressor = LinearRegression()

    meta_regressor.fit(X_train_predictions[train_idx], y_train[train_idx])

    y_pred = meta_regressor.predict(X_train_predictions[val_idx])

    y_pred = y_scaler.inverse_transform(y_pred.reshape(-1,1))

    y_true = y_scaler.inverse_transform(y_train[val_idx].reshape(-1,1))

    print("rmsle = {}".format(np.sqrt(mean_squared_log_error(y_true, y_pred))))

    models.append(meta_regressor)
y_pred = np.array([model.predict(X_test_predictions) for model in models]).mean(axis=0).reshape(-1,1)
y_pred = y_scaler.inverse_transform(y_pred)
sample['SalePrice'] = y_pred
sample.to_csv('sample_submission.csv', index=False)