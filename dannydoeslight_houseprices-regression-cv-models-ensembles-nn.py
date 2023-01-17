import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

# this does not help so we can remove it

df_train.drop(columns=['Id'], axis=1, inplace=True)

df_test.drop(columns=['Id'], axis=1, inplace=True)

# just save the shape for later to split train and test (see bellow)

n_train = df_train.shape[0]

n_test = df_test.shape[0]
y_train = df_train.SalePrice.values

# getting train and test data together for an easier workflow:

all_data = pd.concat((df_train, df_test), sort=True).reset_index(drop = True)

all_data.drop(['SalePrice'], axis = 1, inplace = True)
# getting rid of all columns that have more than 90% data missing

all_data = all_data.dropna(thresh=len(all_data)*0.9, axis=1)

# deleted: "Alley","PoolQC","MiscFeature","Fence","FireplaceQu" - though you can keep it if you desire a more complex model
# now will take care of the missing values and assign different dtypes to some variables, e.g.: that should be categorical variables not numerical(years, months).

all_data['GarageYrBlt']=all_data["GarageYrBlt"].fillna(1980)

missing_val_col0 = ['Electrical',

                    'SaleType',

                    'KitchenQual',

                    'Exterior1st',

                    'Exterior2nd',

                    'Functional',

                    'Utilities',

                    'MSZoning']

for i in missing_val_col0:

    all_data[i] = all_data[i].fillna(method='ffill')

all_data['MasVnrType']=all_data['MasVnrType'].fillna(0)

missing_val_col1 = ["GarageType",

                   "GarageFinish",

                   "GarageQual",

                   "GarageCond",

                   'BsmtQual',

                   'BsmtCond',

                   'BsmtExposure',

                   'BsmtFinType1',

                   'BsmtFinType2'] 

for i in missing_val_col1:

    all_data[i] = all_data[i].fillna('None')

missing_val_col2 = ['BsmtFinSF1',

                    'BsmtFinSF2',

                    'BsmtUnfSF',

                    'TotalBsmtSF',

                    'BsmtFullBath', 

                    'BsmtHalfBath',

                    'GarageArea',

                    'GarageCars',

                    'MasVnrArea']

for i in missing_val_col2:

    all_data[i] = all_data[i].fillna(0)
all_data["Functional"] = all_data["Functional"].fillna("Typ")

all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None").astype(str)

all_data['OverallCond'] = all_data['OverallCond'].astype(str) 

all_data['OverallQual'] = all_data['OverallQual'].astype(str)

all_data['YearBuilt'] = all_data['YearBuilt'].astype(int)

all_data['YearRemodAdd'] = all_data['YearRemodAdd'].astype(int)

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str) 
# and for whatever has been left out: 

NAcols = all_data.columns

for col in NAcols:

    if all_data[col].dtype == "object": # categorical values

        all_data[col] = all_data[col].fillna("None")

for col in NAcols:

    if all_data[col].dtype != "object": # numerical values

        all_data[col]= all_data[col].fillna(0)
# quick check to see if still missing values:

all_data.isnull().sum().sort_values(ascending=False).head()
# quick feature engineering

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF'] + all_data['GrLivArea'] + all_data['GarageArea']

all_data['Bathrooms'] = all_data['FullBath'] + all_data['HalfBath']*0.5 

all_data['Year average']= (all_data['YearRemodAdd']+all_data['YearBuilt'])/2
from scipy.stats import skew

# what one has to do for a normal distribution...

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data, drop_first=True)

train = all_data[:n_train]

test = all_data[n_train:]

y_train = np.log1p(y_train)

train.drop(index= train[(train.GrLivArea > 4600) & (train['MasVnrArea'] > 1500)].index.tolist(), inplace=True) #getting rid of outliers 
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train, y_train, test_size = .33, random_state = 0)
from sklearn.preprocessing import RobustScaler 

# which is pretty similar to StandardScaler, though is better for outliers

scaler= RobustScaler()

X_train = scaler.fit_transform(X_train)

X_val = scaler.transform(X_val)

test = scaler.transform(test)
from sklearn.metrics import accuracy_score, mean_squared_error

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor
def rmse(y_val, y_pred):

     return np.sqrt(mean_squared_error(y_val, y_pred)) 
randforest = RandomForestRegressor(n_estimators=300)

randforest.fit(X_train, y_train)
ridge = Ridge()

parameters = {'alpha':[x for x in range(0,101)]}



ridgeCV = GridSearchCV(ridge, param_grid=parameters, 

                       scoring='neg_mean_squared_error', cv=15)

ridgeCV.fit(X_train,y_train)



print("The best value of alpha is: ", ridgeCV.best_params_)

print(f'The best score, which is the mean of scores, achieved with {ridgeCV.best_params_} is: {np.sqrt(-ridgeCV.best_score_)} ')
ridgeCV = Ridge(alpha=8)

ridgeCV.fit(X_train, y_train)
lasso = Lasso()

parameters = {'alpha':[0.0001,0.0002,0.0003,0.0009,0.001,0.002,0.003,0.01,0.1,1,10,100], 

             } # 'max_iter':[2000, 3000, 5000]



lassoCV = GridSearchCV(lasso, param_grid=parameters, scoring='neg_mean_squared_error', cv=15)

lassoCV.fit(X_train,y_train)



print('The best value of alpha is: ',lassoCV.best_params_)

print("The best score achieved with alpha is: ", np.sqrt(-lassoCV.best_score_))
# I thought to give it a try to LassoCV which actually got me some better score than having CV on Lasso.

from sklearn.linear_model import LassoCV

reg = LassoCV(cv=5, random_state=0).fit(X_train, y_train)
gbr = GradientBoostingRegressor(n_estimators=4000, random_state=9, learning_rate=0.01, max_depth=5, 

                                max_features='sqrt', min_samples_leaf=6, min_samples_split=30, loss='huber')

gbr.fit(X_train, y_train)
xgbr = XGBRegressor(n_estimators=3500, learning_rate=0.01, max_depth=3, gamma=0.001, subsample=0.7,

                    colsample_bytree=0.7, objective='reg:linear', nthread=-1, seed=9, reg_alpha=0.0001) 

xgbr.fit(X_train, y_train)
print(f" randforest score on train set: {randforest.score(X_train, y_train)}")

print(f" randforest score on val. set: {randforest.score(X_val, y_val)}", "\n")

print(f" ridgeCV score on train set: {ridgeCV.score(X_train, y_train)}")

print(f" ridgeCV score on val. set: {ridgeCV.score(X_val, y_val)}", "\n")

print(f" lassoCV score on train set: {reg.score(X_train, y_train)}")

print(f" lassoCV score on val. set: {reg.score(X_val, y_val)}")

print(f" Number of features used by lassoCV: {np.sum(reg.coef_ != 0)}", "\n")

print(f" gbr score on train set: {gbr.score(X_train, y_train)}")

print(f" gbr score on val. set: {gbr.score(X_val, y_val)}", "\n")

print(f" xgbr score on train set: {xgbr.score(X_train, y_train)}")

print(f" xgbr score on val. set:  {xgbr.score(X_val, y_val)}", "\n")



y_pred = randforest.predict(X_val)

print(f"RandForest Root Mean Square Error validation = {rmse(y_val, y_pred)}")

y_pred_ridge = ridgeCV.predict(X_val)

print(f"RidgeCV Root Mean Square Error validation = {rmse(y_val, y_pred_ridge)}")

y_pred_CV = lassoCV.predict(X_val)

print(f"Simple Lasso Root Mean Square Error test = {rmse(y_val, y_pred_CV)}")

y_pred_reg = reg.predict(X_val)

print(f"LassoCV Root Mean Square Error test = {rmse(y_val, y_pred_reg)}")  # slightly better score

y_pred_gbr = gbr.predict(X_val)

print(f"GBR Root Mean Square Error test = {rmse(y_val, y_pred_gbr)}")     

y_pred_xgbr = xgbr.predict(X_val)

print(f"XGBR Root Mean Square Error test = {rmse(y_val, y_pred_xgbr)}")
# now let`s have a look at ensembles (keeping in mind that is always good to pick "diverse" models, including more 'risky' ones)

from sklearn.ensemble import VotingRegressor

voter = VotingRegressor([('Ridge', ridgeCV), ('XGBRegressor', xgbr), ('GradientBoostingRegressor', gbr)]) #('XGBRegressor', xgbr) ('GradientBoostingRegressor', gbr)

voter.fit(X_train, y_train.ravel())

y_pred_voter = voter.predict(X_val)

print(f"Root Mean Square Error test = {rmse(y_val, y_pred_voter)}")
from mlxtend.regressor import StackingRegressor

stacker = StackingRegressor(regressors = [ridge, xgbr, gbr], 

                           meta_regressor = voter, use_features_in_secondary=True) #voting had the best score

stacker.fit(X_train, y_train.ravel())

y_pred_stacker = stacker.predict(X_val)

print(f"Root Mean Square Error test = {rmse(y_val, y_pred_stacker)}")
final_validation = (0.2*y_pred_voter + 0.4*y_pred_stacker + 0.4*y_pred_xgbr )



print(f"Root Mean Square Error test = {rmse(y_val, final_validation)}")
from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.callbacks import EarlyStopping

from keras import optimizers
n_cols = train.shape[1]

early_stopping_monitor = EarlyStopping(patience=3)
model = Sequential()

model.add(Dense(4, activation='elu', input_shape = (n_cols,))) #, kernel_initializer='normal'

model.add(Dense(2, activation='elu'))

model.add(Dense(1, activation='linear', kernel_regularizer = 'l2',

                kernel_initializer='normal'))
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=300, batch_size=18

                       , callbacks=[early_stopping_monitor])
import matplotlib.pyplot as plt

print(history.history.keys())

# "Loss"

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper right')

plt.show()
score = model.evaluate(X_val, y_val, batch_size=32, verbose=1)

print('Test score:', score)