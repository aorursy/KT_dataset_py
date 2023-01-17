import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('ggplot')



seed = 123456

np.random.seed(seed)
target_variable = 'saleprice'

df = (

    pd.read_csv('../input/train.csv') # change this to run on kaggle

    #pd.read_csv('../input/train.csv')



    # Rename columns to lowercase and underscores

    .pipe(lambda d: d.rename(columns={

        k: v for k, v in zip(

            d.columns,

            [c.lower().replace(' ', '_') for c in d.columns]

        )

    }))

    # Switch categorical classes to integers

    #.assign(**{target_variable: lambda r: r[target_variable].astype('category').cat.codes})

)

print('Done')
# Categorical / Ordinal

# using binary coding according to this artical 

# to encode categoricals 

# http://www.kdnuggets.com/2015/12/beyond-one-hot-exploration-categorical-variables.html

# class 4 = x0100 = 0 | 1 | 0 | 0 in columns

# use this encoder https://github.com/scikit-learn-contrib/categorical-encoding

categorical_vars = ['MSZoning'

                    ,'Street'

                    ,'Alley'

                    ,'LotShape'

                    ,'LandContour'

                    ,'Utilities'

                    ,'LotConfig'

                    ,'LandSlope'

                    ,'Neighborhood'

                    ,'Condition1'

                    ,'Condition2'

                    ,'BldgType'

                    ,'HouseStyle'

                    ,'RoofStyle'

                    ,'RoofMatl'

                    ,'Exterior1st'

                    ,'Exterior2nd'

                    ,'MasVnrType'

                    ,'ExterQual'

                    ,'ExterCond'

                    ,'Foundation'

                    ,'BsmtQual'

                    ,'BsmtCond'

                    ,'BsmtExposure'

                    ,'BsmtFinType1'

                    ,'BsmtFinSF1'

                    ,'BsmtFinType2'

                    ,'Heating'

                    ,'HeatingQC'

                    ,'CentralAir'

                    ,'Electrical'

                    ,'KitchenQual'

                    ,'Functional'

                    ,'FireplaceQu'

                    ,'GarageType'

                    ,'GarageFinish'

                    ,'GarageQual'

                    ,'GarageCond'

                    ,'PavedDrive'

                    ,'PoolQC'

                    ,'Fence'

                    ,'MiscFeature'

                    ,'SaleType'

                    ,'SaleCondition'

]



# Nominal:

# Could try binary encoding these too

nominal_vars = ['OverallQual'

                ,'OverallCond'

                ,'YearBuilt'

                ,'YearRemodAdd'

                ,'BsmtFullBath'

                ,'BsmtHalfBath'

                ,'FullBath'

                ,'HalfBath'

                ,'Bedroomabvgr'

                ,'KitchenAbvGr'

                ,'TotRmsAbvGrd'

                ,'Fireplaces'

                ,'GarageYrBlt'

                ,'GarageCars'

                ,'MoSold'

                ,'YrSold'

                ,'MSSubClass'

                ]



bin_enc = categorical_vars + nominal_vars

# make the list lowercase

bin_enc = [x.lower() for x in bin_enc]





# Continuous 

# Anything not mentioned above



# Feature engineering

# (1stFlrSF + 2ndFlrSF) / number levels # avg level size

# 2ndFlrSF / 1stFlrSF  # how big is 1st floor compared to 2nd floor

# LotArea / LotFrontage # lot shape and ratio

# TotalBsmtSF / GrLivArea # basement size compared to living space

# LowQualFinSF / GrLivArea # ratio of Low quality finished to total

# FullBath / GrLivArea # bathrooms per living space

# HalfBath / GrLivArea # half bathrooms per living space

# Bedroom / GrLivArea # bedroom per living space

# Kitchen / GrLivArea # kitchen per living space

# TotRmsAbvGrd / GrLivArea # rooms per living space

# GarageCars / GarageArea # number of cars per garage area

# (WoodDeckSF+OpenPorchSF+EnclosedPorch+3SsnPorch+ScreenPorch+PoolArea) / GrLivArea # ratio of entertaining area vs living space

# Log saleprice

df['saleprice'] = np.log(df['saleprice'])
import sklearn.preprocessing as preprocessing

import seaborn as sns
# Encode the categorical features as numbers

import category_encoders as ce

def number_encode_features(df):

    result = df.copy()

    encoders = {}

    for column in result.columns:

        #print(column)

        #print(result.dtypes[column])

        if (result.dtypes[column] == np.int64 or result.dtypes[column] == np.int32 or result.dtypes[column] == np.float64):

            # impute missing values in column

            print('Imputing...')

            

        if result.dtypes[column] == np.object:

            encoders[column] = preprocessing.LabelEncoder()

            # if there are NaN's in the categorical data fill it with 'None' which becomes another category

            result[column] = encoders[column].fit_transform(result[column].fillna(value='None'))

            #encoder = ce.BinaryEncoder(cols=bin_enc)

    return result, encoders



# METHOD 1: NUMERICAL ENCODING

#encoded_data, _ = number_encode_features(df)



# METHOD 2: BINARY ENCODING

target_variable = 'saleprice'

encoder = ce.BinaryEncoder(cols=bin_enc)

encoder.fit(X=df.drop([target_variable, 'id'], axis=1))

encoded_data = encoder.transform(df.drop([target_variable, 'id'], axis=1)) 



print('Done')
# Data is now in dataframe "encoded_data"

y = df[target_variable].values

X = encoded_data.fillna(0.5).as_matrix()



# Scale the values between 0,1

scaler = preprocessing.MinMaxScaler((0,1), copy=True)

scaler.fit(X)

X = scaler.transform(X)



print('Done')
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split, cross_val_score



test_size = 0.2



X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=test_size, random_state=seed

)

print('Done')
import xgboost as xgb

from sklearn.grid_search import GridSearchCV
col_names = encoded_data.columns.values.tolist()
xg_train = xgb.DMatrix(X_train, y_train, feature_names=col_names)

xg_test = xgb.DMatrix(X_test, y_test, feature_names=col_names)

watchlist = [(xg_train,'train'),(xg_test,'eval')]
shift = 200

ind_params = {'eta':0.1,

              'gamma':0.5290,

             # 'n_estimators':300,

              'max_depth':7,

              'min_child_weight':4.2922,

              'seed':42,

              'subsample':0.9930,

              'colsample_bytree':0.3085,

              'objective':'reg:linear',

              'silent':1}

clf = xgb.train(ind_params, xg_train, 1000,

                    watchlist, early_stopping_rounds = 100 )

    
score = clf.predict(xg_test,ntree_limit=clf.best_ntree_limit )



from sklearn.metrics import mean_absolute_error

cv_score = mean_absolute_error(np.exp(y_test),np.exp(score))



shift = 200



print('MAE %0.4f' %(cv_score))

y_pred = np.exp(clf.predict(xg_test,

                ntree_limit = clf.best_ntree_limit)) #- shift

#plt.scatter( y_test_preds, y_test)



y_real = np.exp(y_test)

y_pred = y_pred



print(y_real.shape, y_pred.shape )



fig, ax = plt.subplots()

fit = np.polyfit(y_pred, y_real, deg=1)

ax.plot(y_pred, fit[0] * y_pred + fit[1], color='red')

ax.scatter(y_pred, y_real)



fig.show()
%matplotlib inline

import seaborn as sns

sns.set(font_scale = 1.5)



fig, ax = plt.subplots(figsize=(15, 15))

xgb.plot_importance(clf, ax=ax)
df = (

    pd.read_csv('../input/test.csv') # change this to run on kaggle

    #pd.read_csv('../input/train.csv')



    # Rename columns to lowercase and underscores

    .pipe(lambda d: d.rename(columns={

        k: v for k, v in zip(

            d.columns,

            [c.lower().replace(' ', '_') for c in d.columns]

        )

    }))

    # Switch categorical classes to integers

    #.assign(**{target_variable: lambda r: r[target_variable].astype('category').cat.codes})

)

print('Done')
# make the dummy columns for categoricals

encoded_data = encoder.transform(X=df.drop(['id'], axis=1))



# Fill Missing values with 0.5

X_sub = encoded_data.fillna(0.5).as_matrix()



# Scale values between 0,1

X_sub = scaler.transform(X_sub)





print('Done')
# Export dataframes to csv for inspection

#pd.DataFrame(X_train).to_csv('X_train.csv', index = False)



pd.DataFrame(X_sub).to_csv('X_sub.csv', index = False)
## Save to CSV with Image name and results

# Run the model



# Turn X_sub into a D matrix

xg_sub = xgb.DMatrix(X_sub)



# bring them back to sales prices using exponential

y_sub_preds = np.exp(clf.predict(xg_sub, ntree_limit = clf.best_ntree_limit)) 
pred = pd.DataFrame(data=y_sub_preds) 



print("Here is a sample...")



result = pd.concat([df['id'], pred], axis=1)

result.columns = ['Id','SalePrice'] 

print(result[0:10])



# Header: [image ALB BET DOL LAG NoF OTHER   SHARK   YFT]

result.to_csv('submission.csv', index = False)



print('Done')