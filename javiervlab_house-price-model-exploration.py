#Libraries import



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



sns.set_style("whitegrid")



from sklearn.ensemble import RandomForestRegressor
#dataset import

train_raw = pd.read_csv("../input/train.csv")

test_raw = pd.read_csv("../input/test.csv") 
def data_import():

    train_raw = pd.read_csv("../input/train.csv")

    test_raw = pd.read_csv("../input/test.csv") 

    

    from sklearn import preprocessing

    

    #Label encode, encode NaN value as other label

    le = preprocessing.LabelEncoder()

    

    for i in test_raw:

        if test_raw[i].dtype == "object":

            

            le.fit(test_raw[i].fillna('NA').value_counts().index.unique())

            test_raw[i] = le.transform(np.array(test_raw[i].fillna('NA')))

            

            le.fit(train_raw[i].fillna('NA').value_counts().index.unique())

            train_raw[i] = le.transform(np.array(train_raw[i].fillna('NA')))

            

    #Drop outliers

    train_raw = train_raw.drop(train_raw.index[[523,1298]])

    

    #Fill missing values by the mean in each features

    train_raw = train_raw.fillna(train_raw.mean())

    test_raw  = test_raw.fillna(test_raw.mean())

    

    return train_raw, test_raw
train_raw, test_raw = data_import()
#Columns chosen before



cols = ['MSZoning', 'LotFrontage', 'LotArea', 'LotShape', 'Neighborhood',

       'HouseStyle', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',

       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrArea', 'ExterQual',

       'ExterCond', 'Foundation', 'BsmtQual', 'BsmtExposure', 'BsmtFinSF1',

       'BsmtUnfSF', 'TotalBsmtSF', 'HeatingQC', 'CentralAir', 'Electrical',

       '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'FullBath',

       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',

       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'GarageType', 'GarageYrBlt',

       'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',

       'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',

       'ScreenPorch', 'SaleCondition']
from sklearn.ensemble import ExtraTreesClassifier
forest = ExtraTreesClassifier(n_estimators=100, random_state=0)

forest.fit(train_raw[cols], train_raw['SalePrice'])
features = pd.DataFrame({"category": cols,

                            "importance": forest.feature_importances_})



features = features.sort(['importance'],ascending=False)
plt.figure(figsize=(8,12))

plt.title("Order of feature importance", fontsize="large")

sns.barplot(x="importance",y="category",data=features)
train = forest.transform(train_raw[cols].drop(

    [197, 581, 1324, 1349,112, 688, 691, 769, 803, 898, 1046, 1169,1182]))

test = forest.transform(test_raw[cols])

target = train_raw['SalePrice'].drop(

    [197, 581, 1324, 1349,112, 688, 691, 769, 803, 898, 1046, 1169,1182])
def model_test():

    train_n = train.copy()

    regressor = RandomForestRegressor(n_estimators=100, min_samples_split=2)

    regressor.fit(train_n,target)

    score = regressor.score(train_n,target)

    print("Model score: ",score)

    predict = regressor.predict(train_n)

    residuals = target-predict

    residuals.hist(bins=100)

    #np.log(residuals).hist(bins=100)

    

    return None
model_test()
from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
rg = RandomForestRegressor(n_estimators=200, min_samples_split=4)

kf = KFold(n_splits=10, random_state=0)

results = cross_val_score(rg, train, target, cv=kf, verbose=0)

print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
# Keras model

def keras_model():

	model = Sequential()

	model.add(Dense(22, input_dim=22, init='normal', activation='relu'))

	model.add(Dense(1, init='normal'))

	model.compile(loss='mean_squared_error', optimizer='adam')

	return model