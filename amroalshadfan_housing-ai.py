# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

 # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from numpy.random import seed
import tensorflow




import pandas as pd
housing = pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')
housing.head(4)
housing.info()


housing['ocean_proximity'].value_counts()
housing.describe()

import matplotlib.pyplot as plt

plt.figure(figsize=(10,7))
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
        s=housing["population"]/100, label="population", figsize=(15,8),
        c="median_house_value", cmap=plt.get_cmap("jet"),colorbar=True,
    )
plt.legend
corr_matrix=housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
print(corr_matrix)

print(corr_matrix)

import numpy as np

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set),"train +", len(test_set),"test")

from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)
corr_matrix.plot.hist()

housing.hist(bins=50, figsize=(20,15))
plt.show()
housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)


for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)
housing=strat_train_set.drop("median_house_value",axis=1)
housing_label=strat_train_set["median_house_value"].copy()
#handling missing features
housing.dropna(subset=["total_bedrooms"])

mySimpleImputer = SimpleImputer(strategy="mean")
housing_num = housing.drop('ocean_proximity',axis=1)
mySimpleImputer.fit(housing_num)
mySimpleImputer.statistics_
housing_num.median().values
x=mySimpleImputer.transform(housing_num)
housing_cat= housing[["ocean_proximity"]]
housing_cat.head(10)
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder=OrdinalEncoder()
housing_cat_encoded=ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]
from sklearn.preprocessing import OneHotEncoder
cat_encoder=OneHotEncoder()
housing_cat_1hot=cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
housing_cat_1hot.toarray()
from sklearn.base import BaseEstimator,TransformerMixin
rooms_ix,household_ix=3,6
class CombinedAttributeAdder(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform (self,X,y=None):
        rooms_per_houshold=X[:,rooms_ix] / X[:,household_ix]
        return np.c_[X,rooms_per_houshold]

attr_adder= CombinedAttributeAdder()
housing_extra_attribs=attr_adder.transform(housing.values)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('attribs_adder',CombinedAttributeAdder()),
    ('std_scaler',StandardScaler()),
])
housing_num_tr = num_pipeline.fit_transform(housing_num)
from sklearn.compose import ColumnTransformer
num_attribs=list(housing_num)
cat_attribs=["ocean_proximity"]
full_pipeline=ColumnTransformer([
    ("num",num_pipeline,num_attribs),
    ("cat",OneHotEncoder(),cat_attribs),
])
housing_prep=full_pipeline.fit_transform(housing)
from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(housing_prep,housing_label)
tdata=housing.iloc[:5]
tlabels=housing_label.iloc[:5]
data_prepared=full_pipeline.transform(tdata)
print("prediction: \t ",linreg.predict(data_prepared))
print("labels:\t ",list(tlabels))
def create_model(lyrs=[8,8,8] , act='relu' , opt='Adam' , dr=0.0):
    model=Sequential()
    model.add(Dense(lyrs[0], input_dim=housing_prep.shape[1],activation=act))
    for i in range(1,len(lyrs)):
        model.add(Dense(lyrs[i] , activation=act))
    model.add(Dropout(dr))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error' , optimizer=opt , metrics=['accuracy'])
    return model
model = create_model()
print(model.summary())
model = KerasClassifier(build_fn=create_model, verbose=0)

# define the grid search parameters
batch_size = [16, 32, 64]
epochs = [50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)

# search the grid
grid = GridSearchCV(estimator=model, 
                    param_grid=param_grid,
                    cv=3,
                    verbose=2)  # include n_jobs=-1 if you are using CPU

grid_result = grid.fit(housing_prep, housing_label)
print("Best Score:" , grid_result.best_score_)
print("Best Parameters :", grid_result.best_params_)

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
training = model.fit(housing_prep , housing_label , epochs=100, batch_size=32, validation_split=0.2, verbose=0)
val_acc = np.mean(training.history['accuracy'])
print("\n%s: %.2f%%" % ('Accuracy', val_acc*100))
from sklearn.model_selection import GridSearchCV

model_reg = KerasRegressor(build_fn=create_model, verbose=0)

activations = ['relu', 'tanh', 'sigmoid']

param_grid = dict(act=activations)

grid_search = GridSearchCV(estimator=model_reg, 
                    param_grid=param_grid,
                    cv=3,
                    verbose=2,
                    scoring='neg_mean_squared_error')
res=grid_search.fit(housing_prep, housing_label)
print("Best Score:" , res.best_score_)
print("Best Parameters :", res.best_params_)

means = res.cv_results_['mean_test_score']
stds = res.cv_results_['std_test_score']
params = res.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))