import pandas as pd

import numpy as np 

np.random.seed(42)



%matplotlib inline  

import matplotlib as mpl 

import matplotlib.pyplot as plt

mpl.rc("axes", labelsize = 14)

mpl.rc("xtick", labelsize = 12)

mpl.rc("ytick", labelsize = 12) 
import warnings

import matplotlib.cbook

warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
housing = pd.read_csv('../input/california-housing-prices/housing.csv')

housing.info()
housing.head()
housing.ocean_proximity.unique()
housing.plot(kind='scatter',x='longitude', y='latitude', alpha=0.1)
housing.plot.scatter(x="longitude", y="latitude", alpha=0.4,

    s=housing["population"]/100, label="population", figsize=(10,7),

    c="median_house_value", cmap=plt.get_cmap("OrRd"), colorbar=True,

    sharex=False)

plt.legend()
corr_matrix = housing.corr()

corr_matrix['median_house_value'].sort_values(ascending = False)
from sklearn.model_selection import train_test_split



train_set, test_set = train_test_split(housing, test_size=0.2, random_state=1)
chosen_features = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income']

housing_train_set = train_set[chosen_features]

housing = train_set.drop("median_house_value", axis=1) # drop labels for training set

housing_labels = train_set["median_house_value"].copy()
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

imputer.fit(housing_train_set)

X = imputer.transform(housing_train_set)

housing_tr = pd.DataFrame(X, columns=housing_train_set.columns, index=housing_train_set.index)

housing_tr.info()

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()

cat_ocean_proximity = housing[['ocean_proximity']]

housing_cat = cat_encoder.fit_transform(cat_ocean_proximity)

#Change it to an array 

housing_cat.toarray()



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



num_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="median")),

        ('std_scaler', StandardScaler()),

    ])





from sklearn.compose import ColumnTransformer



num_attribs = list(housing_train_set)

cat_attribs = ["ocean_proximity"]



full_pipeline = ColumnTransformer([

        ("num", num_pipeline, num_attribs),

        ("cat", OneHotEncoder(), cat_attribs),

    ])



housing_prepared = full_pipeline.fit_transform(housing)
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)

forest_reg.fit(housing_prepared, housing_labels)
from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

housing_predictions = forest_reg.predict(housing_prepared)

forest_mse = mean_squared_error(housing_labels, housing_predictions)

forest_rmse = np.sqrt(forest_mse)

forest_rmse