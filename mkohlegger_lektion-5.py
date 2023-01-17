import pandas as pd

from sklearn.model_selection import train_test_split
data = pd.read_csv("../input/dsia19-california-housing/housing.csv")



input_features = [

    'longitude',

    'latitude',

    'housing_median_age',

    'total_rooms',

    'total_bedrooms',

    'population',

    'households',

    'median_income',

    'ocean_proximity'

]



output_features = [

    'median_house_value'

]



X_train, X_test, y_train, y_test = train_test_split(

    data[input_features],

    data[output_features]

)
from sklearn.preprocessing import MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

imputer.fit(data[["longitude", "latitude", "total_bedrooms"]])

imputed_data = imputer.transform(data[["longitude", "latitude", "total_bedrooms"]])
scaler = MinMaxScaler()

scaler.fit(imputed_data)

scaled_data = scaler.transform(imputed_data)
from sklearn.pipeline import Pipeline
pipeline = Pipeline(steps=[

    ("impute data", SimpleImputer(strategy="median")),

    ("scale data", MinMaxScaler())

])
pipeline.fit(data[["longitude", "latitude", "total_bedrooms"]])

processed_data = pipeline.transform(data[["longitude", "latitude", "total_bedrooms"]])
class ColumnSelector:

    

    def __init__(self, select_numeric=True):

        self.select_numeric = select_numeric

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        if self.select_numeric:

            return X.select_dtypes(include=["number"])

        elif not self.select_numeric:

            return X.select_dtypes(exclude=["number"])
num_pipeline = Pipeline(steps=[

    ("select numbers", ColumnSelector(select_numeric=True)),

    ("impute data", SimpleImputer(strategy="median")),

    ("scale data", MinMaxScaler())

])
non_num_pipeline = Pipeline(steps=[

    ("select non numeric", ColumnSelector(select_numeric=False)),

    ("encode data", OneHotEncoder())

])
num_pipeline.fit(data)

non_num_pipeline.fit(data)



num_pipeline.transform(data)

non_num_pipeline.transform(data)
from sklearn.pipeline import FeatureUnion
X_pipeline = FeatureUnion(transformer_list=[

    ("numeric pipeline", Pipeline(steps=[

        ("select numbers", ColumnSelector(select_numeric=True)),

        ("impute data", SimpleImputer(strategy="median")),

        ("scale data", MinMaxScaler())

    ])),

    ("non_numeric pipeline", Pipeline(steps=[

        ("select non numeric", ColumnSelector(select_numeric=False)),

        ("encode data", OneHotEncoder())

    ]))

])
X_pipeline.fit(X_train)

X_train_p = X_pipeline.transform(X_train)

X_test_p = X_pipeline.transform(X_test)
y_pipeline = Pipeline(steps=[

    ("scale data", MinMaxScaler())

])
y_pipeline.fit(y_train)

y_train_p = y_pipeline.transform(y_train)

y_test_p = y_pipeline.transform(y_test)
# Platz für euren Code