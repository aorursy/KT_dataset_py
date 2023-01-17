import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.pipeline import FeatureUnion, Pipeline

from sklearn.linear_model import LinearRegression
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



y_pipeline = Pipeline(steps=[

    ("scale data", MinMaxScaler())

])



X_pipeline.fit(X_train)

X_train_p = X_pipeline.transform(X_train)

X_test_p = X_pipeline.transform(X_test)



y_pipeline.fit(y_train)

y_train_p = y_pipeline.transform(y_train)

y_test_p = y_pipeline.transform(y_test)
lr_model = LinearRegression()

lr_model.fit(X_train_p, y_train_p)
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score, max_error
r2_score(y_test_p, lr_model.predict(X_test_p))
r2_score(y_test, y_pipeline.inverse_transform(lr_model.predict(X_test_p)))
explained_variance_score(y_test_p, lr_model.predict(X_test_p))
explained_variance_score(y_test, y_pipeline.inverse_transform(lr_model.predict(X_test_p)))
mean_squared_error(y_test, y_pipeline.inverse_transform(lr_model.predict(X_test_p)))
max_error(y_test, y_pipeline.inverse_transform(lr_model.predict(X_test_p)))
# Platz für euren Code