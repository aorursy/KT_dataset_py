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
X = data[input_features]

y = data[output_features]
X_train, X_test, y_train, y_test = train_test_split(X, y)
# Platz für euren Code