from pandas import read_csv, DataFrame



data = read_csv("../input/train.csv")
# Features

features = data.columns.tolist()



if "Id" in features:

    features.remove("Id")

    

if "SalePrice" in features:

    features.remove("SalePrice")



features



##



# Features by dtype



features

features_by_dtype = {}



for feature in features:

    

    feature_dtype = str(data.dtypes[feature])

    

    try:

        features_by_dtype[feature_dtype]

    except KeyError:

        features_by_dtype[feature_dtype] = []

        

    

    features_by_dtype[feature_dtype].append(feature)



dtypes = features_by_dtype.keys()



##



# Categorical Features



categorical_features = features_by_dtype["object"]

categorical_features = categorical_features + ["MSSubClass"]



categorical_features



# Binary Features



binary_features = [c for c in categorical_features if len(data[c].unique()) == 2]



binary_features



# Numerical Features



float_features = features_by_dtype["float64"]

int_features = features_by_dtype["int64"]

numerical_features = float_features + int_features

remove_list = ["GarageYrBlt", "YearBuilt", "YearRemodAdd", "MoSold", "YrSold", "MSSubClass"]

numerical_features = [n for n in numerical_features if n not in remove_list]



numerical_features



# Has Zero Features



has_zero_features = []



for n in numerical_features:

    if 0 in data[n].unique():

        has_zero_features.append(n)

        

has_zero_features



# Bounded Features



bounded_features = ["OverallQual", "OverallCond"]



# Temporal Features



temporal_features = remove_list.copy()

temporal_features.remove("MSSubClass")



temporal_features



# Summary



features

categorical_features, numerical_features, temporal_features

binary_features, has_zero_features, bounded_features



pass
data = data[categorical_features + ["SalePrice"]]

data = data.fillna("Unknown")
from IPython.display import display

from itertools import combinations

from pandas import pivot_table



for x,y in combinations(categorical_features, 2):

    display(pivot_table(data=data[[x,y,"SalePrice"]], index=x, columns=y, values="SalePrice").round(-4).fillna(""))