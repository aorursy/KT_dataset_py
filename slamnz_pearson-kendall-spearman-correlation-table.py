from pandas import read_csv, DataFrame



data = read_csv("../input/train.csv")



# Features

target = "SalePrice"

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



features, target

categorical_features, numerical_features, temporal_features

binary_features, has_zero_features, bounded_features



pass
spearman = data[numerical_features + [target]].corr("spearman")

kendall = data[numerical_features + [target]].corr("kendall")

pearson = data[numerical_features + [target]].corr("pearson")



p = pearson[target].rename("Pearson")

k = kendall[target].rename("Kendall")

s = spearman[target].rename("Spearman")



from pandas import DataFrame



corr = DataFrame(data = p)

corr = corr.assign(Spearman = s)

corr = corr.assign(Kendall = k)



from IPython.display import display



display(corr.round(2).sort_values("Pearson", ascending=False))
from math import log

data["SalePrice"] = data["SalePrice"].apply(log)

spearman = data[numerical_features + [target]].corr("spearman")

kendall = data[numerical_features + [target]].corr("kendall")

pearson = data[numerical_features + [target]].corr("pearson")

p = pearson[target].rename("Pearson")

k = kendall[target].rename("Kendall")

s = spearman[target].rename("Spearman")

corr = DataFrame(data = p)

corr = corr.assign(Spearman = s)

corr = corr.assign(Kendall = k)

display(corr.round(2).sort_values("Pearson", ascending=False))