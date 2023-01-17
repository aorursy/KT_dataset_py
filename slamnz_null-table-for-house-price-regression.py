from pandas import DataFrame, read_csv

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



# Numerical Features



float_features = features_by_dtype["float64"]

int_features = features_by_dtype["int64"]

numerical_features = float_features + int_features

remove_list = ["GarageYrBlt", "YearBuilt", "YearRemodAdd", "MoSold", "YrSold", "MSSubClass"]

numerical_features = [n for n in numerical_features if n not in remove_list]



numerical_features



# Temporal Features



temporal_features = remove_list.copy()

temporal_features.remove("MSSubClass")



temporal_features



# Summary



features, categorical_features, numerical_features, temporal_features



def feature_type(column):

    if column in categorical_features:

        return "Categorical"

    elif column in numerical_features:

        return "Numerical"

    elif column in temporal_features:

        return "Temporal"

    else:

        return "Unknown"
features = data.columns



features_with_null_values = []



for feature in features:

    

    column = data[feature]

    

    has_null = any(column.isnull())

    

    if has_null:

        

        features_with_null_values.append(feature)

        

print(features_with_null_values)

print()

print("Number of Features with Null Values: %s/%s" % (len(features_with_null_values), len(features)))
# The Code



features = data.columns

dictionary = {}



for feature in features:

    

    column = data[feature]

    

    has_null = any(column.isnull())

    

    if(has_null):

        

        null_count = column.isnull().value_counts()[True]

        not_null_count = column.notnull().value_counts()[True]

        total_rows = len(column)

        

        row = {}

        row["Type"] = feature_type(feature)

        row["Null Count"] = null_count

        row["Not Null Count"] = not_null_count

        row["Null Count / Total Rows"] = "%s / %s" %  (null_count, total_rows)

        row["Percentage of Nulls"] = "%.2f" % ((null_count / total_rows) * 100) + "%"

        row["Ratio"] = "%.2f : 1" %  ((null_count / not_null_count))

        

        if feature_type(feature) == "Categorical":

            row["Subcategories Count"] = str(len(data[feature].unique()))

        else:

            row["Subcategories Count"] = "N/A"

        

        dictionary[feature] = row



ordered_columns = ["Type", "Subcategories Count", "Null Count", "Not Null Count", "Ratio", "Null Count / Total Rows", "Percentage of Nulls"]

new_dataframe = DataFrame.from_dict(data = dictionary, orient="index")

new_dataframe[ordered_columns].sort_values("Null Count", ascending=False)