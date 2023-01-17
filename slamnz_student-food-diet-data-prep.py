from pandas import read_csv

data = read_csv("../input/food_coded.csv")
features = data.columns
data.shape
data.isnull().any().value_counts()
features_by_dtype = {}

for f in features:

    dtype = str(data[f].dtype)

    

    if dtype not in features_by_dtype.keys():

        features_by_dtype[dtype] = [f]

    else:

        features_by_dtype[dtype] += [f]
features_by_dtype.keys()
keys = iter(features_by_dtype.keys())
key = next(keys)

dtype_list = features_by_dtype[key]

for f in dtype_list:

    string = "{}: {}".format(f,len(data[f].unique()))

    print(string)
textual_data = dtype_list 
key = next(keys)

dtype_list = features_by_dtype[key]

for f in features_by_dtype[key]:

    string = "{}: {}".format(f,data[f].unique())

    print(string)
binary_features = [f for f in dtype_list if len(data[f].unique()) == 2]

categorical_features = binary_features

numerical_features = [f for f in dtype_list if f not in categorical_features]

count_features = numerical_features
key = next(keys)

features_by_dtype[key]

for f in features_by_dtype[key]:

    string = "{}: {}".format(f,data[f].unique())

    print(string)
numerical_features
data[numerical_features].head()
categorical_features
data[categorical_features].head()
dictionary = {}



for feature in features:

    

    column = data[feature]

    

    has_null = any(column.isnull())

    

    if(has_null):

        

        null_count = column.isnull().value_counts()[True]

        not_null_count = column.notnull().value_counts()[True]

        total_rows = len(column)

        

        row = {}

        row["Null Count"] = null_count

        row["Not Null Count"] = not_null_count

        row["Null Count / Total Rows"] = "%s / %s" %  (null_count, total_rows)

        row["Percentage of Nulls"] = "%.2f" % ((null_count / total_rows) * 100) + "%"

        row["Ratio (Not Null : Null)"] = "%.2f : 1" %  ((null_count / not_null_count))

        

        dictionary[feature] = row



ordered_columns = ["Null Count", "Not Null Count", "Ratio (Not Null : Null)", "Null Count / Total Rows", "Percentage of Nulls"]



from pandas import DataFrame



new_dataframe = DataFrame.from_dict(data = dictionary, orient="index")

new_dataframe[ordered_columns].sort_values("Null Count", ascending=False)
data[textual_data].head()
data["comfort_food_reasons"].head(20)
data["eating_changes"].head(20)
data["ideal_diet"].head(20)
data["diet_current"].head(20)