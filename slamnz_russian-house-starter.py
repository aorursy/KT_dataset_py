from pandas import read_csv

train = read_csv("../input/train.csv")

macro = read_csv("../input/macro.csv")
# Preview The Data

# ----------------------------------------------------------------------------------------------------
train.head()
macro.head()
# Data Shapes

# ----------------------------------------------------------------------------------------------------
train.shape
macro.shape
# Missing Data

# ----------------------------------------------------------------------------------------------------
def get_missing_data(data,features):

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

    

    return new_dataframe[ordered_columns].sort_values("Null Count", ascending=False)
get_missing_data(train, train.columns)
train.isnull().any().value_counts()
get_missing_data(macro, macro.columns)
macro.isnull().any().value_counts()
len(train.columns) + len(macro.columns)
def get_feature_dtypes(data,features):



    features_by_dtype = {}

    for f in features:

        dtype = str(data[f].dtype)



        if dtype not in features_by_dtype.keys():

            features_by_dtype[dtype] = [f]

        else:

            features_by_dtype[dtype] += [f]

            

    return features_by_dtype



def display_dtype_counts(dictionary):

    for key in dictionary.keys():

        string = "{}: {}".format(key,len(dictionary[key]))

        print(string)

        

def display_feature_by_dtype(data, dictionary, key):

    print(key)

    

    for feature in dictionary[key]:

        string = str(feature) + ": "

        

        if len(data[feature].unique()) < 15:

            string += str(data[feature].unique())

            print(string)

        else:

            string += "count(%s)"% len(data[feature].unique())

            print(string)
dtype_dict = get_feature_dtypes(train, train.columns)

display_dtype_counts(dtype_dict)
i = iter(dtype_dict.keys())
display_feature_by_dtype(train,dtype_dict,next(i))
display_feature_by_dtype(train,dtype_dict,next(i))
display_feature_by_dtype(train,dtype_dict,next(i))