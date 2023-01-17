from pandas import read_csv

data = read_csv("../input/Kaggle_Test_Dataset_v2.csv")
data.head()
data.isnull().any()
data.shape
def get_feature_lists_by_dtype(data):

    features = data.columns.tolist()

    output = {}

    for f in features:

        dtype = str(data[f].dtype)

        if dtype not in output.keys(): output[dtype] = [f]

        else: output[dtype] += [f]

    return output



def show_uniques(data,features):

    for f in features:

        if len(data[f].unique()) < 30:

            print("%s: count(%s) %s" % (f,len(data[f].unique()),data[f].unique()))

        else:

            print("%s: count(%s/%s) %s" % (f,len(data[f].unique()),len(data),data[f].unique()[0:10]))



def show_all_uniques(data):

    dtypes = get_feature_lists_by_dtype(data)

    for key in dtypes.keys():

        print(key + "\n")

        show_uniques(data,dtypes[key])

        print()
show_all_uniques(data)
dtypes = get_feature_lists_by_dtype(data)



remove = ["sku"]

categories = [feature for feature in dtypes["object"] if feature not in remove]

numerics = dtypes["float64"]
data[categories].head()
data[categories].isnull().any()
for c in categories: data[c] = data[c].apply(str)
data[numerics].head()
data[numerics].isnull().any()