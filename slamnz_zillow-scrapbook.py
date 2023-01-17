from pandas import read_csv



train = read_csv("../input/train_2016_v2.csv")

properties = read_csv("../input/properties_2016.csv")
train.shape
properties.shape
properties.head()
train.head()
from pandas import merge



data = merge(train, properties, on=["parcelid"]) 
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

            print("%s: count(%s) %s" % (f,len(data[f].unique()),data[f].unique()[0:10]))



def show_all_uniques(data):

    dtypes = get_feature_lists_by_dtype(data)

    for key in dtypes.keys():

        print(key + "\n")

        show_uniques(data,dtypes[key])

        print()
show_all_uniques(data)