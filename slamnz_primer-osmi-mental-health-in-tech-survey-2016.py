def get_feature_lists_by_dtype(data,features):

    output = {}

    for f in features:

        dtype = str(data[f].dtype)

        if dtype not in output.keys(): output[dtype] = [f]

        else: output[dtype] += [f]

    return output



def show_uniques(data,features):

    for f in features:

        if len(data[f].unique()) < 30:

            print("%s: \ncount(%s) \n%s" % (f,len(data[f].unique()),data[f].unique()))

        else:

            print("%s: \ncount(%s) \n%s" % (f,len(data[f].unique()),data[f].unique()[0:10]))



def show_all_uniques(data):

    features = data.columns.tolist()

    dtypes = get_feature_lists_by_dtype(data,features)

    for key in dtypes.keys():

        print(key + "\n")

        show_uniques(data,dtypes[key])

        print()
from pandas import read_csv

data = read_csv("../input/mental-heath-in-tech-2016_20161114.csv")
show_all_uniques(data)