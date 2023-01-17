from pandas import read_csv

data = read_csv("../input/Melbourne_housing_extra_data.csv")
target = "Price"

features = [d for d in data.columns if (d != target) & (len(data[d].unique()) > 1)]
data = data[data[target].notnull()]
data.head()
data.shape
def get_dtype_lists(data,features):

    output = {}

    for f in features:

        dtype = str(data[f].dtype)

        if dtype not in output.keys(): output[dtype] = [f]

        else: output[dtype] += [f]

    return output



def show_uniques(data,features):

    for f in features:

        if len(data[f].unique()) < 30:

            print("%s: %s" % (f,data[f].unique()))

        else:

            print("%s: count(%s) %s" % (f,len(data[f].unique()),data[f].unique()[0:10]))



def show_all_uniques(data,features):

    dtypes = get_dtype_lists(data,features)

    for key in dtypes.keys():

        print(key + "\n")

        show_uniques(data,dtypes[key])

        print()
show_all_uniques(data,features)
dtype_lists = get_dtype_lists(data,features)



to_be_transformed = ["Date", "Address", "YearBuilt"]

categories = [c for c in dtype_lists["object"] if c not in to_be_transformed]

categories += ["Postcode"]

counts = ["Rooms", "Bedroom2", "Bathroom", "Car"]

numerics = [i for i in dtype_lists["float64"] if i not in categories + to_be_transformed]
from seaborn import countplot, set_style

from matplotlib.pyplot import show,figure



set_style("whitegrid")



for c in categories + counts:

    i = int(len(data[c].fillna(-1).unique()) * 0.3)

    figure(figsize=(12.5, 5 + i))

    ax=countplot(y=data[c].fillna(-1))

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    show()