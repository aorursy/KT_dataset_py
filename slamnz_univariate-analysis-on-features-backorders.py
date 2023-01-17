from pandas import read_csv

data = read_csv("../input/Kaggle_Test_Dataset_v2.csv")
data.head()
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
from seaborn import countplot, set_style,despine, axes_style

from matplotlib.pyplot import show

from IPython.display import display

from pandas import DataFrame



def category_analysis(series):

    

    set_style("whitegrid")

    

    with axes_style({'axes.grid': False}):

        cp = countplot(series)

        cp.set_title(cp.get_xlabel())

        cp.set_xlabel("",visible=False)

        despine()

    

    show()

    display(DataFrame(series.value_counts().apply(lambda x: x / len(data) * 100).round(2)).T)

    
for category in categories:

    category_analysis(data[category])
from seaborn import distplot, boxplot

from matplotlib.pyplot import subplot



def numeric_analysis(series):

    

    no_nulls = series.dropna()

    

    with axes_style({"axes.grid": False}):

        

        cell_1 = subplot(211)

        dp = distplot(no_nulls, kde=False)

        dp.set_xlabel("",visible=False)

        dp.set_yticklabels(dp.get_yticklabels(),visible=False)

        despine(left = True)



        cell_2 = subplot(212, sharex=cell_1)

        boxplot(no_nulls)

        despine(left=True)

    

    show()

    

    display(DataFrame(series.describe().round(2)).T)
for n in numerics: 

    numeric_analysis(data[data[n].notnull()][n])

    if 0 in data[n].unique(): 

        print("Removed Zeros")

        numeric_analysis(data[data[n] > 0][n])