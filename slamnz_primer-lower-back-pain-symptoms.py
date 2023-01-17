from pandas import read_csv

raw_data = read_csv("../input/Dataset_spine.csv")
raw_data.head()
# === Rename The Columns === #



column_names = ("pelvic_incidence",

"pelvic_tilt",

"lumbar_lordosis_angle",

"sacral_slope",

"pelvic_radius",

"degree_spondylolisthesis",

"pelvic_slope",

"Direct_tilt",

"thoracic_slope",

"cervical_tilt",

"sacrum_angle",

"scoliosis_slope")



# === Rename === #



rename = {}

for i in range(0,12):

    temp = "Col" + str(i+1)

    rename[temp] = column_names[i]



renamed_data = raw_data.rename(columns = rename)
data = renamed_data.drop("Unnamed: 13",1)

target = "Class_att"

features = [feature for feature in data.columns if feature != target]
def get_feature_lists_by_dtype(data):

    output = {}

    for f in data.columns:

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

    features = data.columns.tolist()

    dtypes = get_feature_lists_by_dtype(data)

    for key in dtypes.keys():

        print(key + "\n")

        show_uniques(data,dtypes[key])

        print()
data.shape
data.groupby(target).count()[features[0]]
data.isnull().any()
show_all_uniques(data)
from seaborn import distplot, boxplot, countplot, set_style,despine, axes_style, set_palette, color_palette

from matplotlib.pyplot import subplot, show

from IPython.display import display

from pandas import DataFrame

from scipy.stats import normaltest, skew, skewtest



# === Numeric Analysis === #



def numeric_analysis(series):

    

    no_nulls = series.dropna()

    

    with axes_style({"axes.grid": False}):

        

        cell_1 = subplot(211)

        dp = distplot(no_nulls, kde=True)

        dp.set_xlabel("",visible=False)

        dp.set_yticklabels(dp.get_yticklabels(),visible=False)

        despine(left = True)



        cell_2 = subplot(212, sharex=cell_1)

        boxplot(no_nulls)

        despine(left=True)

    

    show()

    

    display(DataFrame(series.describe().round(2)).T)

    

    display(DataFrame(list(normaltest(series)), columns=["Normal Test"], index=["statistic","p-value"]).T.round(2))

    

    display(DataFrame(list(skewtest(series)), columns=["Skew Test"], index=["statistic","p-value"]).T.round(2))

    

    display(DataFrame([skew(series)], columns=["Skew"], index=[""]).T)

    

# === Category Analysis === #

    

def category_analysis(series):

    

    set_style("whitegrid")

    set_palette = color_palette("colorblind")

    

    with axes_style({'axes.grid': False}):

        cp = countplot(series)

        cp.set_title(cp.get_xlabel())

        cp.set_xlabel("",visible=False)

        despine()

    

    show()

    display(DataFrame(series.value_counts().apply(lambda x: "{:.2f}%".format(x / len(series) * 100))).T)
category_analysis(data[target])
for feature in features:

    numeric_analysis(data[feature])