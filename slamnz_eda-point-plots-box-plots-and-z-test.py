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
#Categorical Nan Values



data[categorical_features] = data[categorical_features].fillna("Unknown")
def display_ztest_dataframe(data, category, target):

    

    subclasses = data[category].unique()

    

    output = []

    

    from itertools import combinations

    for i,j in combinations(subclasses,2):

        

        subclass_1 = data[data[category] == i][target]

        subclass_2 = data[data[category] == j][target]

        

        from statsmodels.stats.weightstats import ztest

        ttest, p = ztest(subclass_1,subclass_2)

        

        unit = {}

        unit["Category A"] = i

        unit["Category B"] = j

        unit["t-test"] = ttest

        unit["p-value"] = p

        

        output.append(unit)

        

    from pandas import DataFrame

    from IPython.display import display

    display(DataFrame(data=output).sort_values("p-value", ascending=False).round(2))

    

def display_analysis(data, feature, target):

    

    from seaborn import pointplot, boxplot, cubehelix_palette, set_style

    from matplotlib.pyplot import show, figure, rc_context, subplot



    chosen_palette = cubehelix_palette(rot = 3)

    set_style("whitegrid")



    fig = figure(figsize=(12.5,5))

    fig.suptitle(feature)



    with rc_context({'lines.linewidth': 0.8}):

        

        subplot(121)

        pointplot = pointplot(x=feature, y=target, data=data, capsize=.14, color = chosen_palette[5])

        pointplot.set_ylabel("Average of %s" % feature)

        pointplot.set_xlabel("")



        subplot(122)

        boxplot = boxplot(x=feature, y=target, palette = chosen_palette, data=data)

        boxplot.set_ylabel(feature)

        boxplot.set_xlabel("")



    show()

    

    display_ztest_dataframe(data,feature,target)
for f in categorical_features:

    display_analysis(data, f, target)