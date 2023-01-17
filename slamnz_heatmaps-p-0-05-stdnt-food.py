from pandas import read_csv

data = read_csv("../input/food_coded.csv")
features = data.columns
features_by_dtype = {}

for f in features:

    dtype = str(data[f].dtype)

    

    if dtype not in features_by_dtype.keys():

        features_by_dtype[dtype] = [f]

    else:

        features_by_dtype[dtype] += [f]
features_by_dtype.keys()
numerical_features = ["weight","GPA"]
calories_features = [i for i in features_by_dtype["int64"] if "calories" in i]

calories_features += ["calories_scone", "tortilla_calories"]
discrete_features = [i for i in features_by_dtype["int64"] if (not "calories" in i) & (data[i].unique().size > 2) & ("coded" not in i)]

discrete_features += ["calories_day","cook","exercise","father_education","income","life_rewarding","mother_education","persian_food"]
def code(value, code_dictionary):

    if value in code_dictionary.keys():

        return code_dictionary[value]

    else:

        return value

    

def ordinalizer(data,feature):

    output = {}

    unique = sorted(data[feature].unique().tolist())

    j = 1

    for i in [i for i in unique if str(i) != "nan"]:

        output[i] = j

        j += 1

    

    return output

    

def code_features_as_ordinal(data, to_be_coded):

    

    for feature in to_be_coded:

        cd = ordinalizer(data,feature)

        data[feature] = data[feature].apply(code, code_dictionary=cd)
code_features_as_ordinal(data,calories_features)
from numpy import NaN



nullify = [[15,61,102,104],[2,32,74,61]]



for i in nullify[0]:

    data.set_value(i,"GPA",NaN)

    

for i in nullify[1]:

    data.set_value(i,"weight",NaN)
data.set_value(67, "weight",144)

data.set_value(3, "weight", 240)

data.set_value(73,"GPA", 3.79)

pass
selected_features = discrete_features + numerical_features + calories_features
X = data

for f in selected_features:

    X = X[X[f].notnull()]

X = X[selected_features]

X.reset_index()

pass
for f in numerical_features:

    X[f] = X[f].apply(float)
from scipy.stats import pearsonr, spearmanr, kendalltau

from pandas import DataFrame

from itertools import permutations



funcs = [pearsonr,spearmanr,kendalltau]

corr = {}



for f in funcs:

    corr[f.__name__] = DataFrame(columns=selected_features, index=selected_features)



for i,j in permutations(selected_features,2):

    

    for f in funcs:

        c, p = f(X[i],X[j])



        if p < 0.05:

            corr[f.__name__].set_value(j,i,c)
from seaborn import heatmap, axes_style, diverging_palette

from matplotlib.pyplot import show, title, suptitle, figure



with axes_style("whitegrid"):



    for f in funcs:

        

        figure(figsize=(13,13))

        

        df = corr[f.__name__]

        mask = DataFrame(df).isnull()

        df = df.fillna(0)

        heatmap(df, mask=mask, cmap=diverging_palette(240, 10, n=9, as_cmap=True), square=True, linewidths=0.8, linecolor="dimgrey", annot=True, cbar=False, annot_kws={"size": 9.5})

        title(f.__name__,fontsize=48, x=0.45, y=1.05, horizontalalignment = "center")

        show()