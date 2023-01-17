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

        

textual_features = features_by_dtype["object"]

numerical_features = ["weight","GPA"]

binary_category_features = ["Gender", "vitamins", "sports"]

calories_features = [i for i in features_by_dtype["int64"] if "calories" in i]

calories_features += ["calories_scone", "tortilla_calories"]

discrete_features = [i for i in features_by_dtype["int64"] if (not "calories" in i) & (data[i].unique().size > 2) & ("coded" not in i)]

discrete_features += ["calories_day","cook","exercise","father_education","income","life_rewarding","mother_education","persian_food"]

coded_features = [i for i in features_by_dtype["int64"] if (not "calories" in i) & (data[i].unique().size > 2) & ("coded" in i)]

coded_features += ["self_perception_weight", "comfort_food_reasons_coded", "cuisine", "employment", "fav_food", "marital_status","on_off_campus"]

image_questions = ["drink","soup","coffee", "fries", "breakfast"]



# === === #



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

        

# === === #



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



data["weight"] = data["weight"].apply(float)

data["GPA"] = data["GPA"].apply(float)
categorical_features = binary_category_features + coded_features + image_questions

for c in categorical_features: data[c] = data[c].apply(str)

pass
from pandas import pivot_table

from IPython.display import display

from itertools import combinations
for x,y in combinations(categorical_features,2):

    display(pivot_table(data[data["GPA"].notnull()], values="GPA", index=x, columns=y).fillna(0).round(2))
for x,y in combinations(categorical_features,2):

    display(pivot_table(data[data["weight"].notnull()], values="weight", index=x, columns=y).fillna(0).round(0))