from pandas import read_csv, DataFrame



data = read_csv("../input/train.csv")
# Features

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



features

categorical_features, numerical_features, temporal_features

binary_features, has_zero_features, bounded_features



pass
data = data[categorical_features + ["SalePrice"]]

data = data.fillna("Unknown")
from scipy.stats import chisquare



temp = {}



for c in categorical_features:

    unit = data.groupby(c).count()["SalePrice"]

    temp[c] = chisquare(unit)

    

chisquare_dataframe = DataFrame(data=temp, index=["chi-square test statistic", "p-value"])

chisquare_dataframe.round(2).T
p_value_table = DataFrame(index = categorical_features, columns = categorical_features)



from scipy.stats import chi2_contingency

from pandas import crosstab



def is_statistically_significant(p):

        

    if p < 0.05:

        return 1

    else:

        return 0



duplicate = []

for c in categorical_features:

    

    duplicate.append(c)

    

    for cc in categorical_features:

        if not c == cc: # and cc not in duplicate:

            crosstable = crosstab(data[c], data[cc])

            chi2, p, dof, expected = chi2_contingency(crosstable)

            p_value_table[c][cc] = is_statistically_significant(p)
p_value_table
from seaborn import heatmap

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import colors

from numpy import float32



plt.figure(figsize=(13,12))



p = sns.diverging_palette(10, 220, sep=80, n=10)

myColors = [p[4], p[1]]

cmap = colors.LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))



ax = heatmap(p_value_table.fillna(float32(None)), linewidths=2, cmap=cmap, cbar=False, square=True)

cbar = ax.figure.colorbar(ax.collections[0])

cbar.set_ticks([1, 0])

cbar.set_ticklabels(["p < 0.05", "p >= 0.05"])

plt.show()
p_value_data = {}



for c in categorical_features:

    

    p_value_data[c] = {}

    row = p_value_data[c]

    

    column_value_counter = p_value_table[c].value_counts()

    relationships_counter = len(categorical_features) - 1

    significant = 0

    nonsignificant = 0

    

    if not column_value_counter.empty:

        

        if 1 in column_value_counter.keys():

            

            significant += column_value_counter[1]

            

        if 0 in column_value_counter.keys():

            

            nonsignificant += column_value_counter[0]

        

        row["p < 0.05"] = significant

        row["p >= 0.05"] = nonsignificant

        row["Percentage of Relationships Significant"] = ("%.1f" % ((significant / relationships_counter) * 100)) + "%"

        

DataFrame(data = p_value_data).T.sort_values("p < 0.05", ascending=False)