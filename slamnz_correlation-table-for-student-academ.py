from pandas import read_csv

data = read_csv("../input/xAPI-Edu-Data.csv")
target = "Class"

features = data.drop(target,1).columns.tolist()
features_by_dtype = {}



for f in features:

    dtype = str(data[f].dtype)

    if dtype not in features_by_dtype.keys():

        features_by_dtype[dtype] = [f]

    else:

        features_by_dtype[dtype] += [f]
keys = iter(features_by_dtype.keys())

k = next(keys)

l = features_by_dtype[k]

categorical_features = l

k = next(keys)

l = features_by_dtype[k]

numerical_features = l
categorical_features, numerical_features

features, target

pass
data[numerical_features].head()
data[categorical_features].head()
numerical_features
from scipy.stats import pearsonr,spearmanr,kendalltau

from itertools import combinations



rows_list = []



for x1,x2 in combinations(numerical_features,2):

    

    row = {}

    row["Variable A"] = x1 

    row["Variable B"] = x2

    

    pearson = pearsonr(data[x1],data[x2])

    row["Pearson"] = pearson[0]

    row["Pearson's p-value"] = pearson[1]

    

    spearman = spearmanr(data[x1],data[x2])

    row["Spearman"] = spearman[0]

    row["Spearman's p-value"] = spearman[1]

    

    kendall = kendalltau(data[x1],data[x2])

    row["Kendall"] = kendall[0]

    row["Kendall's p-value"] = kendall[1]

    

    rows_list.append(row)



ordered_columns = ["Variable A", "Variable B", "Pearson", "Pearson's p-value", "Spearman", "Spearman's p-value", "Kendall", "Kendall's p-value"]



from pandas import DataFrame



correlation_table = DataFrame(data=rows_list)[ordered_columns]



from IPython.display import display



display(correlation_table.sort_values("Pearson", ascending=False).round(2))