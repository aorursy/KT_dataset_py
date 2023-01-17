from pandas import read_csv

data = read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")



target = "Attrition"



feature_by_dtype = {}

for c in data.columns:

    

    if c == target: continue

    

    data_type = str(data[c].dtype)

    

    if data_type not in feature_by_dtype.keys():

         feature_by_dtype[data_type] = [c]

    else:

        feature_by_dtype[data_type].append(c)



objects = feature_by_dtype["object"]

remove = ["Over18"]

categorical_features = [f for f in objects if f not in remove]

int64s = feature_by_dtype["int64"]

remove.append("StandardHours")

remove.append("EmployeeCount")

count_features = []

for i in [i for i in int64s if len(data[i].unique()) < 20 and i not in remove]:

    count_features.append(i)

count_features = count_features + ["TotalWorkingYears", "YearsAtCompany", "HourlyRate"]

remove.append("EmployeeNumber")

numerical_features = [i for i in int64s if i not in remove]
from scipy.stats import pearsonr, spearmanr, kendalltau

from pandas import DataFrame

from itertools import permutations



funcs = [pearsonr,spearmanr,kendalltau]

corr = {}



for f in funcs:

    corr[f.__name__] = DataFrame(columns=numerical_features, index=numerical_features)



for i,j in permutations(numerical_features,2):

    

    for f in funcs:

        c, p = f(data[data[i].notnull()][i],data[data[j].notnull()][j])



        if p < 0.05:

            corr[f.__name__].set_value(j,i,c)
from seaborn import heatmap, axes_style, diverging_palette

from matplotlib.pyplot import show, title, suptitle, figure



with axes_style("whitegrid"):



    for f in funcs:

        

        figure(figsize=(12.5,12.5))

        

        df = corr[f.__name__]

        mask = DataFrame(df).isnull()

        df = df.fillna(0)

        heatmap(df.round(2), mask=mask, cmap=diverging_palette(240, 10, n=9, as_cmap=True), square=True, linewidths=0.5, linecolor="dimgrey", annot=True, annot_kws={"size": 9.5})

        title(f.__name__,fontsize=48, x=0.45, y=1.05, horizontalalignment = "center")

        show()