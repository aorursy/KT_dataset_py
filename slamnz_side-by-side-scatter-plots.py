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
from seaborn import lmplot

from matplotlib.pyplot import show,figure

from itertools import combinations



for x,y in combinations(features,2):

    figure(figsize=(12.5,4))

    lmplot(x=x, y=y, col=target, data=data, palette={"Abnormal": "red", "Normal": "darkgrey"})

    show()
data = renamed_data

numerical_features = features



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



correlation_table = DataFrame(columns=ordered_columns, data=rows_list)



from IPython.display import display



display(correlation_table.sort_values("Pearson", ascending=False).round(2))



target_filter = (correlation_table["Variable B"] == target) | (correlation_table["Variable A"] == target)



display(correlation_table.sort_values("Pearson", ascending=False).round(2))