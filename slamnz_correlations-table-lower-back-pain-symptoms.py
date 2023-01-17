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
from scipy.stats import pearsonr,spearmanr,kendalltau

from itertools import combinations



rows_list = []



for x1,x2 in combinations(features,2):

    

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
where = (correlation_table["Pearson's p-value"] < 0.05) & ((correlation_table["Pearson"] > 0.1) | (correlation_table["Pearson"] < -0.1))
from IPython.display import display

from pandas import set_option

set_option("display.max_rows", 66)

display(correlation_table[where].sort_values("Spearman",ascending=False).round(2))