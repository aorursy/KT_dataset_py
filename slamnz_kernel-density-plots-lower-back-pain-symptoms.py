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
from seaborn import kdeplot, regplot, set_style,despine

from itertools import combinations

from matplotlib.pyplot import show, subplot, figure, suptitle



for num1, num2 in combinations(features,2):

    set_style('whitegrid')

    

    figure(figsize=(12.5,6))

    

    suptitle(num1 + " x " + num2)



    subplot(121)

    ax = kdeplot(data[data[target] == data[target].unique()[0]][num1], data[data[target] == data[target].unique()[0]][num2], cmap="Reds", shade=True, shade_lowest=False)

    ax = kdeplot(data[data[target] == data[target].unique()[1]][num1], data[data[target] == data[target].unique()[1]][num2], cmap="Blues", shade=True, shade_lowest=False)

    

    subplot(122)

    ax = regplot(data=data[data[target] == data[target].unique()[0]], x=num1, y=num2, fit_reg=False, color="red")

    ax = regplot(data=data[data[target] == data[target].unique()[1]], x=num1, y=num2, fit_reg=False, color="blue")

    

    show()