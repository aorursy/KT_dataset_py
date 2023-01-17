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
from sklearn.decomposition import PCA

pca = PCA()

pca.fit(data[features])
from pandas import DataFrame



transformed = pca.transform(data[features])

labels = ["P" + str(i) for i in range(1,len(features) + 1)]

transformed = DataFrame(transformed, columns=labels)

transformed[target] = data[target]
from seaborn import lmplot, kdeplot

from matplotlib.pyplot import show

from itertools import combinations
for label in labels:

    for value in data[target].unique():

        kdeplot(transformed[transformed[target] == value][label], shade=True)

    show()
for a,b in combinations(labels,2):

    lmplot(data=transformed, x=a, y=b, hue=target, fit_reg=False)

    show()