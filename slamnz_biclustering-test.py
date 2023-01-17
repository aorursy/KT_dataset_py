from pandas import read_csv, DataFrame

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
data.head()
from sklearn.preprocessing import normalize, scale
normed = normalize(data.drop(target,1))
from sklearn.cluster.bicluster import SpectralBiclustering, SpectralCoclustering
from numpy import argsort

from matplotlib.pyplot import show, imshow, figure, subplot, suptitle, tight_layout, xticks, xlabel



def pipeline(n, input_df):

    

    cocluster = SpectralCoclustering(n_clusters = n)

    cocluster.fit(input_df.values)

    cocluster_fit_data = input_df.values[argsort(cocluster.row_labels_)]

    cocluster_fit_data = cocluster_fit_data[:, argsort(cocluster.column_labels_)]

    

    bicluster = SpectralBiclustering(n_clusters = n)

    bicluster.fit(input_df.values)

    bicluster_fit_data = input_df.values[argsort(bicluster.row_labels_)]

    bicluster_fit_data = bicluster_fit_data[:, argsort(bicluster.column_labels_)]



    figure(figsize=(16,25))

    #suptitle("Mushrooms\n" + "n_clusters = " + str(n),fontsize=32, fontweight='bold')

    

    left_plot = subplot(211)

    ax = imshow(bicluster_fit_data, aspect='auto', cmap="bone")

    #xticks(range(0,len(input_df.columns)), list(input_df.columns[bicluster.column_labels_]),rotation='vertical')

    xlabel("Biclustering")

    

    right_plot = subplot(212)

    ax = imshow(cocluster_fit_data, aspect='auto', cmap="bone")

    #xticks(range(0,len(input_df.columns)), list(input_df.columns[cocluster.column_labels_]),rotation='vertical')

    xlabel("Coclustering")

    

    tight_layout()

    

    show()
from sklearn.preprocessing import LabelEncoder
transformed = DataFrame(normed, columns=features)

#transformed = data.copy()

transformed[target] = LabelEncoder().fit_transform(data[target])

transformed[target] = transformed[target].apply(lambda x: 0.1 if x == 0 else 0.9)
pipeline(12,transformed)