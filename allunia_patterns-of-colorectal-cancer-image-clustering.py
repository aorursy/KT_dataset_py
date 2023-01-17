%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import stats



from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



from os import listdir

print(listdir("../input"))



import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", category=FutureWarning)
images = pd.read_csv('../input/hmnist_64_64_L.csv')

images.head()
print("The data consists of {} samples".format(images.shape[0]))
class_names = {1: "Tumor", 2: "Stroma", 3: "Complex", 4: "Lympho",

               5: "Debris", 6: "Mucosa", 7: "Adipose", 8: "Empty"}

class_colors = {1: "Red", 2: "Orange", 3: "Gold", 4: "Limegreen",

                5: "Mediumseagreen", 6: "Darkturquoise", 7: "Steelblue", 8: "Purple"}
sns.set_style("whitegrid")

fig, ax = plt.subplots(2,4, figsize=(25,11))

for n in range(2):

    for m in range(4):

        class_idx = n*4+(m+1)

        sns.distplot(images[images.label == class_idx].drop("label", axis=1).values.flatten(),

                     ax=ax[n,m],

                     color=class_colors[class_idx])

        ax[n,m].set_title(class_names[class_idx])

        ax[n,m].set_xlabel("Intensity")

        ax[n,m].set_ylabel("Density")
def get_overall_statistics(data, cancer_class):

    class_intensities = data[data.label == cancer_class].values.flatten()

    class_stats = np.zeros(10)

    class_stats[0] = stats.mode(class_intensities)[0][0]

    for q in range(1, 10):

        class_stats[q] = np.quantile(class_intensities, (q * 10)/100)

    return class_stats



stats_quantities = ["Mode", "Q10", "Q20", "Q30", "Q40", "Median", "Q60", "Q70", "Q80", "Q90"]

overall_statistics = pd.DataFrame(index = np.arange(1,9), columns=stats_quantities)



for class_idx in range(1,9):

    overall_statistics.loc[class_idx,:] = get_overall_statistics(images, class_idx)



overall_statistics = overall_statistics.reset_index()

overall_statistics["index"] = overall_statistics["index"].apply(lambda l : class_names[l])

overall_statistics = overall_statistics.set_index("index")

overall_statistics.index.name = None
fig, ax = plt.subplots(1,1,figsize=(10,5))

sns.heatmap(overall_statistics, annot=True, cbar=False, fmt="g", cmap="YlGnBu_r", ax=ax)
image_statistics = pd.DataFrame(index=images.index)

image_statistics["Mode"] = images.apply(lambda l: stats.mode(l)[0][0], axis=1)

for q in range(1, 10):

    col_name = "Q" + str(q*10)

    image_statistics[col_name] = images.apply(lambda l: np.quantile(l, (q*10)/100), axis=1)



image_statistics["label"] = images.label.values

image_statistics.head()
your_choice = ["Mode"]
sns.set_style("whitegrid")

fig, ax = plt.subplots(2,4, figsize=(25,11))

for n in range(2):

    for m in range(4):

        class_idx = n*4+(m+1)

        sns.distplot(images[images.label == class_idx].drop("label", axis=1).values.flatten(),

                     ax=ax[n,m],

                     color=class_colors[class_idx], 

                     norm_hist=True,

                     bins=50)

        sns.distplot(image_statistics[image_statistics.label == class_idx][your_choice].values,

                     ax=ax[n,m],

                     color="lightskyblue",

                     norm_hist=True, 

                     bins=50)

        ax[n,m].set_title(class_names[class_idx])

        ax[n,m].set_xlabel("Intensity")

        ax[n,m].set_ylabel("Density")
sns.pairplot(data=image_statistics,

             vars=["Mode", "Q20", "Q50", "Q70"],

             hue="label",

             palette=class_colors,

             plot_kws={"s": 20, "alpha": 0.2},

             size=4)
from sklearn.mixture import GaussianMixture

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X = scaler.fit_transform(image_statistics.drop("label", axis=1).values)

y = image_statistics.label.values
model = GaussianMixture(n_components=8,

                        covariance_type="full",

                        n_init=10,

                        random_state=0)
cluster = model.fit_predict(X)

image_statistics["cluster"] = cluster
counts = image_statistics.groupby("label").cluster.value_counts()

counts = counts.unstack()

counts.fillna(0, inplace=True)



sns.set()

plt.figure(figsize=(20,6))

for n in range(8):

    for m in range(1,9):

        plt.scatter(n, m, s=counts.loc[m,n], color=class_colors[m])

plt.xlabel("Cluster")

#plt.ylabel("Cancer label")

plt.yticks(np.arange(1,9), [class_names[label] for label in list(np.arange(1,9))]);

plt.title("Which cancer type is covered most per cluster?");
selection = image_statistics[(image_statistics.cluster==6) & (image_statistics.label==2)].index.values

opposite = np.random.choice(

    image_statistics[(image_statistics.cluster==0) & (image_statistics.label==2)].index.values, 

    size=len(selection)

)



fig, ax = plt.subplots(2, np.min([len(selection),5]), figsize=(20,10));

for n in range(len(selection)):

    ax[0,n].imshow(images.drop("label", axis=1).loc[selection[n],:].values.reshape(64,64), cmap="gray")

    ax[0,0].set_title("Images of comet tail cluster {}".format(str(6)))

    ax[0,1].set_title("of the class: {}".format(class_names[2]))

    ax[1,n].imshow(images.drop("label", axis=1).loc[opposite[n],:].values.reshape(64,64), cmap="gray")

    ax[1,0].set_title("Images of major cluster {}".format(str(0)))

    ax[1,1].set_title("of the class: {}".format(class_names[2]))
pal = sns.color_palette("Set2", n_colors=10)

fig, ax = plt.subplots(2,5, figsize=(20,9))

cols_to_use = image_statistics.drop("label", axis=1).columns.values

for n in range(5):

    for m in range(2):

        col = cols_to_use[m*5 + n]

        sns.distplot(image_statistics[col].values, ax=ax[m,n], color=pal[m*5 + n])

        ax[m,n].set_title(col)
cols_to_use = ["Q10", "Q90", "Mode", "label"]

reduced_statistics = image_statistics.loc[:,cols_to_use].copy()
fig, ax = plt.subplots(1,3, figsize=(20,5));

sns.kdeplot(reduced_statistics.Q10, reduced_statistics.Mode, shade=True, ax=ax[0]);

sns.kdeplot(reduced_statistics.Mode, reduced_statistics.Q90, shade=True, ax=ax[1]);

sns.kdeplot(reduced_statistics.Q10, reduced_statistics.Q90, shade=True, ax=ax[2]);
scaler = StandardScaler()

X = scaler.fit_transform(reduced_statistics.drop("label", axis=1).values)

y = image_statistics.label.values



model = GaussianMixture(n_components=3,

                        covariance_type="full",

                        n_init=10,

                        random_state=0)



cluster = model.fit_predict(X)

reduced_statistics["cluster"] = cluster



counts = reduced_statistics.groupby("label").cluster.value_counts()

counts = counts.unstack()

counts.fillna(0, inplace=True)



counts = counts.astype(np.int)
sns.set()

plt.figure(figsize=(10,6))

for n in range(3):

    for m in range(1,9):

        plt.scatter(n, m, s=counts.loc[m,n], color=class_colors[m])

plt.xlabel("Cluster")

plt.xticks([0, 1, 2])

#plt.ylabel("Cancer label")

plt.yticks(np.arange(1,9), [class_names[label] for label in list(np.arange(1,9))]);

plt.title("Which cancer type is covered most per cluster?");
outliers = images.loc[reduced_statistics.cluster == 2]

normals = images.loc[reduced_statistics.cluster == 1]

comet = images.loc[reduced_statistics.cluster == 0]
fig, ax = plt.subplots(6, 4, figsize=(20,35))

n = 0

for value in [1,2,3,4,5,6]:

    for m in range(2):

        outlier_image = outliers[outliers["label"] == value].drop("label", axis=1).values[m]

        ax[n,m].imshow(outlier_image.reshape(64,64), cmap="gray")

        ax[n,m].set_title("Outlier of class {}".format(class_names[value]))

        normal_image = normals[normals.label == value].drop("label", axis=1).values[m]

        ax[n,m+2].imshow(normal_image.reshape(64,64), cmap="gray")

        ax[n,m+2].set_title("Normal of class {}".format(class_names[value]))

    n+=1
images["cluster"] = reduced_statistics.cluster.values

images.cluster.value_counts()
images.to_csv("image_clustering_results.csv")