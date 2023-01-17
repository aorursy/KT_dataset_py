 ## Import Python libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(color_codes=True)



import scipy as sp #collection of functions for scientific computing and advance mathematics

from scipy import stats

from scipy.stats import norm, skew

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax





from sklearn.cluster import KMeans





from sklearn.model_selection import GroupKFold

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor



import os

print(os.listdir("../input"))





# warnings mute

import warnings

warnings.simplefilter('ignore')
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

sub = pd.read_csv("../input/sample_submission.csv")

structures = pd.read_csv("../input/structures.csv")

dipole = pd.read_csv("../input/dipole_moments.csv")

potential = pd.read_csv("../input/potential_energy.csv")

mulliken = pd.read_csv("../input/mulliken_charges.csv")

contributions = pd.read_csv("../input/scalar_coupling_contributions.csv")
train.head()
test.head()
structures.head()
dipole.head()
potential.head()
mulliken.head()
contributions.head()
sub.head()
df1 = pd.merge(train,  dipole)

df = pd.merge(df1, potential)



# reorder columns: locate the target variable at the end

df = df[["id", "molecule_name", "atom_index_0", "atom_index_1", 

         "type", "X", "Y", "Z", "potential_energy", "scalar_coupling_constant"]]



df.head()
# check if some null values were introduced

df.isnull().any()
# Note:

N = 3000

print("The original dataset has {} records, which is quite a lot.\nIn some cases, we will select just {} rows to speed up the computation.". format(df.shape[0], N))

df.groupby("type").median()
structures.groupby("atom").median()
#different molecules

structures["molecule_name"].nunique()
df.groupby('type').count()['molecule_name'].sort_values().plot(kind='barh',

                                                                color='steelblue',

                                                               figsize=(15, 8),

                                                               title='Count of Coupling Type in Train Set')
structures.groupby('atom').count()['molecule_name'].sort_values().plot(kind='barh',

                                                                color='steelblue',

                                                               figsize=(15, 5),

                                                               title='Count of Atom Type in Structures Set')
df_plot = df.drop('id', 1)

df_plot.hist(figsize=(12,12))

plt.show()
sns.pairplot(df_plot)
# Testing for normal distribution hypothesis in numerical features

test_normality = lambda x: stats.shapiro(x.fillna(0))[1] < 0.01

numerical_features = [f for f in df.columns if df.dtypes[f] != 'object']

normal = pd.DataFrame(df[numerical_features])

normal = normal.apply(test_normality)

print(normal)
# Calculate correlations

corr = df.corr(method='spearman')

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

# Heatmap

plt.figure(figsize=(12, 7))

sns.heatmap(corr,

            vmax=.5,

            mask=mask,

            #annot=True, 

            fmt='.2f',

            linewidths=.2, cmap="YlGnBu");
target = df["scalar_coupling_constant"]



# let's get some stats on the 'scalar_coupling_constant' variable

print("Statistics for the scalar_coupling_constant training dataset:\n")

print("Minimum value: {:,.2f}".format(np.min(target)))

print("Maximum value: {:,.2f}".format(np.max(target)))

print("Mean value: {:,.2f}".format(np.mean(target)))

print("Median value {:,.2f}".format(np.median(target)))

print("Standard deviation: {:,.2f}".format(np.std(target)))
#  To get a visual of the outliers, let's create a box plot.

sns.boxplot(y = target)

plt.ylabel('target')

plt.title(' ');



# count number of outliers in the variable

Q1 = target.quantile(0.25)

Q3 = target.quantile(0.75)

IQR = Q3 - Q1

print("IQR value: {}\n# of outliers: {}".format(

    IQR,

    ((target < (Q1 - 1.5 * IQR)) | (target > (Q3 + 1.5 * IQR))).sum()))
# let's plot a histogram with the fitted parameters used by the function

sns.distplot(target , fit=norm);

(mu, sigma) = norm.fit(target)

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.title('Price (Log)');

print("Skewness: %f" % target.skew())
df_sampled = df.sample(n = N, random_state = 0)



type_1JHC = df_sampled[df_sampled["type"] == "1JHC"]

type_1JHN = df_sampled[df_sampled["type"] == "1JHN"]

type_2JHC = df_sampled[df_sampled["type"] == "2JHC"]

type_2JHH = df_sampled[df_sampled["type"] == "2JHH"]

type_2JHN = df_sampled[df_sampled["type"] == "2JHN"]

type_3JHC = df_sampled[df_sampled["type"] == "3JHC"]

type_3JHH = df_sampled[df_sampled["type"] == "3JHH"]

type_3JHN = df_sampled[df_sampled["type"] == "3JHN"]
sns.set(rc={'figure.figsize':(17.7,8.27)})



ax = sns.kdeplot(type_1JHC["scalar_coupling_constant"], label = "type 1JHC")

sns.kdeplot(type_1JHN["scalar_coupling_constant"], label = "type 1JHN")



sns.kdeplot(type_2JHC["scalar_coupling_constant"], label = "type 2JHC")

sns.kdeplot(type_2JHH["scalar_coupling_constant"], label = "type 2JHH")

sns.kdeplot(type_2JHN["scalar_coupling_constant"], label = "type 2JHN")



sns.kdeplot(type_3JHC["scalar_coupling_constant"], label = "type 3JHC")

sns.kdeplot(type_3JHH["scalar_coupling_constant"], label = "type 3JHH")

sns.kdeplot(type_3JHN["scalar_coupling_constant"], label = "type 3JHN")



ax.set(xlabel='Scalar coupling constant', ylabel='Density', title='Distribution of the scalar coupling constant by type')
df_sampled = df.sample(n = N, random_state = 0)



sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})



# Initialize the FacetGrid object

pal = sns.cubehelix_palette(10, rot=-.25, light=.7)

g = sns.FacetGrid(df_sampled, row ="type", hue="type", aspect=15, height=.5, palette=pal)

# Draw the densities in a few steps

g.map(sns.kdeplot, "scalar_coupling_constant", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)

g.map(sns.kdeplot, "scalar_coupling_constant", clip_on=False, color="w", lw=1.6, bw=.2)

g.map(plt.axhline, y=0, lw=1, clip_on=False)



# Define and use a simple function to label the plot in axes coordinates

def label(x, color, label):

    ax = plt.gca()

    ax.text(0, .2, label, fontweight="bold", color=color,

            ha="left", va="center", transform=ax.transAxes)





g.map(label, "scalar_coupling_constant")



# Set the subplots to overlap

g.fig.subplots_adjust(hspace=-.25)



# Remove axes details that don't play well with overlap

g.set_titles("")

g.set(yticks=[])

g.despine(bottom=True, left=True)
df_sampled = df.sample(n = N, random_state = 0)

sns.relplot(x="scalar_coupling_constant", y="potential_energy", hue="type", data=df_sampled);
df_sampled = df.sample(n = N, random_state = 0)

sns.catplot(x="type", y="scalar_coupling_constant", kind="swarm", data=df_sampled);
#clusters with the numerical variables considered







df_numeric = df[["atom_index_0", "atom_index_1", "X", "Y", "Z", "potential_energy", "scalar_coupling_constant"]]



sse = {}

for k in range(1, 10):

    kmeans = KMeans(n_clusters=k, max_iter=1000, random_state=0).fit(df_numeric)

    df_numeric["clusters"] = kmeans.labels_

    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

plt.figure()

plt.plot(list(sse.keys()), list(sse.values()))

plt.title("Elbow Rule")

plt.xlabel("Number of clusters")

plt.ylabel("SSE")

plt.show()

df_numeric = df[["atom_index_0", "atom_index_1", "X", "Y", "Z", "potential_energy", "scalar_coupling_constant"]]

k = 3



# --- subselect data to speed up computations --- if needed just delete the following lines -------

N = 2000

df_numeric = df_numeric.sample(n = N, random_state = 0)

## ------------------------------------------------------------------------------------------------





# build the clustering model

kmeans = KMeans(n_clusters=k)

kmeans.fit(df_numeric)



# create the clusters

clusters = kmeans.predict(df_numeric)



df_numeric["clusters"] = clusters

df_numeric.head()
# apply t-SNE to the data to visualize it in 2-Dimensions



# standardize the data -------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



scaler.fit(df_numeric) # fit the scaler to the data

scaled_data = scaler.transform(df_numeric) # scale the data





## t-SNE -------

from sklearn.manifold import TSNE

# call t-SNE object

tsne = TSNE(random_state=0)

# fit it to the data: scaled data previously obtained: for t-SNE data should be also scaled

TSNE_data = tsne.fit_transform(scaled_data)



## plot ------

plt.figure(figsize=(7,7))

# loop through all the digits and add the points to the graph

for i in range(0,k):

    plt.scatter(TSNE_data[clusters == i,0], TSNE_data[clusters == i,1], alpha = 1, label = "Cluster {}".format(i))

plt.xlabel("First component")

plt.ylabel("Second Component")

plt.title("T-SNE")

plt.legend(loc='best')
df_model = df.drop('molecule_name', axis = 1)

# create validation set

from sklearn.model_selection import train_test_split



X_train = df_model.drop('scalar_coupling_constant', 1)

y_train = df_model["scalar_coupling_constant"]

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.3, random_state=0)
# Label Encoding

lbl = LabelEncoder()

lbl.fit(list(train['type'].values) + list(train['type'].values))

train['type'] = lbl.transform(list(train['type'].values))

test['type'] = lbl.transform(list(test['type'].values))
molecules = train.pop('molecule_name')

test = test.drop('molecule_name', axis=1)



y = train.pop('scalar_coupling_constant')



yoof = np.zeros(len(train))

yhat = np.zeros(len(test))



n_splits = 3

gkf = GroupKFold(n_splits=n_splits) # we're going to split folds by molecules



fold = 0

for in_index, oof_index in gkf.split(train, y, groups=molecules):

    fold += 1

    print(f'fold {fold} of {n_splits}')

    X_in, X_oof = train.values[in_index], train.values[oof_index]

    y_in, y_oof = y.values[in_index], y.values[oof_index]

    reg = RandomForestRegressor(n_estimators=250,

                                max_depth=9,

                                min_samples_leaf=3,

                                n_jobs=-1)

    reg.fit(X_in, y_in)

    yoof[oof_index] = reg.predict(X_oof)

    yhat += reg.predict(test)



yhat /= n_splits
sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='id')



benchmark = sample_submission.copy()

benchmark['scalar_coupling_constant'] = yhat

benchmark.to_csv('atomic_distance_benchmark.csv')