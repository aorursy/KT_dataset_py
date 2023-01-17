# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd



import os

import matplotlib.pyplot as plt

import seaborn as sns

print(os.listdir("/kaggle/input"))
import matplotlib as mpl



mpl.rcParams['figure.figsize'] = [15, 7]

mpl.rcParams['figure.dpi'] = 80

mpl.rcParams['savefig.dpi'] = 100



mpl.rcParams['font.size'] = 14

mpl.rcParams['legend.fontsize'] = 'large'

mpl.rcParams['figure.titlesize'] = 'medium'
!head -n 5 /kaggle/input/expedia-hotel-recommendations/train.csv
!wc -l /kaggle/input/expedia-hotel-recommendations/train.csv
data = pd.read_csv("/kaggle/input/expedia-hotel-recommendations/train.csv", nrows=100000)
CATEGORICALS = ["site_name", "posa_continent", "user_location_country", "user_location_region", "user_location_city", "is_mobile", "is_package", "channel", 

               "srch_destination_type_id", "hotel_continent", "hotel_country", "hotel_market", "srch_destination_id"]

NUMERICALS = ["orig_destination_distance", "srch_adults_cnt", "srch_children_cnt", "srch_rm_cnt", "cnt"]

USER_ID = "user_id"

IS_BOOKING = "is_booking"

HOTEL_CLUSTER = "hotel_cluster"
ax = data[HOTEL_CLUSTER].value_counts().plot.bar(color='dodgerblue')

ax.set_xticklabels([])

plt.title("Hotel clusters' clicks - Looks like the beginnings of a power-law distribution");
data_cat_dummies = pd.get_dummies(pd.get_dummies(data[CATEGORICALS].astype('category')))
def name_scores(featurecoef, col_names, label="Score", sort=False):

    df_feature_importance = pd.DataFrame([dict(zip(col_names, featurecoef))]).T.reset_index()

    df_feature_importance.columns = ["Feature", label]

    if sort:

        return df_feature_importance.sort_values(ascending=False, by=label)

    return df_feature_importance
from sklearn.feature_selection import chi2



sample_n = 10000

data_cat_dummies_sample = data_cat_dummies.sample(sample_n)

chi2_scores = chi2(data_cat_dummies_sample, data[HOTEL_CLUSTER].loc[data_cat_dummies_sample.index])

df_chi2_scores = name_scores(chi2_scores[0], data_cat_dummies_sample.columns)

df_chi2_scores = df_chi2_scores.sort_values(by="Score", ascending=False)
# get the top 100 features and graph the categorical

# 상위 100개를 뽑은 후 그래프로 범주화한다.

n = 100

top_n_features = df_chi2_scores[:n]["Feature"]
top_n_features.apply(lambda s : ' '.join(s.split("_")[:len(s.split("_"))-1])).value_counts()
def create_matrix(data, group_column, val_column):

    grouped_data = data.groupby(group_column)[val_column].value_counts()

    grouped_data = grouped_data.groupby(level=0).nlargest(3)

    grouped_data.index = grouped_data.index.droplevel(0)

    # transfrom to a square matrix

    grouped_data_unstacked = grouped_data.unstack()

    return grouped_data_unstacked.fillna(0)
hotel_user_city_vc_matrix = create_matrix(data, "hotel_cluster", "user_location_city")

hotel_market_vc_matrix = create_matrix(data, "hotel_cluster", "hotel_market")

hotel_country_vc_matrix = create_matrix(data, "hotel_cluster", "hotel_country")

hotel_continent_vc_matrix = create_matrix(data, "hotel_cluster", "hotel_continent")



# hotel user city

fig, axes = plt.subplots(2, 2, figsize=(15, 15))



axes[0][0].set_title("Looks like user city ~17 and ~90 is very active\n all throughout the different hotel clusters")

axes[0][0].imshow(hotel_user_city_vc_matrix, cmap='gray')



axes[0][1].set_title("Looks like hotel_market ~59 and 62 are\nrelated to many hotel_clusters")

axes[0][1].imshow(hotel_market_vc_matrix, cmap='gray')



axes[1][0].set_title("Looks like hotel_country ~8 is\nrelated to many hotel_clusters (France?)")

axes[1][0].imshow(hotel_country_vc_matrix, cmap='gray');



axes[1][1].set_title("I think hotel_continent 1 is Europe!")

axes[1][1].imshow(hotel_continent_vc_matrix, cmap='gray');
sns.distplot(data["orig_destination_distance"].dropna())



plt.title("Majority of destination places are close to the origin. \nThus, let's just use the median for imputation.");
sample_n = 10000

data_numericals_sample = data[NUMERICALS].sample(sample_n)

chi2_scores = chi2(data_numericals_sample.fillna(data_numericals_sample.median()), data[HOTEL_CLUSTER].loc[data_numericals_sample.index])

df_chi2_scores = name_scores(chi2_scores[0], data_numericals_sample.columns)



df_chi2_pvalues = name_scores(chi2_scores[1], data_numericals_sample.columns)

df_chi2_pvalues.columns = ["Feature", "PValue"]



df_chi2_scores.merge(df_chi2_pvalues).sort_values(by="Score", ascending=False)
sns.boxplot(data=data, x=HOTEL_CLUSTER, y="orig_destination_distance")



plt.title("There's a lot of outlier distances for each hotel_cluster");
data_cluster = data[[HOTEL_CLUSTER]].copy()

data_cluster["cluster_num"] = pd.cut(data_cluster[HOTEL_CLUSTER], bins=4, labels=range(4))
clusters_1 = data[data[HOTEL_CLUSTER].isin(data_cluster.loc[data_cluster["cluster_num"] == 0, HOTEL_CLUSTER])]

clusters_1_unstacked = clusters_1.groupby(HOTEL_CLUSTER)["srch_children_cnt"].value_counts().unstack().fillna(0)



clusters_1_unstacked= clusters_1_unstacked.drop(0, axis=1).sort_values(by=2,)

ax = clusters_1_unstacked.plot.barh(stacked=True)

ax.legend(loc='upper right')



plt.title("Some hotel_clusters are more perceived to be family friendly", );
print("Search destination id's nunique: ", data["srch_destination_id"].nunique())

print("Search destination types nunique: ", data["srch_destination_type_id"].nunique())
hotel_search_type_matrix = create_matrix(data, HOTEL_CLUSTER, "srch_destination_type_id")

print("Looks like the search types tend around 1 and 6.")

display(hotel_search_type_matrix[:10])

plt.imshow(hotel_search_type_matrix, cmap='gray');
hotel_to_search_n = data.groupby(HOTEL_CLUSTER)["srch_destination_id"].nunique()

ax = sns.distplot(hotel_to_search_n, bins=50, kde=False)

ax.set_xlabel("Number of unique search IDs per hotel cluster")



display(data.groupby(HOTEL_CLUSTER)["srch_destination_id"].nunique().describe().to_frame(

    "Stats of the number of unique search IDs per hotel cluster"))
destinations = pd.read_csv("/kaggle/input/expedia-hotel-recommendations/destinations.csv")
def create_latent_search_img(index, ax):

    # to make the image 10x15, we create a 150th feature with the mean of the array

    img = np.array(destinations.loc[index].values[1:].tolist() + [destinations.loc[index].values[1:].mean()])

    img = img.reshape((15,10))

    sns.heatmap(img, cmap='gray', ax=ax)

    ax.set_xticklabels([])

    ax.set_yticklabels([])

    return ax
ax1 = create_latent_search_img(0, plt.subplot(2, 3, 1))

ax2 = create_latent_search_img(1, plt.subplot(2, 3, 2))

ax3 = create_latent_search_img(2, plt.subplot(2, 3, 3))

ax4 = create_latent_search_img(3, plt.subplot(2, 3, 4))

ax5 = create_latent_search_img(3, plt.subplot(2, 3, 5))

ax6 = create_latent_search_img(3, plt.subplot(2, 3, 6))
def get_latent_search_hotel_array(hotel_cluster_index):

    values = data.loc[data[HOTEL_CLUSTER] == hotel_cluster_index, "srch_destination_id"].to_frame().merge(destinations)

    values = values.drop("srch_destination_id", axis=1)

    values = values.sum()

    return values



def create_latent_search_hotel_image(hotel_cluster_index, ax):

    img = get_latent_search_hotel_array(hotel_cluster_index)

    img = np.array(img.tolist() + [img.mean()])

    img = img.reshape((15,10))

    

    ax.imshow(img, cmap='gray')

    ax.set_title("Cluster " + str(hotel_cluster_index))

    ax.set_xticklabels([])

    ax.set_yticklabels([])

    return ax
hotel_arrays = []

for i in range(data[HOTEL_CLUSTER].nunique()):

    hotel_arrays.append(get_latent_search_hotel_array(i))

hotel_arrays = np.array(hotel_arrays)
from scipy.cluster import hierarchy

import matplotlib.pyplot as plt



Z = hierarchy.linkage(hotel_arrays, 'single')

plt.figure(figsize=(20, 8))

dn = hierarchy.dendrogram(Z)



# We change the fontsize of minor ticks label 

plt.tick_params(axis='both', which='major', labelsize=10)

plt.tick_params(axis='both', which='minor', labelsize=10)

plt.xticks(rotation=0)

plt.show()
fig = plt.figure(figsize=(20, 20))



for i in range(data[HOTEL_CLUSTER].nunique()):

    create_latent_search_hotel_image(i, fig.add_subplot(10, 10, i+1))
user_id_col = "user_id"

item_id_col = HOTEL_CLUSTER

ratings = data[[user_id_col, item_id_col]]



num_users = ratings[user_id_col].nunique()

num_items = ratings[item_id_col].nunique()

possible_combinations = num_users * num_items

nnz = len(ratings)

nnz_percent = nnz / possible_combinations



print("Num Users:", num_users)

print("Num Items:", num_items)

print("Sparsity:", nnz_percent)

print("Not very sparse. CF will work wonders here.")



# average number of hotel_clusters per user

# 사용자 당 평균 호텔 군집 수

hotel_per_user = ratings.groupby(user_id_col)[item_id_col].nunique()



fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(211)

sns.distplot(hotel_per_user, kde=False, ax=ax)

ax.set_title("Mean number of clusters per user: {:.2f}".format(hotel_per_user.mean()))



# average number of users per hotel

# 호텔당 평균 사용자 수

user_per_hotel = ratings.groupby(item_id_col)[user_id_col].nunique()

ax = fig.add_subplot(212)

sns.distplot(user_per_hotel, kde=False, ax=ax)

ax.set_title("Mean number of users per cluster: {:.2f}".format(user_per_hotel.mean()))



fig.tight_layout()