import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

airbnb_data = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
airbnb_data.tail()
print('Host name :\n')
print(airbnb_data['host_name'].value_counts())
print('----------------------------------------')
print('Neighbourhood Group :\n')
print(airbnb_data['neighbourhood_group'].value_counts())
print('----------------------------------------')
print('Room Type :\n')
print(airbnb_data['room_type'].value_counts())
print('----------------------------------------')
print('Price :')
print(' - Max : {}'.format(airbnb_data['price'].max()))
print(' - Mean : {}'.format(airbnb_data['price'].mean()))
print(' - Min : {}'.format(airbnb_data['price'].min()))
print('----------------------------------------')
print('Minimum Night :')
print(' - Max : {}'.format(airbnb_data['minimum_nights'].max()))
print(' - Mean : {}'.format(airbnb_data['minimum_nights'].mean()))
print(' - Min : {}'.format(airbnb_data['minimum_nights'].min()))
print('----------------------------------------')
print('Number of review :')
print(' - Max : {}'.format(airbnb_data['number_of_reviews'].max()))
print(' - Mean : {}'.format(airbnb_data['number_of_reviews'].mean()))
print(' - Min : {}'.format(airbnb_data['number_of_reviews'].min()))
print('----------------------------------------')
print('Last review :\n')
print(airbnb_data['last_review'].value_counts())
print('----------------------------------------')
print('Review per month :')
print(' - Max : {}'.format(airbnb_data['reviews_per_month'].max()))
print(' - Mean : {}'.format(airbnb_data['reviews_per_month'].mean()))
print(' - Min : {}'.format(airbnb_data['reviews_per_month'].min()))
print('----------------------------------------')
print('Calculated host listing count :')
print(' - Max : {}'.format(airbnb_data['calculated_host_listings_count'].max()))
print(' - Mean : {}'.format(airbnb_data['calculated_host_listings_count'].mean()))
print(' - Min : {}'.format(airbnb_data['calculated_host_listings_count'].min()))
print('----------------------------------------')
print('Availability 365 :')
print(' - Max : {}'.format(airbnb_data['availability_365'].max()))
print(' - Mean : {}'.format(airbnb_data['availability_365'].mean()))
print(' - Min : {}'.format(airbnb_data['availability_365'].min()))
print('----------------------------------------')
from datetime import datetime
import time

def str_to_timestamp(dt_str):
    return time.mktime(datetime.strptime(dt_str, '%Y-%m-%d').timetuple())
from sklearn.cluster import KMeans

tmp_df = airbnb_data.copy(True)

# Replace NaN
tmp_df['minimum_nights'] = tmp_df.apply(lambda x: 0 if pd.isna(x['minimum_nights']) else x['minimum_nights'], axis=1)
tmp_df['number_of_reviews'] = tmp_df.apply(lambda x: 0 if pd.isna(x['number_of_reviews']) else x['number_of_reviews'], axis=1)
tmp_df['last_review'] = tmp_df.apply(lambda x: str_to_timestamp('1970-01-01') if pd.isna(x['last_review']) else str_to_timestamp(x['last_review']), axis=1)
tmp_df['reviews_per_month'] = tmp_df.apply(lambda x: 0 if pd.isna(x['reviews_per_month']) else x['reviews_per_month'], axis=1)
tmp_df['calculated_host_listings_count'] = tmp_df.apply(lambda x: 0 if pd.isna(x['calculated_host_listings_count']) else x['calculated_host_listings_count'], axis=1)
tmp_df['availability_365'] = tmp_df.apply(lambda x: 0 if pd.isna(x['availability_365']) else x['availability_365'], axis=1)
# Pick some
# tmp_df = tmp_df[:5000]

tmp_df = tmp_df.drop(['id', 'name', 'host_name', 'neighbourhood'], axis=1)
tmp_df
tmp_df['neighbourhood_group'].unique()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

# Label encode in important field
tmp_df['room_type'] = le.fit_transform(tmp_df['room_type'])
tmp_df['neighbourhood_group'] = le.fit_transform(tmp_df['neighbourhood_group'])
true_labels = tmp_df['neighbourhood_group']
n_clusters = len(true_labels.unique())
print('True labels : ', true_labels)
print('N-Cluster : ', n_clusters)
row_partition_rate = 0.7
row_split_index = round(len(tmp_df) * row_partition_rate)
X_Train, X_Test, y_test = tmp_df[:row_split_index], tmp_df[row_split_index+1:], tmp_df[row_split_index+1:]['neighbourhood_group']
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
preprocessor = Pipeline(
    [
        ("scaler", preprocessing.StandardScaler()),
        ("pca", PCA(n_components=2, random_state=42)),
    ]
)
clusterer = Pipeline(
    [
        (
            "kmeans",
            KMeans(
                n_clusters=n_clusters,
                init="k-means++",
                n_init=50,
                max_iter=1000,
                random_state=42,
            ),
        ),
    ]
)
pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("clusterer", clusterer)
    ]
)
pipe.fit(X_Train)
y_pred = pipe.predict(X_Test)
preprocessed_data = pipe["preprocessor"].transform(X_Test)
predicted_labels = y_pred
from sklearn.metrics import silhouette_score
silhouette_score(preprocessed_data, y_pred)
from sklearn.metrics import adjusted_rand_score
adjusted_rand_score(y_test, y_pred)
import seaborn as sns

pcadf = pd.DataFrame(
    pipe["preprocessor"].transform(X_Test),
    columns=["component_1", "component_2"],
)

pcadf["predicted_cluster"] = y_pred

# Decoding label(neighbourhood group)
pcadf["true_label"] = le.inverse_transform(y_test)

plt.style.use("fivethirtyeight")
plt.figure(figsize=(8, 8))

scat = sns.scatterplot(
    "component_1",
    "component_2",
    s=50,
    data=pcadf,
    hue="predicted_cluster",
    style="true_label",
    palette="Set2",
)

# Plot cluster centers
cc = pipe["clusterer"]["kmeans"].cluster_centers_
plt.scatter(cc[:, 0], cc[:, 1], c='black', s=200, alpha=0.5);

scat.set_title(
    "Clustering results from Neighbourhood Group\nHost in each borough of New York City"
)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
plt.show()
unique_label = true_labels.unique()
print('Legend Encoding : ', unique_label)
print('Legend Name : ', le.inverse_transform(unique_label))
X_Test
chart_df = X_Test.copy(True)

# Removed no meaning info
chart_df = chart_df.drop(['host_id'], axis=1)

# Pick some
chart_df = chart_df[:500]
sns.clustermap(chart_df, metric="euclidean", standard_scale=1, method="ward")