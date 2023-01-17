import os

print(os.listdir("../input"))

import pandas as pd

from matplotlib import pyplot as plt

import warnings

warnings.filterwarnings('ignore')

# Import the campaign dataset from Excel (Sheet 0 = Non Responders, Sheet 1 = Responders)

campaign_type = pd.read_excel("../input/campaign_response.xlsx", sheetname = 0)



# Adding a column of value 1 to act as a count for the response instance

campaign_response = pd.read_excel("../input/campaign_response.xlsx", sheetname = 1)

campaign_response["response"] = 1
print ("Contains campaign size of: " + str(len(campaign_type)))

campaign_type.head(10)
print ("Contains patient response size of: " + str(len(campaign_response)))

campaign_response.head(10)
# Merge on the CampaignID columns

merge_campaign = pd.merge(campaign_type, campaign_response, on = "CampaignID")

print ("\n The shape of the merged campaign dataset is: " + str(merge_campaign.shape))

merge_campaign.head()
pivot_campaign = merge_campaign.pivot_table(index = ["Patient"], columns = ["CampaignID"], values = "response")

pivot_campaign.head()
pivot_campaign = pivot_campaign.fillna(0).reset_index()

pivot_campaign.head()
import pylab as pl

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
# Create K Clusters of 15

k = range(1, 15)

# Instantiate and Fit KMeans of Clusters 1-15

kmeans = [KMeans(n_clusters=i) for i in k]

score = [kmeans[i].fit(pivot_campaign[pivot_campaign.columns[2:]]).score(pivot_campaign[pivot_campaign.columns[2:]]) for i in range(len(kmeans))]

# Plot the Elbow Method

#pl.plot(k,score)

#pl.xlabel('Number of Clusters')

#pl.ylabel('Score')

#pl.title('Elbow Curve')

#pl.show()
# Choose Cluster Size of 3

cluster = KMeans(n_clusters = 3) # At least 7-times times cluster = patients

# Predict the cluster from first patient down all the rows

pivot_campaign["cluster"] = cluster.fit_predict(pivot_campaign[pivot_campaign.columns[2:]])
pivot_campaign.head()
# Principal component separation to create a 2-dimensional picture

pca = PCA(n_components = 2)

pivot_campaign['x'] = pca.fit_transform(pivot_campaign.iloc[:,1:33])[:,0]

pivot_campaign['y'] = pca.fit_transform(pivot_campaign.iloc[:,1:33])[:,1]

pivot_campaign = pivot_campaign.reset_index()
pivot_campaign.head()
kmeans_colors = ['green' if c == 0 else 'blue' if c == 2 else 'red' for c in cluster.labels_]

fig = plt.figure(figsize=(10, 6))

plt.scatter(x="x",y="y", data=pivot_campaign,alpha=0.25,color = kmeans_colors)

plt.xlabel("PC1")

plt.ylabel("PC2")

plt.title("KMeans Clusters")

plt.show()
# Tidy up our Data

campaign_cluster = pivot_campaign[["Patient", "cluster", "x", "y"]]

campaign_cluster.head()
# Merge back together

final_campaign = pd.merge(campaign_response, campaign_cluster)

final_campaign = pd.merge(campaign_type, final_campaign)

final_campaign.head()
# e-mails, short-message services, WhatsApp messages, pamphlets, telephone and long letters

final_campaign["cluster_1"] = final_campaign.cluster == 0

final_campaign.groupby("cluster_1").Type.value_counts()
# e-mails, short-message services, WhatsApp messages, pamphlets, telephone and long letters

final_campaign["cluster_2"] = final_campaign.cluster == 1

final_campaign.groupby("cluster_2").Type.value_counts()
# e-mails, short-message services, WhatsApp messages, pamphlets, telephone and long letters

final_campaign["cluster_3"] = final_campaign.cluster == 2

final_campaign.groupby("cluster_3").Type.value_counts()
# Number of patients in this cluster

print("Total respondents: " + str(final_campaign[final_campaign.cluster == 0]["Patient"].count()))



# Show all respondents to cluster 1

final_campaign[final_campaign.cluster == 0]["Patient"]