%matplotlib inline
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import pandas as pd
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
print(df.shape)
df.head()
#dataset = df.drop(columns=['host_name', 'id', "last_review"]).to_numpy()
df.drop(['last_review', 'host_name','id'], axis = 1, inplace= True)
df.fillna({'reviews_per_month':0}, inplace=True)
df.head()
def plotDendogram(df_grouped,W,H):
  list_grouped_price_mean = [ [price] for price in df_grouped.price.to_list()]
  Z = linkage(list_grouped_price_mean, 'complete')
  fig = plt.figure(figsize=(W, H))
  dendrogram(Z, labels=df_grouped.neighbourhood.to_list(), leaf_rotation=0, orientation="left")
  my_palette = plt.cm.get_cmap("Accent", 3)
  plt.show()

df.groupby("room_type").count()[["name"]]
roomtype1_df = df.groupby("room_type").get_group("Entire home/apt")
roomtype1_df.head()
roomtype1_grouped_df = roomtype1_df.groupby("neighbourhood").mean().reset_index().loc[:,['neighbourhood','price']]
roomtype1_grouped_df
plotDendogram(roomtype1_grouped_df,20,15)
roomtype2_df = df.groupby("room_type").get_group("Private room")
roomtype2_df.head()
roomtype2_grouped_df = roomtype2_df.groupby("neighbourhood").mean().reset_index().loc[:,['neighbourhood','price']]
roomtype2_grouped_df
plotDendogram(roomtype2_grouped_df,20,15)
roomtype3_df = df.groupby("room_type").get_group("Shared room")
roomtype3_df.head()
roomtype3_grouped_df = roomtype3_df.groupby("neighbourhood").mean().reset_index().loc[:,['neighbourhood','price']]
roomtype3_grouped_df
plotDendogram(roomtype3_grouped_df,20,15)