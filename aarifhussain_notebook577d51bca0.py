import os
import cv2
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.stats import boxcox
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from os import listdir
listdir("../input/")
base_path="../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset"
train_image_path = base_path + "/train/"
some_files = listdir(train_image_path)[0:10]
some_files

extensions = [".jpeg",".jpg",".JPEG"]
def path(root_dir):
    files=[]
    for (root,directory,filenames) in os.walk(root_dir):
        for name in filenames:
            if any(ext in name for ext in extensions):
                files.append(os.path.join(root,name))
    return files
        
    
              
train_data=path("../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train")
test_data=path("../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test")
def stats(files):
    df=pd.DataFrame(index=np.arange(len(files)),columns = ["Rows", "Columns" ,"Depth" ,"img_mean" ,
                                                           "img_skew" ,"img_std" ,"channel_mean"])

    for i in tqdm(range(len(files))):
        image_path = files[i]
        img = cv2.imread(image_path)
        
        df.iloc[i]["Rows"]=img.shape[0]
        df.iloc[i]["Columns"]=img.shape[1]
        df.iloc[i]["Depth"]=img.shape[2]
        df.iloc[i]["img_mean"]=np.mean(img.flatten())
        df.iloc[i]["img_skew"]=skew(img.flatten())
        df.iloc[i]["img_std"]=np.std(img.flatten())  
        df.iloc[i]["channel_mean"]=np.mean(img[: ,: ,0])
    return df

train = stats(train_data)
train['image_paths']=train_data
test = stats(test_data)

test['image_paths'] = test_data
train =stats(train_data)
print(train)
train_image_stats.info()
test_image_stats.info()
train_image_stats.head(10)
test_image_stats.head(10)

train_image_names = train.image_paths.values
test_image_names = test.image_paths.values
print(train_image_stats)
print(test_image_stats)
train['img_area']=train['Rows']*train['Columns']
test['img_area']=train['Rows']*train['Columns']
fig, ax = plt.subplots(1,2,figsize=(20,10))
ax[0].scatter(train["Rows"].values, train["Columns"].values, c="orangered")
ax[1].scatter(test["Rows"].values, test["Columns"].values, c="lightseagreen")

ax[0].set_title("Train images")
ax[1].set_title("Test images")
def preprocess_k_means(train, test, feature, constant, lam):
    minmax_scaler = MinMaxScaler()
    scaled_train_feature = minmax_scaler.fit_transform(train[feature].values.reshape(-1, 1))
    scaled_test_feature = minmax_scaler.fit_transform(test[feature].values.reshape(-1,1))
    
    boxcox_train_feature = boxcox(scaled_train_feature[:,0] + constant, lam)
    boxcox_test_feature = boxcox(scaled_test_feature[:,0] + constant, lam)

    scaler = StandardScaler()
    preprocessed_train_feature = scaler.fit_transform(boxcox_train_feature.reshape(-1,1))
    preprocessed_test_feature = scaler.fit_transform(boxcox_test_feature.reshape(-1,1))
    
    train.loc[:, "preprocessed_" + feature] = preprocessed_train_feature
    test.loc[:, "preprocessed_" + feature] = preprocessed_test_feature
    return train, test
train, test= preprocess_k_means(train, test, "channel_mean",constant=1, lam=10)

train, test = preprocess_k_means(train, test, "img_skew",  constant=0.05,lam=2)
fig, ax = plt.subplots(1,2,figsize=(20,5))
sns.distplot(train.preprocessed_channel_mean, ax=ax[0], color="crimson", label="train")
sns.distplot(test.preprocessed_channel_mean, ax=ax[0], color="crimson", label="test")
sns.distplot(train.preprocessed_img_skew, ax=ax[1], color="crimson", label="train")
sns.distplot(test.preprocessed_img_skew, ax=ax[1], color="lightseagreen", label="test")

train_shapes = train.groupby(["Rows", "Columns"]).size().sort_values(ascending=False) / train.shape[0] * 100
test_shapes = test.groupby( ["Rows", "Columns"]).size().sort_values(ascending=False) / test.shape[0] * 100
train.shape[0] * 0.2/100
common_train_shapes = set(list(train_shapes[train_shapes > 0.3].index.values))
common_test_shapes = set(list(test_shapes[test_shapes > 0.3].index.values))
common_shape_groups = common_train_shapes.union(common_test_shapes)
common_shape_groups
num_clusters = len(common_shape_groups)
num_clusters
combined_stats = train.append(test)
combined_stats.head(1)
kmeans = KMeans(n_clusters=num_clusters, 
                random_state=0)

x = combined_stats.loc[:, ["img_mean", "img_std", "preprocessed_img_skew",
                           "preprocessed_channel_mean"]].values #,
                           #"img_area", "rows", "columns"]].values
cluster_labels = kmeans.fit_predict(x)
combined_stats["cluster_label"] = cluster_labels
train = combined_stats.iloc[0:train.shape[0]]
test= combined_stats.iloc[train.shape[0]::]

fig = make_subplots(rows=1, cols=2, subplot_titles=("Train  stats", "Test  stats"))

trace0 = go.Scatter(
    x = train.img_std.values,
    y = train.img_mean.values,
    mode='markers',
    text=train["cluster_label"].values,
    marker=dict(
        color=train.cluster_label.values,
        colorbar=dict(thickness=10, len=1.1, title="cluster label"),
        colorscale='Jet',
        opacity=0.4,
        size=2
    )
)

trace1 = go.Scatter(
    x = test.img_std.values,
    y = test.img_mean.values,
    mode='markers',
    text=test["cluster_label"].values,
    marker=dict(
        color=test.cluster_label.values,
        colorscale='Jet',
        opacity=0.4,
        size=2
    )
)

fig.add_trace(trace0, row=1, col=1)
fig.add_trace(trace1, row=1, col=2)

fig.update_xaxes(title_text="Image std", row=1, col=1)
fig.update_yaxes(title_text="Image mean", row=1, col=1)
fig.update_xaxes(title_text="Image std", row=1, col=2)
fig.update_yaxes(title_text="Image mean", row=1, col=2)

fig.update_layout(height=425, width=850, showlegend=False)
fig.show()