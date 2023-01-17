import numpy as np
import pandas as pd
from pathlib import Path
import cv2

import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use("ggplot")

import folium
from folium import plugins
flat_df = pd.read_csv("../input/flats.csv", index_col=0, header=0)
flat_df.head(3)
rnd_flat_row = flat_df.sample()
rnd_folder_name = rnd_flat_row["image_folder_name"].values[0]
rnd_image_folder_path = Path("../input/flat_images/flat_images") / rnd_folder_name
image_paths = list(rnd_image_folder_path.glob("*.jpg"))
images = [cv2.cvtColor(cv2.imread(str(x)), cv2.COLOR_RGB2BGR) for x in image_paths]
n_rows = 2
n_cols = int(np.ceil(len(images)/n_rows))

fig = plt.figure(figsize=(10, 4))
for i, image in enumerate(images):
    ax = fig.add_subplot(n_rows, n_cols, i+1)
    ax.axis("off")
    ax.imshow(image)
locations = flat_df["location"]
locations = [list(eval(x)) for x in locations]
loc = np.array(list(zip(*locations)))
lat = loc[0]
lon = loc[1]

# Dirty cleanup
lat[np.where(lat==None)] = 0
lon[np.where(lon==None)] = 0
CENTER_OF_BUDAPEST = (47.4979, 19.0402)
budapest_map = folium.Map(location=CENTER_OF_BUDAPEST,tiles = "Stamen Toner", zoom_start = 11)
budapest_map.add_child(plugins.HeatMap(list(zip(lat, lon)), radius=8))
price = flat_df["price"]
price = price.fillna(-1).values.astype(int)
sorted(price, reverse=True)[:10]
# lazy outlier removal
# 5.000.000 HUF is pretty unrealistic, so above we consider it as a outlier
price[np.where(price > 5000000)] = -1
print("Number of flats that has no price: {0}".format(len(np.where(price == -1)[0])))
fig = plt.figure(figsize=(12, 5))
fig.suptitle("Price of the flats")

ax1 = fig.add_subplot(1, 2, 1)
ax1.hist(price, bins=20);
ax1.set_xticklabels(labels=ax1.get_xticks().astype(int) // 1000, rotation=45)
ax1.set_xlabel("price (thousand)")

ax2 = fig.add_subplot(1, 2, 2)
ax2.boxplot([price], vert=False, labels=[""])
ax2.set_xticklabels(labels=ax2.get_xticks().astype(int) // 1000, rotation=45)
ax2.set_xlabel("price (thousand)");