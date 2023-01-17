# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")

labels = np.array(df['room_type'])
df = df.iloc[:, 0:17]
df.head()
room_data = df[['latitude', 'longitude', 'price', 'number_of_reviews']]
room_data.head()
room_data = room_data.iloc[:60, :]
room_data.describe()
Z = linkage(room_data, 'ward')
plt.figure(figsize=(50, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Airbnb Dataset')
plt.ylabel('distance')
dendrogram(
    Z,
    labels=labels,
    leaf_rotation=90.,
    leaf_font_size=8.,
)
plt.show()