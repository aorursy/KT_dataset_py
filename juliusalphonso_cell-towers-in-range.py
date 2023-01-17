import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px # plotting
from tqdm import tqdm

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
def haverside_distance_squared(lat1, lon1, center):
    test = tf
    pbar.update(1)
    if center in test['LocationCode'].values:
        return center
    
    r = 6378
    phi1 = np.radians(lat1)
    phi2 = np.radians(test['latitude'])
    delta_phi = np.radians(test['latitude'] - lat1)
    delta_lambda = np.radians(test['longitude'] - lon1)
    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    c = np.sin(3 / (2 * r))**2
    close_centers = test['LocationCode'][a.values<c]
    if len(close_centers) > 0:
      return close_centers.values[0]
    else:
      return np.nan
df = pd.read_parquet('../input/cell-towers-in-india/towers_with_centers.parquet')
df.head()
tf = pd.read_json('../input/crfotestingcenters/testCenters.json')
tf.head()
with tqdm(total=len(df)) as pbar:
    df['center'] = df.apply(lambda x: haverside_distance_squared(x['lon'], x['lat'], x['center']), axis=1)
df.dropna(inplace=True)
df.to_csv('towers_in_range.csv', float_format="%.6f") # Download the file output/kaggle/working/towers_in_range.csv
output = pd.read_csv('./towers_in_range.csv')
df.head()
