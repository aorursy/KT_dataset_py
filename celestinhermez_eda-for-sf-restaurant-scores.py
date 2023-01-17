import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas
from shapely.geometry import Point

%matplotlib inline
# Load the dataset and visualize the first few rows
scores = pd.read_csv('../input/restaurant-scores-lives-standard.csv')
scores.head()
# Visualize the number of distinct businesses
scores.business_id.nunique()
# Visualize the distribution of inspection scores
# Count unique inspection ID's since each violation is a different row
score_distribution = scores.groupby('inspection_score', as_index=False).agg({'inspection_id':'nunique'}).rename({'inspection_id': 'count'}, axis=1)
plt.bar(score_distribution.inspection_score, score_distribution['count'])
plt.xlabel('Inspection Score')
plt.ylabel('Frequency')
plt.show();
# We visualize average inspection score per parts of the city
scores['Coordinates'] = list(zip(scores.business_longitude, scores.business_latitude))
geo_scores_df = scores.groupby('Coordinates', as_index=False).agg({'inspection_score': 'mean'})
geo_scores_df['Coordinates'] = geo_scores_df['Coordinates'].apply(Point)

geo_scores = geopandas.GeoDataFrame(geo_scores_df, geometry='Coordinates')
# Visualize average score geographically
min_lat = scores.dropna().business_latitude.min() - 0.01
max_lat = scores.dropna().business_latitude.max() + 0.01
max_lon = scores.dropna().business_longitude.max() + 0.01
min_long = scores.dropna().business_longitude.min() - 0.01
vmin = geo_scores_df.inspection_score.min()
vmax = geo_scores_df.inspection_score.max()

# Create colorbar as a legend
sm = plt.cm.ScalarMappable(cmap='Spectral', norm=plt.Normalize(vmin=vmin, vmax=vmax))
# empty array for the data range
sm._A = []
# add the colorbar to the figure
geo_scores.plot(cmap='Spectral')
plt.ylim(min_lat,max_lat)
plt.xlim(min_long, max_lon)
plt.colorbar(sm)
plt.show();
