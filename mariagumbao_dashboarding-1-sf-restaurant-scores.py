import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../input/restaurant-scores-lives-standard.csv')
# Overview of the data
print(data.shape)
data = data.dropna()
data.inspection_date = pd.to_datetime(data.inspection_date, format='%Y-%m-%dT%H:%M:%S')
data.head()
data = data[data.business_latitude > 37]
data.describe()
score = data.groupby('business_id')['business_latitude','business_longitude',
                                    'inspection_score'].mean()
plt.figure(figsize=(10,7))
_=plt.scatter(score.business_longitude, 
              score.business_latitude,
              c = score.inspection_score,
              alpha=1
             )
plt.title('Spacial analysis of resaturant scores', fontsize=20)
plt.xlabel('Longitude', fontsize=16)
plt.ylabel('Latitude', fontsize=16)
cbar=plt.colorbar()
cbar.set_label('Inspection score', fontsize=16)
plt.tight_layout()
# Score relation with date of the inspection, is there bias?
date = data.groupby('business_id')['inspection_date','inspection_score'].max()
weekday = date.inspection_date.dt.weekday

plt.figure(figsize=(10,7))
_=plt.scatter(list(date.inspection_date), 
              list(date.inspection_score),
              c=weekday.values
             )
plt.title('Score vs Time', fontsize=20)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Score', fontsize=16)
cbar=plt.colorbar()
cbar.set_label('Weekday', fontsize=16)
plt.show()