import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
calls = pd.read_csv('../input/911.csv')

calls = calls.assign(incident=calls.title.str.split(':', expand=True)[0])
calls = calls.assign(reason=calls.title.str.split(':', expand=True)[1])

calls.drop(['desc', 'title', 'e'], 1, inplace = True)

calls.timeStamp = pd.to_datetime(calls.timeStamp)
year = calls.timeStamp.dt.year
month = calls.timeStamp.dt.month
day = calls.timeStamp.dt.day
day_of_week = calls.timeStamp.dt.dayofweek
hour = calls.timeStamp.dt.hour

calls.head()
plt.pie(calls.incident.value_counts().values, labels = calls.incident.value_counts().index, autopct = '%1.2f%%', explode = (0.01, 0.01, 0.01))
plt.title('911 calls by type of incident')
plt.show()
calls.groupby(['incident', year]).size().unstack()
ems_by_year = calls[calls.incident == 'EMS'].groupby(['incident', year]).size().tolist()            # [3898, 70127, 70669, 16747]
fire_by_year = calls[calls.incident == 'Fire'].groupby(['incident', year]).size().tolist()          # [1095, 21577, 20451, 5796]
traffic_by_year = calls[calls.incident == 'Traffic'].groupby(['incident', year]).size().tolist()    # [2923, 50656, 48497, 13989]
y_labels = year.unique().tolist()
indexes = np.arange(len(y_labels))

plt.bar(indexes, fire_by_year, label = 'Fire', bottom = np.array(ems_by_year) + np.array(traffic_by_year))
plt.bar(indexes, traffic_by_year, label = 'Traffic', bottom = np.array(ems_by_year))
plt.bar(indexes, ems_by_year, label = 'EMS')
plt.xticks(indexes, y_labels); plt.xlabel('Year'); plt.ylabel('Calls'); plt.legend(loc = 'best');
plt.show()
plt.figure(figsize=(20, 5.5))
#plt.imshow(calls.groupby([year, month]).count()['twp'].unstack(), cmap='hot', interpolation='nearest') # equivalent in matplotlib
sns.heatmap(calls.groupby([year, month]).count()['twp'].unstack(), cmap = 'viridis', linewidths = 1, annot = True, fmt = '1.0f')
plt.title('911 calls for month by year'); plt.xlabel('Month'); plt.ylabel('Year');
plt.show()