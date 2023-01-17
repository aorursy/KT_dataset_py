import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline


df = pd.read_csv('../input/tweets.csv', parse_dates=['time'],
                 infer_datetime_format=True, encoding="utf8")

hour_profiles = df.groupby('username').apply(lambda x: x.time.dt.hour.value_counts()) \
    .unstack().fillna(0)
hour_profiles.index = [w[0:40] for w in hour_profiles.index]
hour_profiles.columns = ["%04d" % (x * 100) for x in range(24)]
hour_profiles.iloc[:5,:5]
g = sns.clustermap(hour_profiles, z_score=0,
                   col_cluster=False, row_cluster=True,
                   figsize=(20, 30))
wmult=.25
hmult=.75
hm_pos = g.ax_heatmap.get_position()
rd_pos = g.ax_row_dendrogram.get_position()
cd_pos = g.ax_col_dendrogram.get_position()
g.ax_heatmap.set_position([hm_pos.x0 - rd_pos.width * (1-hmult),
                           hm_pos.y0,
                           hm_pos.width * wmult,
                           hm_pos.height])
g.ax_row_dendrogram.set_position([rd_pos.x0,
                                  rd_pos.y0,
                                  rd_pos.width * hmult,
                                  rd_pos.height])
g.ax_col_dendrogram.set_position([cd_pos.x0 - rd_pos.width * (1-hmult),
                                  cd_pos.y0,
                                  cd_pos.width*wmult,
                                  cd_pos.height/2])
#g.cax.set_visible(False)
g.ax_heatmap.set_title("Post Activity by Hour of the Day", fontsize=20)
plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), fontsize=10, rotation=0)
plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), fontsize=10, rotation=90)
plt.show()
from sklearn.preprocessing import StandardScaler

df['weekday'] = df.time.dt.dayofweek
df['hour'] = df.time.dt.hour
top50_users = df[df.username.isin(df.username.value_counts()[:50].keys().tolist())]
profile = top50_users.groupby('username').apply(lambda x: pd.crosstab(x.hour, x.weekday)).unstack().fillna(0)
profile = profile.applymap(lambda x: 1 if x > 0 else 0)
profile = pd.DataFrame(StandardScaler().fit_transform(profile), index=profile.index)
profile.iloc[10:14,10:14]
from scipy.spatial.distance import cosine

sims = np.dot(profile, profile.T)
square_mag = np.diag(sims)
inv_square_mag = 1 / square_mag
inv_square_mag[np.isinf(inv_square_mag)] = 0
inv_mag = np.sqrt(inv_square_mag)
cosine = sims * inv_mag
cossims = pd.DataFrame(sims * inv_mag, index=profile.index, columns=profile.index)
cossims.values[[np.arange(len(cossims))]*2] = 0

print(cossims['RamiAlLolah'].sort_values(ascending=False)[1:11])
print(cossims['RamiAlLolah'].sort_values(ascending=False)[-10:])
g = sns.clustermap(cossims, figsize=(15, 15))
plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), fontsize=10, rotation=0)
plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), fontsize=10, rotation=90)
plt.show()
