import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_csv('../input/train.csv')

y_train = train['target'].values
train.drop(['ID', 'target'], axis=1, inplace=True)

X_train = train.values
feature_names = train.columns.values
olivier_features= [
        'f190486d6', 'c47340d97', 'eeb9cd3aa', '66ace2992', 'e176a204a',
        '491b9ee45', '1db387535', 'c5a231d81', '0572565c2', '024c577b9',
        '15ace8c9f', '23310aa6f', '9fd594eec', '58e2e02e6', '91f701ba2',
        'adb64ff71', '2ec5b290f', '703885424', '26fc93eb7', '6619d81fc',
        '0ff32eb98', '70feb1494', '58e056e12', '1931ccfdd', '1702b5bf0',
        '58232a6fb', '963a49cdc', 'fc99f9426', '241f0f867', '5c6487af1',
        '62e59a501', 'f74e8f13d', 'fb49e4212', '190db8488', '324921c7b',
        'b43a7cfd5', '9306da53f', 'd6bb78916', 'fb0f5dbfe', '6eef030c1'
    ]
def len_of_intersection(feature, target):
    return len(set(target).intersection(set(feature)))


intersection_features = np.apply_along_axis(
    len_of_intersection, arr=X_train, axis=0, target=y_train
)
intersection_features_top40 = feature_names[
    np.argsort(intersection_features)[-40:]
]
len_of_intersection(intersection_features_top40, olivier_features)
set(olivier_features) - set(intersection_features_top40)
np.where(feature_names[
        np.argsort(intersection_features)[::-1]
    ] == '9306da53f'
)[0][0]
giba_feature = 'f190486d6'
giba_feature in set(intersection_features_top40)
np.where(intersection_features_top40 == giba_feature)[0][0]
df = pd.DataFrame()
df['len_of_intersection'] = np.sort(intersection_features)[-40:]
df['name_of_feature'] = intersection_features_top40
df.plot.barh(y='len_of_intersection', x='name_of_feature', figsize=(6, 12), rot=33)
plt.show()
from collections import Counter

Counter(intersection_features != 0.0)

def corr_with_target(feature, target):
    # for exactly zeros feature return 0.0
    if np.count_nonzero(feature) == 0:
        return 0.0
    return np.corrcoef(feature, target)[0, 1]

corr_features = np.apply_along_axis(
    corr_with_target, arr=X_train, axis=0, target=y_train
)
from sklearn.cluster import KMeans
cluster_features = KMeans(n_clusters=2, random_state=0).fit_predict(
    np.column_stack([corr_features, intersection_features])
)
len(cluster_features)
plt.figure(figsize=(8, 6))
plt.scatter(corr_features, intersection_features, c=cluster_features)
plt.plot(corr_features[np.where(feature_names == '9306da53f')],
         intersection_features[np.where(feature_names == '9306da53f')], 'o', markersize=15, label='olivier_outlier', c='red')
plt.plot(corr_features[np.where(feature_names == 'f190486d6')],
         intersection_features[np.where(feature_names == 'f190486d6')], 'o', markersize=15, label='giba_feature')
plt.legend()
plt.show()
cluster1_fnames = feature_names[cluster_features == 1]
len(cluster1_fnames)
len(set(cluster1_fnames).intersection(set(olivier_features)))
len(set(cluster1_fnames).intersection(set(intersection_features_top40)))
np.save('intersection_features', intersection_features_top40)
np.save('olivier_features', olivier_features)