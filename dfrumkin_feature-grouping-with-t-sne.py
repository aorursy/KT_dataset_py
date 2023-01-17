import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/train.csv').drop(columns=['ID'])
def build_histograms(df):
    df_X = (df.replace(0, np.nan).apply(np.log) * 10).round()
    start = int(df_X.min().min())
    stop = int(df_X.max().max())
    return pd.DataFrame(data={f'bucket{cnt}': (df_X == cnt).sum() for cnt in range(start, stop + 1)})
df_hist = build_histograms(train)
tsne_res = TSNE(n_components=2, verbose=0).fit_transform(df_hist.values)
FEATURES40 = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1', 
              '15ace8c9f', 'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9', 
              'd6bb78916', 'b43a7cfd5', '58232a6fb', '1702b5bf0', '324921c7b', 
              '62e59a501', '2ec5b290f', '241f0f867', 'fb49e4212', '66ace2992', 
              'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', '1931ccfdd', 
              '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a', 
              '6619d81fc', '1db387535', 'fc99f9426', '91f701ba2', '0572565c2', 
              '190db8488', 'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98']

def get_int_cols(df):
    return df.columns[df.dtypes == np.int64]

def get_colors(df):
    colors = pd.Series(index=df.columns, data='b')
    colors[FEATURES40] = 'y'
    colors[get_int_cols(train)] = 'g'
    colors['target'] = 'red'   
    return colors
vis_x = tsne_res[:, 0]
vis_y = tsne_res[:, 1]
plt.figure(figsize=(20,20))
plt.scatter(vis_x, vis_y, c=get_colors(train));
# Red = target, yellow = leak, green = ints, blue = floats