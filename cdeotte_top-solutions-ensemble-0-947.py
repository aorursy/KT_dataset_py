import pandas as pd, numpy as np

import matplotlib.pyplot as plt



p1 = pd.read_csv('../input/top-ion-subs/submission1.csv').open_channels.values

p2 = pd.read_csv('../input/top-ion-subs/submission2.csv').open_channels.values

p3 = pd.read_csv('../input/top-ion-subs/submission3.csv').open_channels.values
sub = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv')

sub.open_channels = np.median( np.stack([p1,p2,p3]).T, axis=1 ).astype('int8')

sub.to_csv('submission.csv', index=False, float_format='%.4f')
sub.open_channels.value_counts()
res=40

plt.figure(figsize=(20,5))

plt.plot(sub.time[::res],sub.open_channels[::res])

plt.show()