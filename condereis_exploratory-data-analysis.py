import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%pylab inline
train_data = pd.read_csv('../input/train.csv',dtype={'place_id':'int64'})
test_data = pd.read_csv('../input/test.csv',dtype={'place_id':'int64'})
full_data = pd.concat([train_data, test_data]).reset_index(drop=True)
print ('Train Set Shape:',train_data.shape)
print ('Test Set Shape:',test_data.shape)
full_data.tail()
full_data.describe()
sns.distplot(full_data.time.sample(frac=0.01))
ts_count = full_data.time.value_counts().sort_index()
ts_fft = np.fft.fft(ts_count.tolist())

p = plt.plot(ts_fft)
plt.xlim([0,2000])
#plt.ylim([0,1e6])
