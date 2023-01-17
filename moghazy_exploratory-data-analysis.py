import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_json("../input/train.json")
test = pd.read_json("../input/test.json")
train.head()
print("The number of traning examples(data points) = %i " % train.shape[0])
print("The number of features we have = %i " % train.shape[1])
train.describe()
train.isnull().sum()
import seaborn as sns


import matplotlib.pyplot as plt


corr = train.corr()
f, ax = plt.subplots(figsize=(25, 25))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5)
corr
((train.is_turkey).value_counts()).plot.bar( ec="orange")
unique, count= np.unique(train.is_turkey, return_counts=True)
print("The number of occurances of each class in the dataset = %s " % dict (zip(unique, count) ), "\n" )
duration = train.end_time_seconds_youtube_clip - train.start_time_seconds_youtube_clip
train['duration'] = duration
duration.hist()
from scipy import stats
import numpy as np

z = np.abs(stats.zscore(train.drop(['audio_embedding', 'vid_id', 'is_turkey'], axis = 1)))

zee = (np.where(z > 2.5))[1]

print("number of data examples greater than 2.5 standard deviations = %i " % len(zee))
f = plt.figure(figsize=(25,10))
sns.boxplot(x=train['duration'])
print("Shape of first audio embedding. 1st dimention = %i, 2nd dimention = %i " % 
      (np.shape((train.audio_embedding[0]))[0] , np.shape((train.audio_embedding[0]))[1]) ) 