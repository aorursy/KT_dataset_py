import numpy as np

import pandas as pd
data = pd.read_hdf("../input/train.h5")
len(pd.unique(data.id))
# First ID

data[data.id == data.id[0]].plot(x = 'timestamp', y = 'y',kind='scatter');
import matplotlib.pyplot as plt



data_binary = data[data.id == data.id[0]].isnull()



fig = plt.figure()

ax = fig.add_subplot(111)

ax.imshow(data_binary, aspect='auto', cmap=plt.cm.gray, interpolation='nearest')

plt.title("NaN structure for first id. White: NaN Black: No NaN")

plt.xlabel("Features")

plt.ylabel("Timestamp");
np.random.seed(352)

ids = np.random.choice(data.id,20)



for i in ids:

    

    data_binary = data[data.id == i].isnull()



    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.imshow(data_binary, aspect='auto', cmap=plt.cm.gray, interpolation='nearest')

    plt.title("NaN structure for id %i" % i)

    plt.xlabel("Features")

    plt.ylabel("Timestamp");
unique_ids = pd.unique(data.id)



NaN_vectors = np.zeros(shape=(1424, data.shape[1]))



for i, i_id in enumerate(unique_ids):

    

    data_sub = data[data.id == i_id]

    NaN_vectors[i, :] = np.sum(data_sub.isnull(),axis=0) / float(data_sub.shape[0])
NaN_vectors[0, :]
import seaborn as sns

g = sns.clustermap(NaN_vectors,col_cluster=False,method='average',metric='euclidean')
unique_id = pd.unique(data.id)



result = pd.DataFrame(np.zeros(shape=(len(unique_id),5)))



for i, i_id in enumerate(unique_id):



    data_sub = data[data.id == i_id]



    count = data_sub.timestamp.count()



    time_min = np.min(data_sub.timestamp)

    time_max = np.max(data_sub.timestamp)



    ratio = count / float(time_max - time_min + 1)

    

    result.loc[i, :] = [i_id, ratio, time_min, time_max, count]

    result.columns = ["id", "ratio", "time_min", "time_max", "count"]
print(pd.unique(np.round(result.ratio,2)))
print(result[result.ratio < 0.99])
ids = result[result.ratio < 0.99].id

ratios = result[result.ratio < 0.99].ratio



for i, r in zip(ids,ratios):



    data[data.id == i].plot(x = "timestamp", y = "y", kind = "scatter", 

                            title = "ratio:" + str("{0:.2f}".format(r)) + " id: " + str(int(i)));