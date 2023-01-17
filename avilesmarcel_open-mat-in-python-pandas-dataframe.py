from scipy.io import loadmat

import pandas as pd
mat = loadmat('../input/train_1/1_1_0.mat')
mdata = mat['dataStruct']
mtype = mdata.dtype
ndata = {n: mdata[n][0,0] for n in mtype.names}
ndata
data_headline = ndata['channelIndices']

print(data_headline)
data_headline = data_headline[0]
data_raw = ndata['data']

len(data_raw)
pdata = pd.DataFrame(data_raw,columns=data_headline)
pdata