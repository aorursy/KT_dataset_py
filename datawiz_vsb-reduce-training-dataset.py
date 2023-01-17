# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pyarrow.parquet as pq

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
op_dict = {}

n_agg = 2000





loop = int(round(8712/1000))

for counter in range(loop):

    st = counter * 1000

    if(((counter+1) * 1000) > 8712):

        en = 8712

    else:

        en = (counter+1)*1000

    start = 0

    end = n_agg

    train = pq.read_pandas('../input/train.parquet', columns=[str(i) for i in range(st,en)]).to_pandas()

    agg_measure_list = []

    print("column start = "+ str(st)+" column end = "+str(en))

    for loc2 in range(400):

        mn = list(train.loc[start:end].mean(axis = 0,skipna = True))

        agg_measure_list.append(mn)

        start += n_agg

        end += n_agg

    print("List length = "+ str(len(agg_measure_list)))

    op_dict[str(counter)] = agg_measure_list

    print("dict length = "+ str(len(op_dict)))
df0 = pd.DataFrame(op_dict["0"])

df0.columns = [list(range(0,1000))]

df1 = pd.DataFrame(op_dict["1"])

df1.columns = [list(range(1000,2000))]

df2 = pd.DataFrame(op_dict["2"])

df2.columns = [list(range(2000,3000))]

df3 = pd.DataFrame(op_dict["3"])

df3.columns = [list(range(3000,4000))]

df4 = pd.DataFrame(op_dict["4"])

df4.columns = [list(range(4000,5000))]

df5 = pd.DataFrame(op_dict["5"])

df5.columns = [list(range(5000,6000))]

df6 = pd.DataFrame(op_dict["6"])

df6.columns = [list(range(6000,7000))]

df7 = pd.DataFrame(op_dict["7"])

df7.columns = [list(range(7000,8000))]

df8 = pd.DataFrame(op_dict["8"])

df8.columns = [list(range(8000,8712))]
df = pd.concat([df0, df1, df2, df3, df4, df5, df6, df7, df8], axis = 1)
plt.plot(df)
df.to_csv('train_compressed.csv', index=False)
test = pq.read_pandas('../input/test.parquet', columns=[str(i) for i in range(8712,8715)]).to_pandas()

test_meta = pd.read_csv("../input/metadata_test.csv")
start_loc = 8712

end_loc = 8712 + 20337
op_dict = {}

n_agg = 2000

import math



loop = int(math.ceil(20337/2000))

for counter in range(loop):

    st = start_loc + (counter * 2000)

    if(((counter+1) * 2000) > end_loc):

        en = start_loc + end_loc

    else:

        en = start_loc + ((counter+1)*2000)

    start = 0

    end = n_agg

    train = pq.read_pandas('../input/test.parquet', columns=[str(i) for i in range(st,en)]).to_pandas()

    agg_measure_list = []

    print("column start = "+ str(st)+" column end = "+str(en))

    for loc2 in range(400):

        mn = list(train.loc[start:end].mean(axis = 0,skipna = True))

        agg_measure_list.append(mn)

        start += n_agg

        end += n_agg

    print("List length = "+ str(len(agg_measure_list)))

    op_dict[str(counter)] = agg_measure_list

    print("dict length = "+ str(len(op_dict)))
df0 = pd.DataFrame(op_dict["0"])

df0.columns = [list(range(8712,10712))]

df1 = pd.DataFrame(op_dict["1"])

df1.columns = [list(range(10712,12712))]

df2 = pd.DataFrame(op_dict["2"])

df2.columns = [list(range(12712,14712))]

df3 = pd.DataFrame(op_dict["3"])

df3.columns = [list(range(14712,16712))]

df4 = pd.DataFrame(op_dict["4"])

df4.columns = [list(range(16712,18712))]

df5 = pd.DataFrame(op_dict["5"])

df5.columns = [list(range(18712,20712))]

df6 = pd.DataFrame(op_dict["6"])

df6.columns = [list(range(20712,22712))]

df7 = pd.DataFrame(op_dict["7"])

df7.columns = [list(range(22712,24712))]

df8 = pd.DataFrame(op_dict["8"])

df8.columns = [list(range(24712,26712))]

df9 = pd.DataFrame(op_dict["9"])

df9.columns = [list(range(26712,28712))]

df10 = pd.DataFrame(op_dict["10"])

df10.columns = [list(range(28712,29049))]
df = pd.concat([df0, df1, df2, df3, df4, df5, df6, df7, df8,df9,df10], axis = 1)
test_meta = pd.read_csv("../input/metadata_test.csv")

max(test_meta['signal_id'])
df.head()
df.to_csv('test_compressed.csv', index=False)