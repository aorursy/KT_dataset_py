# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np 



reg = 10 # trying anokas idea of regularization

eval = True



train = pd.read_csv("../input/clicks_train.csv")



if eval:

	ids = train.display_id.unique()

	ids = np.random.choice(ids, size=len(ids)//10, replace=False)



	valid = train[train.display_id.isin(ids)]

	train = train[~train.display_id.isin(ids)]

	

	print (valid.shape, train.shape)



cnt = train[train.clicked==1].ad_id.value_counts()

cntall = train.ad_id.value_counts()

del train
cnt.head()
cntall.head()
rinds = cnt.index[1:1000]
import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize=(12, 6))

plt.scatter(cnt[rinds], [cnt[i] / (cntall[i] )for i in rinds])
rinds2 = cnt.index[1000:2000]

plt.figure(figsize=(8, 6))

plt.scatter(cnt[rinds2], [cnt[i] / (cntall[i] + 100 )for i in rinds2])