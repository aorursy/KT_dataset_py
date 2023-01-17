# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pickle
data=open('/kaggle/input/radioml2016-deepsigcom/RML2016.10a_dict.pkl','rb')
pickle.load(data)
import pickle

import gzip

import numpy as np



with open('/kaggle/input/radioml2016-deepsigcom/RML2016.10a_dict.pkl', 'rb') as f:

    u = pickle._Unpickler(f)

    u.encoding = 'latin1'

    p = u.load()
import pandas as pd
l=p.values()
l=list(l)
l[1][1]
output=open("write.txt","w")
output.write(str(p))
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], p.keys())))), [1,0])
X = []  

lbl = []

for mod in mods:

    for snr in snrs:

        X.append(p[(mod,snr)])

        for i in range(p[(mod,snr)].shape[0]):  lbl.append((mod,snr))

X = np.vstack(X)
label=[]

mod=[]

data=[]

for i in range(len(lbl)):

        label.append(lbl[i][0])

        mod.append(lbl[i][1])

        data.append(X[i])
data=pd.DataFrame([label,mod,data])
data.head()
n_examples
