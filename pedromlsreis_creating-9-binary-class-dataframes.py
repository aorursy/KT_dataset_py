import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Loading the train datasets:

trainvar = pd.read_csv("../input/training_variants")

traintxt = pd.read_csv("../input/training_text", sep="\|\|", engine="python", skiprows=1, names=["ID","Text"])



train = pd.merge(trainvar, traintxt, how='left', on='ID').fillna('')
train.head()
for i in range(1,10):

    globals()[str("train")+str(i)] = train.copy()

    globals()[str("train")+str(i)].loc[globals()[str("train")+str(i)]["Class"]!=i,"Class"] = 0
train1.head()
train2.head()