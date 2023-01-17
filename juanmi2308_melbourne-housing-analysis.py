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
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



df = pd.read_csv("../input/melbourne-housing-snapshot/melb_data.csv")

df.describe()
fs_value = df["Price"].max() / 2



low_priced = df[df["Price"]<fs_value]

high_priced = df[df["Price"]>fs_value]

low_priced.describe()

high_priced.describe()



range1 = high_priced["Price"].min() - low_priced["Price"].max()

print(range1)



    
ss_value = df["Price"].max() / 2 + df["Price"].max() / 4



low_priced2 = df[df["Price"]<ss_value]

high_priced2 = df[df["Price"]>ss_value]

lp2 = low_priced2["Address"].count()

hp2 = high_priced2["Address"].count()



range2 = high_priced2["Price"].min() - low_priced["Price"].max()



high_priced2
nm = df[df["Regionname"]=="Northern Metropolitan"]

wm = df[df["Regionname"]=="Western Metropolitan"]

sm = df[df["Regionname"]=="Southern Metropolitan"]

em = df[df["Regionname"]=="Eastern Metropolitan"]

sem = df[df["Regionname"]=="South-Eastern Metropolitan"]

ev = df[df["Regionname"]=="Eastern Victoria"]

nv = df[df["Regionname"]=="Northern Victoria"]

wv = df[df["Regionname"]=="Western Victoria"]



nmm = nm["Price"].max()

wmm = wm["Price"].max()

smm = sm["Price"].max()

emm = em["Price"].max()

semm = sem["Price"].max()

evm = ev["Price"].max()

nvm = nv["Price"].max()

wvm = wv["Price"].max()



x1 = ["NM","WM","SM","EM","SEM","EV","NV","WV"]

y1 = [nmm, wmm, smm, emm, semm, evm, nvm, wvm]



#plt.bar(x1, y1)

sns.barplot(x=x1, y=y1)
nmm2 = nm["Price"].mean()

wmm2 = wm["Price"].mean()

smm2 = sm["Price"].mean()

emm2 = em["Price"].mean()

semm2 = sem["Price"].mean()

evm2 = ev["Price"].mean()

nvm2 = nv["Price"].mean()

wvm2 = wv["Price"].mean()



y2 = [nmm2, wmm2, smm2, emm2, semm2, evm2, nvm2, wvm2]



#plt.bar(x1, y1)

sns.barplot(x=x1, y=y2)
total = df["Address"].count()



r1 = df[df["Rooms"]==1]

r2 = df[df["Rooms"]==2]

r3 = df[df["Rooms"]==3]

r4 = df[df["Rooms"]==4]

r5 = df[df["Rooms"]==5]

r6 = df[df["Rooms"]==6]

r7 = df[df["Rooms"]==7]

r8 = df[df["Rooms"]==8]

r9 = df[df["Rooms"]==9]

r10 = df[df["Rooms"]==10]



r1c = r1["Address"].count()

r2c = r2["Address"].count()

r3c = r3["Address"].count()

r4c = r4["Address"].count()

r5c = r5["Address"].count()

r6c = r6["Address"].count()

r7c = r7["Address"].count()

r8c = r8["Address"].count()

r9c = r9["Address"].count()

r10c = r10["Address"].count()



x3 = ["R1","R2","R3","R4","R5","R6","R7","R8","R9","R10"]

y3 = [r1c,r2c,r3c,r4c,r5c,r6c,r7c,r8c,r9c,r10c]



sns.barplot(x=x3, y=y3)
nmm3 = nm["YearBuilt"].mode()

wmm3 = wm["YearBuilt"].mode()

smm3 = sm["YearBuilt"].mode()

emm3 = em["YearBuilt"].mode()

semm3 = sem["YearBuilt"].mode()

evm3 = ev["YearBuilt"].mode()

nvm3 = nv["YearBuilt"].mode()

wvm3 = wv["YearBuilt"].mode()



y4 = [nmm3, wmm3, smm3, emm3, semm3, evm3, nvm3, wvm3]

sns.barplot(x=x1, y=y4)
print(y4)