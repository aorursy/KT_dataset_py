# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd



df = pd.read_csv('../input/mammogram_dataset.csv',delimiter=";")

df.head(5)
import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline

x = df["severity"]

plt.hist(x)
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier



x = df[['BI_RADS_assessment','age','shape','margin','density']] 

y = df['severity'] 



rfc = RandomForestClassifier()



rfc2 = cross_val_score(rfc, x, y, scoring='accuracy', cv=2)

rfc3 = cross_val_score(rfc, x, y, scoring='accuracy', cv=3)

rfc4 = cross_val_score(rfc, x, y, scoring='accuracy', cv=4)

rfc5 = cross_val_score(rfc, x, y, scoring='accuracy', cv=5)

rfc6 = cross_val_score(rfc, x, y, scoring='accuracy', cv=6)

rfc7 = cross_val_score(rfc, x, y, scoring='accuracy', cv=7)

rfc8 = cross_val_score(rfc, x, y, scoring='accuracy', cv=8)

rfc9 = cross_val_score(rfc, x, y, scoring='accuracy', cv=9)

rfc10 = cross_val_score(rfc, x, y, scoring='accuracy', cv=10)

rfc11 = cross_val_score(rfc, x, y, scoring='accuracy', cv=11)

rfc12 = cross_val_score(rfc, x, y, scoring='accuracy', cv=12)

rfc13 = cross_val_score(rfc, x, y, scoring='accuracy', cv=13)

rfc14 = cross_val_score(rfc, x, y, scoring='accuracy', cv=14)

rfc15 = cross_val_score(rfc, x, y, scoring='accuracy', cv=15)

rfc16 = cross_val_score(rfc, x, y, scoring='accuracy', cv=16)

rfc17 = cross_val_score(rfc, x, y, scoring='accuracy', cv=17)

rfc18 = cross_val_score(rfc, x, y, scoring='accuracy', cv=18)

rfc19 = cross_val_score(rfc, x, y, scoring='accuracy', cv=19)

rfc20 = cross_val_score(rfc, x, y, scoring='accuracy', cv=20)

rfc21 = cross_val_score(rfc, x, y, scoring='accuracy', cv=21)

rfc22 = cross_val_score(rfc, x, y, scoring='accuracy', cv=22)

rfc23 = cross_val_score(rfc, x, y, scoring='accuracy', cv=23)

rfc24 = cross_val_score(rfc, x, y, scoring='accuracy', cv=24)

rfc25 = cross_val_score(rfc, x, y, scoring='accuracy', cv=25)
print("rfc2",rfc2.mean())

print("rfc3",rfc3.mean())

print("rfc4",rfc4.mean())

print("rfc5",rfc5.mean())

print("rfc6",rfc6.mean())

print("rfc7",rfc7.mean())

print("rfc8",rfc8.mean())

print("rfc9",rfc9.mean())

print("rfc10",rfc10.mean())

print("rfc11",rfc11.mean())

print("rfc12",rfc12.mean())

print("rfc13",rfc13.mean())

print("rfc14",rfc14.mean())

print("rfc15",rfc15.mean())

print("rfc16",rfc16.mean())

print("rfc17",rfc17.mean())

print("rfc18",rfc18.mean())

print("rfc19",rfc19.mean())

print("rfc20",rfc20.mean())

print("rfc21",rfc21.mean())

print("rfc22",rfc22.mean())

print("rfc23",rfc23.mean())

print("rfc24",rfc24.mean())

print("rfc25",rfc25.mean())