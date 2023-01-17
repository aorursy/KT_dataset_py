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
# modules we'll use

import pandas as pd

import numpy as np



# helpful character encoding module

import chardet



# set seed for reproducibility

np.random.seed(0)

before = "This is the euro symbol: €"

type(before)
after=before.encode("utf-8",errors='replace')

after
type(after)
print(after.decode("utf-8"))
print(after.decode("ascii"))
before = "This is the euro symbol: €"

type(before)
after=before.encode("ascii",errors="replace")

print(after)
kickstarter_2016 = pd.read_csv("../input/kickstarter-projects/ks-projects-201612.csv")
f=open("../input/kickstarter-projects/ks-projects-201612.csv","rb")

res=chardet.detect(f.read(10000))

res
kick=pd.read_csv("../input/kickstarter-projects/ks-projects-201612.csv", encoding='Windows-1252')

kick.head()
kick.to_csv("ks-projects-201612-utf-8.csv")
police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv")
f1=open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv","rb")

pol=chardet.detect(f1.read(5000))

pol
police=pd.read_csv("../input/kickstarter-projects/ks-projects-201612.csv", encoding='ascii')

police.head()
f1=open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv","rb")

pol=chardet.detect(f1.read())

pol
police=pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='Windows-1252')

police.head()