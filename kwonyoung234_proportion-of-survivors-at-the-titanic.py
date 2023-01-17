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

titanic = pd.read_csv("../input/train.csv")
#titanic.head()
tit_sur= titanic.groupby("Survived",as_index=0)["Name"].count()
ax= tit_sur.plot(kind="bar", x="Survived", y="Name",title="people who survivied or not")
ax.set_ylabel("# of people")
ax= tit_sur.plot(kind="bar", x="Survived", y="Name",title="people who survivied or not",ax=ax)
sur_proportion = tit_sur[tit_sur.Survived == 1]["Name"]*1.0 / tit_sur["Name"].sum()
print(sur_proportion.values)

