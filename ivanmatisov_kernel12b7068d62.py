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
data = pd.read_csv("/kaggle/input/titanic/train.csv")
data.head()
names = data[data["Sex"]=="female"]["Name"]
def find_name(name):

    if "Miss." in name:

        lst = name.split(" ")

        idx = lst.index("Miss.")

        return lst[idx+1]

    if "(" in name:

        idx = name.find("(")

        return name[idx+1:-1].split(" ")[0]
names.apply(find_name).value_counts()