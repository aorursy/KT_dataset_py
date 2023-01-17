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
file = open("/kaggle/input/csvsqsbetter/CSvsQS.txt", "r")

cData = dict()

cData["Number of Integers in Array"] = []

cData["Range of Integers in Array"] = []

cData["Time to Sort"] = []

cData["colin"] = []

for i in file:

    l = i.split(":")

    if l[0] == "C":

        cData["colin"] += [10000]

    else:

        cData["colin"] += [0]

    cData["Number of Integers in Array"] += [int(l[1])]

    cData["Range of Integers in Array"] += [int(l[2])]

    cData["Time to Sort"] += [int(l[3].replace("\n", ""))]

df = pd.DataFrame(cData)

print(df)
import plotly.express as px



fig = px.scatter_3d(df, x='Number of Integers in Array', y='Range of Integers in Array', z='Time to Sort', color="colin")

fig.show()