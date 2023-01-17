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

import warnings # current version of seaborn generates a bunch of warnings that we'll ignore

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)

dataset = pd.read_csv("../input/glass.csv")



#print (dataset)



#print (dataset["RI"])

dataset.describe()

#dataset.plot(kind="scatter",x="RI",y="K")

#dataset.plot(kind="scatter",x="RI",y="Na")

#sns.jointplot(x="RI",y="K",data=dataset,size=5)
import numpy as np

import matplotlib.mlab as mlab

import matplotlib.pyplot as plt


