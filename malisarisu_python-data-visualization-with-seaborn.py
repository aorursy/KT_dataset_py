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
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
iris=pd.read_csv("/kaggle/input/iris/Iris.csv")
iris.head()
iris.tail(5)
iris.info()
iris.describe().T
iris.describe().plot(kind = "area",fontsize=20, figsize = (15,8), table = True, colormap="rainbow")

plt.xlabel('Statistics',)

plt.ylabel('Value')

plt.title("General Statistics of Iris Dataset")

plt.show()
iris.corr()
sns.countplot(x="Species", data=iris,saturation=0.9)

plt.title(" Frequency of the Observation")

plt.show()
iris['Species'].value_counts().plot.pie(explode=[0.01,0.01,0.01],

                                        figsize=(10,8),

                                        autopct='%1.1f%%',

                                        pctdistance=0.5,

                                        labeldistance=1.3)

plt.show()
sns.scatterplot(x="PetalLengthCm",y="PetalWidthCm", data=iris)

plt.show()
sns.FacetGrid(data=iris, hue="Species",

              size=5).map(plt.scatter,"SepalLengthCm", 

              "SepalWidthCm").add_legend()

plt.show()
sns.lmplot(x="SepalLengthCm", y="SepalWidthCm",hue="Species", data=iris)

plt.show()