import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/kc_house_data.csv")
import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns

%matplotlib inline



g = sns.lmplot(x="sqft_living", y="price", hue="floors", data=df, size=7 )

g.set_axis_labels("Living Room Square Feet", "Price")
sns.set(style="white")

h = sns.PairGrid(df[["bedrooms", "price", "grade"]], diag_sharey=False)

h.map_lower(sns.kdeplot, cmap="Blues_d")

h.map_upper(plt.scatter)

h.map_diag(sns.kdeplot, lw=3)
df.price.plot(kind="hist", bins=100)

sns.despine()
df.condition.plot(kind="hist", bins=5)
df.yr_built.plot(kind="hist")
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