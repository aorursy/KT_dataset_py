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

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

Clean_Dogs_Milan = pd.read_csv("../input/Clean-Dogs-Milan-Singh.csv")

print("Setup Complete")
plt.title("Latitude and Longitude of Dogs in Cambridge")

sns.scatterplot(x=Clean_Dogs_Milan["Latitude_masked"], y=Clean_Dogs_Milan["Longitude_masked"])
sns.jointplot(data=Clean_Dogs_Milan, x="Latitude_masked", y="Longitude_masked", kind="kde")
sns.catplot(data= Clean_Dogs_Milan, x="Neighborhood", kind="count", height=20)
plt.pie(data=Clean_Dogs_Milan, x="Longitude_masked")
