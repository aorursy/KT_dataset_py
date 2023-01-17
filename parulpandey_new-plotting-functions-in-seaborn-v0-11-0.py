# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install seaborn==0.11.0
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
print(sns.__version__)
df = pd.read_csv('../input/palmer-archipelago-antarctica-penguin-data/penguins_size.csv')

df.head()
df.info()
g = sns.displot(

    data=df, kind="kde", rug=True,

    x="culmen_length_mm", y="culmen_depth_mm",

    col="island", hue="species",

)
g = sns.displot(

    data=df, kind="ecdf",

    x="body_mass_g", col="sex", hue="species",

)
sns.histplot(data=df, x="flipper_length_mm")
sns.histplot(data=df, y="flipper_length_mm")
sns.histplot(

    df, x="culmen_length_mm", hue="island", element="step",

    stat="density", common_norm=False,

)
sns.histplot(df, x="flipper_length_mm", hue="species", element="poly")
sns.histplot(

    df, x="culmen_depth_mm", y="species", hue="species", legend=False

)
sns.ecdfplot(data=df, x="flipper_length_mm")
sns.ecdfplot(data=df, x="culmen_length_mm", hue="species", stat="count")