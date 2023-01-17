import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns





dataframe = pd.read_csv("../input/cereal.csv")
dataframe.describe()
dataframe.head()
dataframe.tail()
category = dataframe['mfr']
sns.countplot(category).set_title("MFR")
sns.swarmplot(category).set_title('MFR')