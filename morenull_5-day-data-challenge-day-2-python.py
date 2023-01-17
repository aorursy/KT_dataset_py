import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

sns.set(color_codes=True)

df = pd.read_csv("../input/1000-cameras-dataset/camera_dataset.csv")

df.describe
print(df.columns)
sns.distplot(df["Normal focus range"],kde= False).set_title('Normal focus range plot')