import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
df= pd.read_csv('../input/titanic/train.csv')

df.info()
df.head()
df["Survived"]=df["Survived"].map({1:"Survived", 0: "Non-Survived"})
df["Pclass"]=df["Pclass"].map({1:"First", 2: "Second", 3:"Third"})
df.info()
df.head()
plt.figure(figsize=(8,6))

sns.set(style="whitegrid", font_scale=1.3)

ax=sns.countplot(x="Survived", data=df, palette="hls")

ax.set_title('foo')
sns.set(style="white", font_scale=1.3)

sns.countplot(x="Pclass", data=df, palette="hls")
sns.set(style="white", font_scale=1.3)

sns.countplot(x="Survived",hue="Pclass", data=df, palette="hls")
sns.set(style="white", font_scale=1.3)

sns.countplot(x="Survived",hue="Sex", data=df, palette="hls")
df[(df["Sex"] =='male')].hist('Age') 