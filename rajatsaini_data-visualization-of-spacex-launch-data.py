import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../input/spacex_launch_data.csv")
df.head()
df.describe()
df.info()
sns.set(rc={'figure.figsize':(15,8)})
plt.xticks(rotation=90)
sns.countplot(x="Landing Outcome", data=df)

g=sns.countplot(x="Launch Site", data=df,hue="Landing Outcome");
g=sns.countplot(x="Orbit", data=df,hue="Landing Outcome");
g=sns.countplot(x="Orbit", data=df,hue="Mission Outcome");
