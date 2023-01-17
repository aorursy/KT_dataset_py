import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df=pd.read_csv("../input/udemy-courses/udemy_courses.csv")
df.head()
df.info()
df.describe()
sns.set_style("whitegrid")
g=sns.FacetGrid(df,col="subject")

g.map(plt.hist,"num_subscribers",bins=10)
fig,ax=plt.subplots(figsize=(10,6))

ax=sns.heatmap(df.corr(),annot=True,cmap="viridis")
plt.figure(figsize=(10,6))

sns.countplot(x="subject",data=df)
df["course_title"].value_counts().head(10)
df["price"].value_counts().head(15).sort_values(ascending=False)
sns.countplot(x="subject",hue="level",data=df)

plt.legend(loc='best', bbox_to_anchor=(0.5, 0.5))
levels=df["level"].value_counts().drop("52")

levels
ax = df['level'].value_counts().drop("52").plot(kind ='bar', figsize = (6,4), width = 0.8)

ax.set_title('Levels vs Amount of courses', fontsize = 15)

ax.set_ylabel('Amount of courses', fontsize = 15)

ax.set_xlabel('Levels', fontsize = 15)
sns.jointplot(x="num_reviews",y="num_subscribers",data=df,kind="scatter",color="red",space=0.4,height=8,ratio=4)
import warnings

warnings.filterwarnings("ignore")
df[(df["subject"]=="Web Development") & (df["price"]>"100")].count()
df[(df["subject"]=="Graphic Design") & (df["price"]<"80")].count()
df[(df["subject"]=="Business Finance") & (df["price"]>"100")].count()
df[(df["subject"]=="Musical Instruments") & (df["price"]<"80")].count()
df[["course_title","num_subscribers"]].max().head()
df[["course_title","num_subscribers"]].min().head()