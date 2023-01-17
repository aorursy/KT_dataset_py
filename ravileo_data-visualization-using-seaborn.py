import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
# Visualizing statistical relationships
df = pd.read_csv("../input/pr-train/Pr_Train.csv")
df.head()
sns.relplot(x="Views",y="Upvotes", data=df)
# Tag Associate with the data
sns.relplot(x="Views",y= "Upvotes", hue="Answers",data=df);
sns.relplot(x="Views",y="Upvotes",size="Tag",data=df);
df2 = pd.read_csv("../input/hr-train/Hr_Train.csv")
df2
sns.catplot(x="education", y="avg_training_score",data=df2);
sns.catplot(x="education",y="avg_training_score",jitter=False,data=df2);
sns.catplot(x="education",y="avg_training_score",hue="gender",data=df2);
sns.catplot(x="education",y="avg_training_score",kind="box",data=df2)
sns.catplot(x="education",y="avg_training_score",hue="is_promoted",kind="box",data=df2);
sns.catplot(x="education",y="avg_training_score",hue="is_promoted",kind="violin",data=df2);
sns.catplot(x="education",y="avg_training_score",hue="is_promoted",split= True,kind ="violin",data=df2);
df2.dtypes
sns.catplot(x="education",y="avg_training_score",hue = "is_promoted",kind= "bar",data=df2)
sns.catplot(x="education",y="avg_training_score",hue="is_promoted",kind="point",data=df2)
# Histogram
sns.distplot(df2.age,rug=True)
df2.dtypes
sns.jointplot(x="avg_training_score", y="age", data=df2);
sns.jointplot(x="age",y="avg_training_score",kind="hex",data=df2);
sns.jointplot(x="age",y="avg_training_score",kind="kde",data=df2);
corrmat = df2.corr()
f,ax = plt.subplots(figsize=(9,6))
sns.heatmap(corrmat,vmax=8,square=True)
sns.catplot(x="age",y="avg_training_score",data=df2,kind="boxen",height=4,aspect=2.7,hue="is_promoted")

sns.pairplot(df2);
from IPython.display import Image
Image(url="https://media.giphy.com/media/13hxeOYjoTWtK8/giphy.gif")
