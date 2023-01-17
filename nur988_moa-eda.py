import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import plotly.express as px


train=pd.read_csv("../input/lish-moa/train_features.csv")
targets=pd.read_csv("../input/lish-moa/train_targets_scored.csv")
test=pd.read_csv("../input/lish-moa/test_features.csv")
train_unscored=pd.read_csv("../input/lish-moa/train_targets_nonscored.csv")
fig,axes=plt.subplots(1,3,figsize=(18,10))

ax=sns.countplot(ax=axes[0],x='cp_type',data=train)
ax=sns.countplot(ax=axes[1],x='cp_time',data=train)
ax=sns.countplot(ax=axes[2],x='cp_dose',data=train)
corr=train.iloc[:,4:15].corr()
fig=plt.figure(figsize=(18,10))
sns.heatmap(corr,annot=True,cmap='coolwarm')
train.groupby(['cp_time'])
import plotly.express as px

fig=px.scatter(train,x='g-4',y='g-2')
fig.show()
fig=px.scatter_matrix(train,dimensions=train.columns[5:15],color='cp_time')
fig.show()

