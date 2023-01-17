import pandas as pd
import seaborn as sns
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
sns.catplot(x="SibSp",y="Survived",data=train,kind="bar",height=4)
sns.catplot(x="SibSp",y="Survived",data=train,kind="point",height=4)
sns.catplot(x="Survived",y="Fare",data=train,kind="swarm",height=4)
g =sns.catplot(x="Survived",y="Fare",data=train,kind="swarm",height=4)
g.set_axis_labels("Survived (Categorical)","Fare (Numerical)")