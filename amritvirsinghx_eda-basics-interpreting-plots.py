import pandas as pd
import seaborn as sns
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
sns.catplot(x="Survived", y="Fare", data=train,kind="bar",height=3)
sns.catplot(x="Survived", y="Age", data=train,kind="bar",height=3,hue="Pclass")
sns.catplot("Survived", data=train,kind="count",height=3)
sns.catplot(x="Survived", y="Age", data=train,kind="bar",height=3,hue="Pclass")
sns.catplot(x="Age", y="Fare", data=train,kind="swarm",height=4)
sns.catplot(x="Sex", y="Age", data=train,kind="box",height=4)
