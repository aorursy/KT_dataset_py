import seaborn as sns 

import matplotlib.pyplot as plt

sns.set(style="ticks", color_codes=True)

import pandas as pd 
titanic=pd.read_csv("../input/train.csv")

titanic.head()
sns.catplot(x="Sex", y="Survived", kind="bar",hue='Embarked', data=titanic);
sns.catplot(x="Sex",hue='Pclass' ,kind="count", data=titanic);

sns.catplot(x="Sex", y="Survived", kind="point",hue='Embarked',jitter=True, data=titanic);
#changing the default line colurs

sns.catplot(x="Pclass", y="Survived", kind="point",hue='Sex', data=titanic);
sns.catplot(x="Pclass", y="Survived", kind="point",hue='Sex',   palette={"male": "g", "female": "m"}, data=titanic);
#chaging the arkers

sns.catplot(x="Pclass", y="Survived", kind="point",hue='Sex',palette={"male": "g", "female": "m"}, markers=["o", "x"], data=titanic);
titanic.head()
sns.catplot(x='Sex',y='Age',col='Survived',row='Pclass',kind='bar',data=titanic)