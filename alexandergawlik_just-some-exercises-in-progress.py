import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("../input/train.csv")
df = df.drop('PassengerId', axis=1)
df.head()
df.info()
print(df.describe())
ax = df["Survived"].value_counts().plot.bar()
ax.set_title("Survivability")
ax.set(ylabel='', xlabel='Not survive/Survive')
ax =df["Sex"].value_counts().plot.bar()
ax.set_title("Sex survivability")
class_sur=df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
class_sur
ax=sns.barplot(y="Survived", x="Pclass",data=class_sur)
ax.set(ylabel='Survived percent', xlabel='Pclass')
ax.set_title("Pclass survivability")
sex_sur=df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sex_sur
ax=sns.barplot(y='Survived', x='Sex', data=sex_sur)
ax.set(ylabel='Survived percent', xlabel='Sex')
ax.set_title("Sex survivability")
g = sns.FacetGrid(df, col='Survived')
g.map(plt.hist, 'Age', bins=20)