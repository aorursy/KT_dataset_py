import numpy as np

import pandas as pd



dataset = pd.read_csv("/kaggle/input/league-of-legends-ranked-games/challenger.csv")

dataset.drop(['vilemaw_kills_team_1', 'vilemaw_kills_team_2'], axis=1, inplace=True)

dataset.head(10)
import seaborn as sns; sns.set()

import matplotlib.pyplot as plt
g = sns.distplot(dataset["win"])
g = sns.lmplot(x="tower_kills_team_2", y="tower_kills_team_1", col="first_tower", height=4, data=dataset, hue="win")
g = sns.lmplot(x="inhibitor_kills_team_2", y="inhibitor_kills_team_1", col="first_inhibitor",data=dataset, height=4, hue="win")
g = sns.lmplot(x="dragon_kills_team_2", y="dragon_kills_team_1", col="first_dragon", data=dataset, height=4, hue="win")
g = sns.lmplot(x="baron_kills_team_1", y="baron_kills_team_2", col="first_baron", height=4, data=dataset, hue="win",  markers=["o", "x"])
g = sns.lmplot(x="rift_herald_kills_team_2", y="rift_herald_kills_team_1", col="first_rift_herald", data=dataset, height=4, hue="win", markers=["o", "x"])
plt.figure(figsize=(15,8))

plt.title("Relationship between objectives achievement factors and the team that won")

sns.heatmap(

    dataset.iloc[:,0:17].corr(),

    annot=False,

    linewidths=.5,

)
playersP = dataset.iloc[:,17:52]

playersP['win'] = dataset['win']



plt.figure(figsize=(15,8))

plt.title("Correlation between the performance of team 1 players")

sns.heatmap(

    playersP.corr(),

    annot=False,

    linewidths=.5,

)
playersP = dataset.iloc[:,52:88]

playersP['win'] = dataset['win']



plt.figure(figsize=(15,8))

plt.title("Correlation between the performance of team 2 players")

sns.heatmap(

    playersP.corr(),

    annot=False,

    linewidths=.5,

)
top=["gold_earned_20m_top_team_1", "cs_20m_top_team_1", "xp_20m_top_team_1", "damege_taken_20m_top_team_1"]

top2=["gold_earned_20m_top_team_2", "cs_20m_top_team_2", "xp_20m_top_team_2", "damege_taken_20m_top_team_2"]

middle=["gold_earned_20m_middle_team_1", "cs_20m_middle_team_1", "xp_20m_middle_team_1", "damege_taken_20m_middle_team_1"]

jungle=["gold_earned_20m_jungle_team_1", "cs_20m_jungle_team_1", "xp_20m_jungle_team_1", "damege_taken_20m_jungle_team_1"]

bottom_duo_carry=["gold_earned_20m_bottom_duo_carry_team_1", "cs_20m_bottom_duo_carry_team_1", "xp_20m_bottom_duo_carry_team_1", "damege_taken_20m_bottom_duo_carry_team_1"]

bottom_duo_support=["gold_earned_20m_bottom_duo_support_team_1", "cs_20m_bottom_duo_support_team_1", "xp_20m_bottom_duo_support_team_1", "damege_taken_20m_bottom_duo_support_team_1"]
g = sns.pairplot(dataset, vars=top, hue="win", height=3)
g = sns.pairplot(dataset, vars=middle, hue="win", height=3)
g = sns.pairplot(dataset, vars=jungle, hue="win", height=3)
g = sns.pairplot(dataset, vars=bottom_duo_carry, hue="win", height=4)
g = sns.pairplot(dataset, vars=bottom_duo_support, hue="win", height=4)
x = dataset.drop('win', axis=1)

y = dataset['win']
print(x.shape)

print(y.shape)
from sklearn.model_selection import train_test_split



#divides the training dataset into training and testing, separating 25% in testing

#x = attributes e y = classes

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
from sklearn.tree import DecisionTreeClassifier

#create the tree

tree = DecisionTreeClassifier(criterion='gini', splitter='random', max_depth=5, random_state=0)

#create the model

model = tree.fit(x_train, y_train)
from sklearn.metrics import accuracy_score

# prediction of test data 

predict = model.predict(x_test)
acc = accuracy_score(y_test, predict)

print("Accuracy: ", format(acc))
!pip install pydotplus
from sklearn.externals.six import StringIO  

from IPython.display import Image  

from sklearn.tree import export_graphviz

import pydotplus as pydot



dot_data = StringIO()

export_graphviz(model, out_file=dot_data, filled=True, rounded=True,special_characters=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())
test = pd.read_csv("/kaggle/input/league-of-legends-ranked-games/others_tiers.csv")

test.drop(['vilemaw_kills_team_1', 'vilemaw_kills_team_2'], axis=1, inplace=True)

test.head(5)
g = sns.distplot(test["win"], kde=False)
x_test_random = test.drop('win', axis=1)

y_test_random = test['win']

print(x_test_random.shape)

print(y_test_random.shape)
predict_test = model.predict(x_test_random)

acc_test = accuracy_score(y_test_random, predict_test)

print("Accuracy: ", format(acc_test))