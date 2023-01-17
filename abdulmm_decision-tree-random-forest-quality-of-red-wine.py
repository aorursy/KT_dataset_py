import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
wine = pd.read_csv("../input/winequality-red.csv")
wine.head()
quality_dist = wine['quality'].value_counts()
plt.bar(quality_dist.index, quality_dist)
plt.xlabel('quality')
plt.ylabel('frequency')
plt.show()
wine['quality'].describe()
values, base = np.histogram(wine['quality'], bins=20)
kumulativ = np.cumsum(values/wine.shape[0])
plt.plot(base[:-1], kumulativ, c='blue')
plt.xlabel('quality')
plt.ylabel('frequency')
plt.show()
indeksDaarlig = wine.loc[wine['quality'] <= 6].index
indeksGod = wine.loc[wine['quality'] > 6].index
wine.iloc[indeksDaarlig, wine.columns.get_loc('quality')] = 0
wine.iloc[indeksGod, wine.columns.get_loc('quality')] = 1
wine['quality'].value_counts()
x = wine.drop('quality',axis=1)
y = wine['quality']
#Choosing 40% as training data.
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.40, random_state = 42)
# Making a decision tree with two levels.
clfTre = tree.DecisionTreeClassifier(max_depth=2)
clfTre.fit(xTrain, yTrain)
#Visualizing the decision tree
dot_data = tree.export_graphviz(clfTre, out_file=None, max_depth=2, feature_names=list(x.columns.values), filled=True, rounded=True)
valgTre = graphviz.Source(dot_data) 
valgTre
utfall = (clfTre.predict(xTest) == yTest).value_counts()
print("The decision tree predicts the test data in", (utfall[1]/(utfall[0]+utfall[1]))*100 , "% of the cases.")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
rf.fit(xTrain, yTrain)
utfall = (rf.predict(xTest) == yTest).value_counts()
print("The decision tree predicts the test data in", (utfall[1]/(utfall[0]+utfall[1]))*100 , "% of the cases.")