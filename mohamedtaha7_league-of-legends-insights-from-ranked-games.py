# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization

import seaborn as sns # data visualization

from sklearn.model_selection import train_test_split # Training/Testing set split for performance measures

from sklearn.tree import DecisionTreeClassifier, plot_tree # Rule based classifier

from sklearn.metrics import accuracy_score, confusion_matrix # Performance measures



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# importing data

dataset = pd.read_csv("../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv")

dataset.info()
dataset.head()
dataset = dataset.drop(['gameId', 'redFirstBlood', 'redGoldDiff', 'redExperienceDiff'], axis=1) #Redundant Columns
dataset.describe()
palette=sns.color_palette(['r', 'b'])

ax1 = sns.countplot(dataset.blueWins, palette=palette)

ax1.set(xticks=[0, 1], xticklabels=['Red Wins', 'Blue Wins'])

ax1.set_title('Blue vs Red Wins')

ax1.set_xlabel('')
fig, ax = plt.subplots(figsize=(12, 12))

sns.heatmap(dataset.corr(), ax=ax, cmap='seismic_r')
def top_n_correlations(dataframe, n):

    

    correlations = dataframe.corr().unstack()

    correlations = correlations['blueWins'].abs().sort_values(kind='quicksort', ascending=False)

    

    if not n:

        return correlations

    

    return correlations[0:n]
correlations = top_n_correlations(dataset, 11).drop('blueWins')

correlations
grid = sns.PairGrid(data = dataset, vars=['blueGoldDiff', 'blueExperienceDiff', 'blueGoldPerMin', 'blueTotalGold', 'blueTotalExperience'], hue='blueWins', palette=palette, hue_kws={"marker": ["D", "o"], "alpha": [0.3, 0.3]})

grid.map_diag(plt.hist)

grid.map_offdiag(plt.scatter)

grid.add_legend()

plt.show()
X_train, X_test, y_train, y_test = train_test_split(dataset.drop(['blueWins'], axis=1), dataset['blueWins'], test_size=0.33, random_state=0)

print('Training examples: ', X_train.shape[0])

print('Testing examples: ',X_test.shape[0])
def test_classifier(classifier, X_train, X_test, y_train, y_test):

    

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    

    return f'{round(accuracy,2)*100}%', cm
classifier = DecisionTreeClassifier(max_depth=3)

accuracy, cm = test_classifier(classifier, X_train, X_test, y_train, y_test)

print("Decision Tree's Accuracy: ", accuracy)

plt.figure(figsize=(8,8))

ax = sns.heatmap(cm, annot=True, cmap='seismic_r', fmt='g')

ax.set_title("Model's Confusion Matrix")

ax.set_ylabel('Actual')

ax.set_xlabel('Predicted')
fig, ax = plt.subplots(figsize=(20,10))

plot_tree(classifier, feature_names=dataset.columns[1:], class_names=['redWins', 'blueWins'], precision=1, filled=True, ax=ax)

plt.show()