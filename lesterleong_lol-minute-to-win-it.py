# Load packages and dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style('darkgrid')
df = pd.read_csv('../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')
df.head()
# check missing values and data type
df.info()
df.isnull().values.any()
df = df.drop(columns='gameId')
#Checking if removed
df.head()
# Next let's check the relationship between parameters of blue team features
g = sns.PairGrid(data=df, vars=['blueKills', 'blueAssists', 'blueWardsPlaced', 'blueTotalGold'], hue='blueWins', size=3, palette='Set1')
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend();
# We can see that a lot of the features are highly correlated, let's get the correlation matrix
plt.figure(figsize=(16, 12))
sns.heatmap(df.drop('blueWins', axis=1).corr(), cmap='YlGnBu', annot=True, fmt='.2f', vmin=0);
# train test split scale the set
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
X = df.drop(columns='blueWins')
y = df['blueWins']

# I am chosing not to scale the data first, since I am going the tree route.
#Decision trees and ensemble methods do not require feature scaling to be performed as they are not sensitive to the the variance in the data.

##scaler = MinMaxScaler()
##scaler.fit(X)
##X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# fit the model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import metrics

gb = GradientBoostingClassifier()

# search the best params
grid = {'n_estimators':[100,200,300,400,500], 'max_depth': [2, 5, 10]}

clf_gb = GridSearchCV(gb, grid, cv=5)
clf_gb.fit(X_train, y_train)

pred_gb = clf_gb.predict(X_test)

# get the accuracy score
acc_gb = accuracy_score(pred_gb, y_test)
print(acc_gb)
# confusion matrix
from sklearn.metrics import plot_confusion_matrix
# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf_gb, X_test, y_test,
                                 #display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()
# Getting settings from GridCV
clf_gb.best_params_
#inputting best settings from GridCV
gb = GradientBoostingClassifier(n_estimators=100, max_depth=3)
clf_gb.fit(X_train, y_train)
from sklearn.inspection import permutation_importance

r = permutation_importance(clf_gb, X_test, y_test,
                           n_repeats=20,
                           random_state=0)

for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{X_train.columns[i]:<10}"
              f"{r.importances_mean[i]:.3f}"
              f" +/- {r.importances_std[i]:.3f}")
