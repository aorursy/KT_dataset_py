from IPython.display import clear_output

! pip install seaborn==0.11.0

clear_output()
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
df = pd.read_csv('/kaggle/input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')

print(f'Columns: {df.columns}')
sns.displot(data=df, x="blueGoldDiff", hue="blueWins")

plt.show()
fig, ax = plt.subplots(1,2, figsize=(12,8))



f1 = sns.stripplot(x="redAvgLevel", y="redWardsPlaced", 

                 hue="blueWins", data=df, alpha=0.5,

                 ax=ax[0])



f2 = sns.stripplot(x="blueAvgLevel", y="blueWardsPlaced", 

                 hue="blueWins", data=df, alpha=0.5,

                 ax=ax[1])



f1.set_xticklabels(f1.get_xticklabels(), rotation=45)

f2.set_xticklabels(f2.get_xticklabels(), rotation=45)

plt.show()
fig, ax = plt.subplots(1,2, figsize=(12,8))



f1 = sns.stripplot(x="redAvgLevel", y="redKills", 

                 hue="blueWins", data=df, alpha=0.5,

                 ax=ax[0])



f2 = sns.stripplot(x="blueAvgLevel", y="blueKills", 

                 hue="blueWins", data=df, alpha=0.5,

                 ax=ax[1])



f1.set_xticklabels(f1.get_xticklabels(), rotation=45)

f2.set_xticklabels(f2.get_xticklabels(), rotation=45)

plt.show()
f1 = sns.stripplot(x="blueAvgLevel", y="blueExperienceDiff", 

                 hue="blueWins", data=df, alpha=0.5,)



f1.set_xticklabels(f1.get_xticklabels(), rotation=45)

plt.show()
fig, ax = plt.subplots(2,2, figsize=(14,12))



sns.countplot(data=df, x="redDragons", hue="blueWins", ax=ax[0,0])

sns.countplot(data=df, x="redHeralds", hue="blueWins", ax=ax[0,1])

sns.countplot(data=df, x="blueDragons", hue="blueWins", ax=ax[1,0])

sns.countplot(data=df, x="blueHeralds", hue="blueWins", ax=ax[1,1])

plt.show()
ax = sns.catplot(x="redAvgLevel", y="blueGoldDiff", 

                 hue="blueWins", data=df, alpha=0.3)

ax.set_xticklabels(rotation=45)

plt.show()
df_bl = df.query('blueWins != 0 & blueGoldDiff > 0')
print(f'Red Average Level: {df_bl.redAvgLevel.mean()}')

print(f'Blue Average Level: {df_bl.blueAvgLevel.mean()}')

print(f'Red Average Kills: {df_bl.redKills.mean()}')

print(f'Blue Average Kills: {df_bl.blueKills.mean()}')
fig, ax = plt.subplots(2,2, figsize=(14,12))



sns.countplot(data=df_bl, x="redDragons", ax=ax[0,0])

sns.countplot(data=df_bl, x="redHeralds", ax=ax[0,1])

sns.countplot(data=df_bl, x="blueDragons", ax=ax[1,0])

sns.countplot(data=df_bl, x="blueHeralds", ax=ax[1,1])

plt.show()
print(f"Probability of winning when you have a gold lead on the blue side: {np.round(len(df.query('blueWins == 1 & blueGoldDiff > 0'))/len(df),3)}")

print(f"Probability of losing when you have a gold lead on the blue side: {np.round(len(df.query('blueWins != 1 & blueGoldDiff > 0'))/len(df),3)}")

print(f"Probability of winning when you have a gold lead on the red side: {np.round(len(df.query('blueWins != 1 & blueGoldDiff < 0'))/len(df),3)}")

print(f"Probability of losing when you have a gold lead on the red side: {np.round(len(df.query('blueWins == 1 & blueGoldDiff < 0'))/len(df),3)}")
red_features = [f for f in df.columns if "red" in f]

blue_features = [f for f in df.columns if "blue" in f]
blue_df = df[blue_features]

blue_df.head()
x = blue_df.drop("blueWins", axis=1)

y = blue_df["blueWins"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=0)

randomforest.fit(X_train,y_train)
from sklearn.metrics import roc_curve, auc



def roc(y_test, y_pred, model_name, title="ROC"):

    """Creates and plots the roc for a model.

    """

    

    fpr, tpr, _ = roc_curve(y_test, y_pred)

    roc_auc = auc(fpr, tpr)

    lw = 2

    plt.plot(fpr, tpr,

             lw=lw, label=f'{model_name} ROC curve area = {roc_auc:0.2f}')

    plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title(title)

    plt.legend(loc="lower right")
from sklearn.metrics import average_precision_score

y_pred_RF = randomforest.predict_proba(X_test)

print(f"Accuracy: {np.around(sum(np.argmax(y_pred_RF, axis=1) == y_test)/len(y_test)*100,1)}%")

average_precision = average_precision_score(y_test, np.argmax(y_pred_RF, axis=1))

roc(y_test, y_pred_RF[:,1], "Random Forest")