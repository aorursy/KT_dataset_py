# import and prepare data

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

df = pd.read_csv('../input/nj-pick-6-lottery-results-jan-1981-mar-2020/NJ Pick 6 - Winning Numbers - Winning-Numbers-Only.csv')

# Show first 5 entries
df.head()
# Dataframe Size
print(f'Dataframe Size: {df.shape}')
df.loc[:, ['Ball1', 'Ball2', 'Ball3', 'Ball4', 'Ball5', 'Ball6']]
# Show information about the dataframe
df.info()
import seaborn as sns
import matplotlib as mpl

mpl.rc("figure", figsize=(12, 12))

#Frequency of Ball #1
sns.countplot(x="Ball1", data=df)
#Frequency of Ball #2
sns.countplot(x="Ball2", data=df)
#Frequency of Ball #3
sns.countplot(x="Ball3", data=df)
#Frequency of Ball #4
sns.countplot(x="Ball4", data=df)
#Frequency of Ball #5
sns.countplot(x="Ball5", data=df)
#Frequency of Ball #6
sns.countplot(x="Ball6", data=df)
from scipy import stats
sns.distplot(df.sum(axis=1), fit=stats.gamma)
df.mode()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

X = np.array([df["Ball1"], df["Ball2"], df["Ball3"], df["Ball4"], df["Ball5"], df["Ball6"]] )
bindex = 0
final = []

for ball in [ df["Ball1"], df["Ball2"], df["Ball3"], df["Ball4"], df["Ball5"], df["Ball6"] ]:
    Y = np.array(ball.values.tolist())
    X_train, X_test, y_train, y_test = train_test_split(X.transpose(), Y, test_size=0.9, random_state=None)
    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)               # prediction
    accuracy = accuracy_score(y_test, y_pred)  # check accuracy
    print(f"Prediction of Ball {bindex + 1} is [{y_pred[bindex]}] with tested accuracy of {accuracy * 100}")
    final.append(y_pred[bindex])
    bindex = bindex + 1
    
from scipy import stats

print(f"Predicted Numbers: {final}")

S = sum(final)
print(f"Sum of numbers: {S}")
print(f"Sum is good!") if S >= 100 and S <= 200 else print(f"Sum of prediction is out of ideal range. Re-run prediction.")

