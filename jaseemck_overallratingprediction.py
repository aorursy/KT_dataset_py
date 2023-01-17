import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
data = pd.read_csv("/kaggle/input/fifa19/data.csv")
data.head()
data.columns
data.corr()
data.shape
#Using Pearson Correlation

plt.figure(figsize=(25,25))

cor = data.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
X = data.drop("Overall",1)   #Feature Matrix

y = data["Overall"]          #Target Variable
cor_target = abs(cor["Overall"])
relevant_features = cor_target[cor_target>0.2]
relevant_features
X = data[['Potential', 'Special', 'ShortPassing', 'Reactions', 'Composure', 'Unnamed: 0', 'International Reputation', 'Vision', 'Age', 'LongPassing', 'Skill Moves', 'Curve', 'BallControl', 'ShotPower', 'LongShots', 'Crossing', 'Finishing', 'HeadingAccuracy', 'Volleys', 'Dribbling', 'FKAccuracy', 'Stamina', 'Strength', 'Aggression', 'Interceptions', 'Positioning', 'Penalties', 'Jumping', 'Marking', 'StandingTackle', 'Weak Foot', 'SprintSpeed', 'Agility', 'SlidingTackle']]
X.isnull().sum()
X['ShortPassing'].unique
X['Reactions'].unique
X['ShortPassing'].fillna((X['ShortPassing'].median()), inplace=True)

X['Reactions'].fillna((X['Reactions'].median()), inplace=True)

X['Composure'].fillna((X['Composure'].median()), inplace=True)

X['International Reputation'].fillna((X['International Reputation'].median()), inplace=True)

X['Vision'].fillna((X['Vision'].median()), inplace=True)

X['Age'].fillna((X['Age'].median()), inplace=True)

X['LongPassing'].fillna((X['LongPassing'].median()), inplace=True)

X['Skill Moves'].fillna((X['Skill Moves'].median()), inplace=True)

X['Curve'].fillna((X['Curve'].median()), inplace=True)

X['BallControl'].fillna((X['BallControl'].median()), inplace=True)

X['ShotPower'].fillna((X['ShotPower'].median()), inplace=True)

X['LongShots'].fillna((X['LongShots'].median()), inplace=True)

X['Crossing'].fillna((X['Crossing'].median()), inplace=True)

X['Dribbling'].fillna((X['Dribbling'].median()), inplace=True)

X['Finishing'].fillna((X['Finishing'].median()), inplace=True)

X['Positioning'].fillna((X['Positioning'].median()), inplace=True)

X['Interceptions'].fillna((X['Interceptions'].median()), inplace=True)

X['FKAccuracy'].fillna((X['FKAccuracy'].median()), inplace=True)

X['Strength'].fillna((X['Strength'].median()), inplace=True)

X['Aggression'].fillna((X['Aggression'].median()), inplace=True)

X['Stamina'].fillna((X['Stamina'].median()), inplace=True)

X['HeadingAccuracy'].fillna((X['HeadingAccuracy'].median()), inplace=True)

X['Volleys'].fillna((X['Volleys'].median()), inplace=True)

X['Penalties'].fillna((X['Penalties'].median()), inplace=True)

X['Jumping'].fillna((X['Jumping'].median()), inplace=True)

X['StandingTackle'].fillna((X['StandingTackle'].median()), inplace=True)

X['SprintSpeed'].fillna((X['SprintSpeed'].median()), inplace=True)

X['Marking'].fillna((X['Marking'].median()), inplace=True)

X['Weak Foot'].fillna((X['Weak Foot'].median()), inplace=True)

X['Agility'].fillna((X['Agility'].median()), inplace=True)

X['SlidingTackle'].fillna((X['SlidingTackle'].median()), inplace=True)
X.shape
from sklearn import datasets, linear_model, metrics

# splitting X and y into training and testing sets 

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) 
reg = linear_model.LinearRegression() 

reg.fit(X_train, y_train) 
y_pred = reg.predict(X_test)
print("MAE : " + str(metrics.mean_absolute_error(y_test, y_pred)))

print("MSE : ", str(metrics.mean_squared_error(y_test, y_pred)))

print("Final rmse value is =", str(np.sqrt(np.mean((y_test-y_pred)**2))))