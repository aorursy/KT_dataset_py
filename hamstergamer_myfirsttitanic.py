import numpy as np 
import pandas as pd

%matplotlib inline 
from matplotlib import pyplot as plt
from matplotlib import style
import seaborn as sns

import string
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
train_frame = pd.read_csv('/kaggle/input/titanic/train.csv')
test_frame = pd.read_csv("/kaggle/input/titanic/test.csv")
survive_women = train_frame.loc[train_frame.Sex == 'female']["Survived"]
rate_women = sum(survive_women)/len(survive_women)

survive_men = train_frame.loc[train_frame.Sex == 'male']["Survived"]
rate_men = sum(survive_men)/len(survive_men)

print("% of men who survived:", rate_men)
print("% of women who survived:", rate_women)
male = train_frame[train_frame['Sex']=='male']
female = train_frame[train_frame['Sex']=='female']
x = male[male['Survived']==1].Age.dropna()
x1 = male[male['Survived']==0].Age.dropna()
y = female[female['Survived']==1].Age.dropna()
y1 = female[female['Survived']==0].Age.dropna()

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
ax = sns.distplot(x, bins=15, label = 'survived', ax = axes[0], kde = False, color = 'blue')
ax = sns.distplot(x1, bins=30, label = 'not survived', ax = axes[0], kde = False, color = 'black')
ax.legend()
ax.set_title('Male')
ax = sns.distplot(y, bins=15, label = 'survived', ax = axes[1], kde = False, color = 'purple')
ax = sns.distplot(y1, bins=30, label = 'not survived', ax = axes[1], kde = False, color = 'black')
ax.legend()
ax.set_title('Female')
plt.show()
y = train_frame["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_frame[features])
X_test = pd.get_dummies(test_frame[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_frame.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Gender based solution ready!")