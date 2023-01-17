# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
from matplotlib import pyplot as plt
corr = train_data.corr()

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,len(train_data.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(train_data.columns)

ax.set_yticklabels(train_data.columns)

plt.show()
##Age and Sex are positively correlated(mid-high)

##survived and SibSp are positively correlated(somewhat)

##Survived and Pclass are negatively correlated

##SibSp and Pclass are negatively correlated(mid- high)

##name and Pclass are negtively correlated(low)
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
from sklearn.ensemble import GradientBoostingClassifier

y_train = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]



X_features_train = pd.get_dummies(train_data[features])

X_features_test = pd.get_dummies(test_data[features])



gb = GradientBoostingClassifier(n_estimators = 150, max_depth = 15, learning_rate = 0.1)

gb_model= gb.fit(X_features_train,y_train)

y_pred = gb_model.predict(X_features_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_pred})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")    

    
