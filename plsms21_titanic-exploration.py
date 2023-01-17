# import standard libraries
import numpy as np
import pandas as pd
import seaborn as sns
import re
titanic = pd.read_csv('../input/train.csv')
titanic.set_index('PassengerId', inplace=True)
titanic.describe()
titanic.head()
features = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
# Heatmap
sns.heatmap((titanic.loc[:, features]).corr(),
            annot=True);
features = ['Survived', 'Pclass', 'SibSp', 'Parch', 'Fare']
# Pairplot
sns.pairplot(titanic[features]);