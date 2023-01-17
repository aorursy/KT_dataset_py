import pandas as pd



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
test.head()
train.info()
test.info()
train.shape
test.shape
train.isnull().sum()
test.isnull().sum()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()
def bar_chart(feature):

    survived = train[train['Survived'] == 1][feature].value_counts()

    dead = train [train['Survived'] == 0][feature].value_counts()

    df = pd.DataFrame([survived,dead])

    df.plot(kind = 'bar', stacked = True, figsize = (15,15))
bar_chart('Pclass')
bar_chart('Sex')
bar_chart('SibSp')
bar_chart('Parch')
bar_chart('Embarked')
bar_chart('Cabin')
bar_chart('Survived')