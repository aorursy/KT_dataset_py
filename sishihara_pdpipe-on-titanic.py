# https://pdpipe.github.io/pdpipe/

!pip install pdpipe
import pandas as pd

import pdpipe as pdp
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
train.head()
train.isnull().sum()
CATEGORICAL_COL = ['Sex', 'SibSp', 'Embarked']

DROP_COL = ['PassengerId', 'Name', 'Ticket', 'Parch', 'Cabin']

# NUMERICAL_COL = ['Pclass', 'Age', 'Fare']

# TARGET_COL = 'Survived'
# https://pdpipe.github.io/pdpipe/doc/pdpipe/

pipeline = pdp.PdPipeline([

    pdp.Encode(CATEGORICAL_COL),

    pdp.ColDrop(DROP_COL)

])
for c in CATEGORICAL_COL:

    train[c].fillna('<missing>', inplace=True)

    test[c].fillna('<missing>', inplace=True)
print(set(test['Parch']) - set(train['Parch']))
train = pipeline.fit_transform(train, verbose=True)

test = pipeline.transform(test)
train.head()
test.head()