from speedml import Speedml



%matplotlib inline
sml = Speedml('../input/train.csv', '../input/test.csv', 

              target = 'Survived', uid = 'PassengerId')

sml.shape()
sml.train.head()
sml.train.describe()
sml.train.info()

print('-'*40)

sml.test.info()
sml.plot.correlate()
sml.plot.distribute()
sml.plot.ordinal('Parch')
sml.plot.ordinal('SibSp')
sml.plot.continuous('Age')
sml.plot.continuous('Fare')
sml.plot.crosstab('Survived', 'Pclass')
sml.plot.crosstab('Survived', 'Parch')
sml.plot.crosstab('Survived', 'SibSp')
sml.plot.crosstab('Survived', 'Sex')
sml.plot.crosstab('Survived', 'Embarked')