import pandas, seaborn

data = pandas.read_csv('../input/titanic/train.csv') #data

_ = seaborn.countplot(x='Sex', hue='Survived', data=data)

data.groupby('Sex')['Survived'].mean()