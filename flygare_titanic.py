import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv(r'../input/train.csv')

test = pd.read_csv(r'../input/test.csv')

print(f'Dataset train / test lenght: {len(train)} / {len(test)}')

print(f'Dataset train attributes {set(train.columns)}, count: {len(train.columns)}')

print(f'Missing in test {set(train.columns) - set(test.columns)}')

print(f'Missing in train {set(test.columns) - set(train.columns)}')
test['Survived']=-1

test['set']='test'

train['set']='train'

full=pd.concat([test,train]);
print(f"""Dataset train goes from {train.PassengerId.min()} to {train.PassengerId.max()} \

and test from {test.PassengerId.min()} to {test.PassengerId.max()}""")
print(f'Check Values (Survived): {full.Survived.unique()}')

mapSurvived={0: 'not survived', 1: 'survived', -1: 'to be predicted'}

full['_Survived_']=full['Survived'].map(mapSurvived)

g=full.groupby(['set','_Survived_'])['PassengerId'].agg({'Count': np.count_nonzero})

for s in ['test','train']:

    g.loc[s,'Pct']=g['Count']/len(full[full.set==s])

g.unstack('set', fill_value='')
print(f'Check Values: {full.Embarked.unique()}')

mapSurvived={'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southhampton'}

full['Embarked1']=full['Embarked'].fillna('U')

g=full.groupby(['set','Embarked1'])['PassengerId'

                         ].agg({'Count': np.count_nonzero})

for s in ['train', 'test']:

    g.loc[s,'Pct']=g['Count']/len(full[full.set==s])

g.unstack('set', fill_value='')
print(f'Check Values: {full.Sex.unique()}')

g=full.groupby(['set','Sex'])['PassengerId'

                         ].agg({'Count': np.count_nonzero})

for s in ['test','train']:

    g.loc[s,'Pct']=g['Count']/len(full[full.set==s])

g.unstack('set', fill_value='')
print(f'Check Values: {full.Parch.unique()}')

g=full.groupby(['set','Parch'])['PassengerId'

                         ].agg({'Count': np.count_nonzero})

for s in ['test','train']:

    g.loc[s,'Pct']=g['Count']/len(full[full.set==s])

g.unstack('set', fill_value='')
print(f'Check Values: {full.SibSp.unique()}')

g=full.groupby(['set','SibSp'])['PassengerId'

                         ].agg({'Count': np.count_nonzero})

for s in ['test','train']:

    g.loc[s,'Pct']=g['Count']/len(full[full.set==s])

g.unstack('set', fill_value='')
print(f'Check Values: {full.Age.unique()}')

print('Below age 1yr')

print(full[full.Age<1].groupby('set')['PassengerId'].count())

print('No age given')

print(full[full.Age.isnull()].groupby('set')['PassengerId'].count())

print('Age estimated')

print(full[full.Age%1==0.5].groupby('set')['PassengerId'].count())
print(f'Check Values: {full.Pclass.unique()}')

g=full.groupby(['set','Pclass'])['PassengerId'

                         ].agg({'Count': np.count_nonzero})

for s in ['test','train']:

    g.loc[s,'Pct']=g['Count']/len(full[full.set==s])

g=g.sort_index()

g.unstack('set', fill_value='')

print(f'Check Values: {full.Ticket.nunique()}')

print(f'Check Values: {full.Ticket.unique()}')

print(f'Check Values: {full.Cabin.nunique()}')

print(f'Check Values: {full.Cabin.unique()}')

print('Missing value')

print(full[full.Cabin.isnull()].groupby('set')['PassengerId'].count())

full['has_Cabin']=full.Cabin.replace('(.)',1, regex=True).fillna(0)
print(f'Check Values: {full.Fare.nunique()}')

print(f'Check Values: {full.Fare.unique()}')

g=full.groupby(['set','Pclass', 'has_Cabin'])['PassengerId'

                         ].agg({'Count': np.count_nonzero})

for s in ['test','train']:

    g.loc[s,'Pct']=g['Count']/len(full[full.set==s])

g
corr = full.corr()

fig, ax = plt.subplots(figsize = (12, 10))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

map   = sns.heatmap(

        corr, 

        cmap = plt.cm.coolwarm,

        square=True, 

        cbar_kws={'shrink': .9}, 

        ax=ax, 

        annot = True, 

        annot_kws={'fontsize': 12})
colormap = plt.cm.viridis

plt.figure(figsize=(12,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(full.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)