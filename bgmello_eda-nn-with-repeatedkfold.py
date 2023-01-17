import os

import warnings



import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import set_matplotlib_formats, display, HTML



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, BatchNormalization

from sklearn.metrics import roc_curve, classification_report

from sklearn.model_selection import RepeatedKFold

from xgboost import XGBClassifier



%matplotlib inline

warnings.filterwarnings('ignore')

plt.style.use('ggplot')

defaultcolor = '#4c72b0'

set_matplotlib_formats('pdf', 'png')

pd.options.display.float_format = '{:.2f}'.format

rc={'savefig.dpi': 75, 'figure.autolayout': False, 'figure.figsize': [12, 8], 'axes.labelsize': 18,\

   'axes.titlesize': 18, 'font.size': 18, 'lines.linewidth': 2.0, 'lines.markersize': 8, 'legend.fontsize': 16,\

   'xtick.labelsize': 16, 'ytick.labelsize': 16}



plt.rcParams['image.cmap'] = 'Blues'



sns.set(style='darkgrid',rc=rc)



data_dir = '../input/' #directory where the dataset is located
train = pd.read_csv(os.path.join(data_dir, 'train.csv'))

test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
train.head()
train.info()
train['Survived'].describe()
fig, ax = plt.subplots()

train['Survived'].value_counts().plot.bar(ax=ax)



plt.xticks(rotation=0);



plt.xticks(ticks=[0,1], labels=['Didn\'t Survived', 'Survived']);



bars = ax.get_children()[:2]



bars[0].set_color('r')



ax.hlines(bars[0].get_height(), bars[0].get_x(), bars[1].get_x()+bars[1].get_width(), linestyles='--')



plt.arrow(bars[1].get_x()+bars[1].get_width()/2, bars[0].get_height(), 0, 

          bars[1].get_height()-bars[0].get_height()+10, color='black', width=0.005, head_length=10)



plt.text(bars[1].get_x()+bars[1].get_width()/2+0.01, (bars[0].get_height()+bars[1].get_height())/2, 

         train[train['Survived']==0].shape[0]-train[train['Survived']==1].shape[0]);
df=train.pivot_table(index='Pclass', values='Survived', aggfunc=np.mean)
fig, ax = plt.subplots(figsize=[15,5])

df=100*train.pivot_table(index='Pclass', values='Survived', aggfunc=np.mean)

df.plot.bar(ax=ax, color=defaultcolor)

for i,p in enumerate(ax.patches):

    ax.annotate('{:.2f}%'.format(df['Survived'][1+i]), (p.get_x()+0.18, p.get_height()+1)).set_fontsize(15)

ax.set(title='Percentage of survivors by Pclass', xlabel='Pclass', ylabel='Percentage', ylim=[0,100]);

plt.xticks(rotation=0);
fig, ax = plt.subplots(figsize=[15,5])

train['Pclass'].value_counts().sort_index().plot.bar(ax=ax, color=defaultcolor)

ax.set(title='Number of passengers by Pclass', xlabel='Pclass', ylabel='# of passangers');

plt.xticks(rotation=0);
fig, ax = plt.subplots()



ax = sns.boxplot(x='Pclass', y='Age', data=train, ax=ax, color=defaultcolor);



ax = sns.pointplot(x='Pclass', y='Age', data=train.groupby('Pclass', as_index=False).mean(), ax=ax, color='g', 

              linestyles='--')



plt.legend((ax.get_children()[22:23]), ['Mean'])
train['Title']=train['Name'].apply(lambda x: x[x.find(',')+2:x.find('.')])

test['Title']=test['Name'].apply(lambda x: x[x.find(',')+2:x.find('.')])
from matplotlib.ticker import PercentFormatter



fig, ax = plt.subplots()

pd.concat([train['Title'].value_counts(), test['Title'].value_counts()], axis=1, sort=False).plot.bar(stacked=True, ax=ax);



tmp_sr = ((train['Title'].value_counts() + test['Title'].value_counts()).fillna(0)/(train.shape[0]+test.shape[0])).copy()



tmp_sr = tmp_sr.sort_values(ascending=False).cumsum()*100



ax.set_ylabel('Number of passengers')

ax.set_xlabel('Titles');





ax2 = ax.twinx()

ax2.plot(tmp_sr.index, tmp_sr, color="g", marker="D", ms=7)

ax2.yaxis.set_major_formatter(PercentFormatter())



ax2.tick_params(axis="y", colors="g")

ax2.set_ylim([0,105])



ax2.grid(False)



ax.legend(['Train', 'Test'], loc='lower right');
temp_series = 100*(train.groupby('Title').count()['PassengerId']/train.shape[0]).copy()



temp_series = temp_series.rename('% of passangers')

with pd.option_context('display.float_format', '{:.3f}'.format):

    display(HTML(pd.concat([train.groupby('Title')['Survived'].mean(), temp_series], axis=1)

                 .sort_values(by='% of passangers', ascending=False).to_html()))
fig, ax = plt.subplots()

dfSex = 100*pd.pivot_table(index='Sex', values='Survived', aggfunc=np.mean, data=train)

dfSex.plot.bar(ax=ax, color=defaultcolor)

plt.xticks(rotation=0)

plt.ylim([0,100])

ax.set(title='Percentage of survivors by sex', xlabel='Sex', ylabel='Percentage');

sex=['female', 'male']

for i,p in enumerate(ax.patches):

    ax.annotate('{:.2f}%'.format(dfSex['Survived'][sex[i]]), (p.get_x()+0.14, p.get_height()+1.0)).set_fontsize(20)
fig, ax = plt.subplots(figsize=[15,5])

sns.kdeplot(train['Age'].sort_values().dropna(), ax=ax, legend=False, color='r')

sns.kdeplot(train[train['Survived']==1]['Age'].sort_values().dropna(), ax=ax, legend=False, color='g')

plt.legend(['Total', 'Survivors'])

plt.grid(True)

plt.title('Distribution of the passangers by age');

ax.set_yticklabels(['0%', '0.5%', '1.0%', '1.5%', '2.0%', '2.5%', '3.0%']);
print("Percentage of missing Ages: {:.2f}%".format(train['Age'].isna().mean()*100))

print("Survival rate of passengers with not missing Age: {:.2f}%".format(train[train['Age'].apply(lambda x: not pd.isna(x))]['Survived'].mean()*100))

print("Survival rate of passengers with missing Age: {:.2f}%".format(train[train['Age'].apply(lambda x: pd.isna(x))]['Survived'].mean()*100))
matrix = np.zeros([9,7])

for i in range(8):

    for j in range(6):

        matrix[i][j]=train[(train['SibSp']==i)&(train['Parch']==j)]['Survived'].mean()

fig, ax = plt.subplots();

sns.heatmap(matrix, annot=True, ax=ax, cbar=False)

ax.set_ylim([0, 9])

ax.set_ylabel('Parch')

ax.set_xlabel('SibSp')

ax.set(title='Survivor rate by SibSp and Parch');
train.groupby(['SibSp', 'Parch']).count()['PassengerId']
train[train['SibSp']==0][train['Parch']==0]['Age'].plot.hist(bins=30)
train[train['SibSp']==0][train['Parch']==0]['Age'].value_counts().sort_index().head(10)
print('Survivor rate for passengers with numerical only tickets: {}%'

      .format(round(train[train['Ticket'].apply(lambda x: x.isdigit())]['Survived'].mean(),2)*100))

print('Percentage of passangers with numerical only tickets: {:.2f}%'.format(100*train['Ticket'].apply(lambda x: x.isdigit()).mean()))

train['Cabin'].apply(lambda x: not pd.isna(x)).mean()
train[train['Cabin'].apply(lambda x: not pd.isna(x))]['Survived'].mean()
train[train['Cabin'].apply(lambda x: pd.isna(x))]['Survived'].mean()
sns.violinplot(x='Survived', y='Fare', data=train)
sns.kdeplot(train['Fare'], shade=True)
train.pivot_table(values='Survived', index='Embarked', aggfunc=np.mean).plot.bar(color=defaultcolor);
train.groupby(['Embarked', 'Pclass']).mean()
fig, ax = plt.subplots(figsize=[15,10])

sns.heatmap(train.isna(), ax=ax, cbar=False, yticklabels=False)

ax.set_title("NaN in each label for train set");

fig2, ax2 = plt.subplots(figsize=[15,10])

sns.heatmap(test.isna(), ax=ax2, cbar=False, yticklabels=False)

ax2.set_title("NaN in each label for test set");
train[train['Embarked'].isnull()]
train[train['Ticket'].str.startswith('1135')]
train.pivot_table(index='Embarked', values='Fare', aggfunc=np.mean)
train.pivot_table(index='Embarked', values='Pclass', aggfunc=np.mean)
train['Embarked'].fillna('C', inplace=True)
def fill_age_train(cols):

    age = cols[0]

    pclass = cols[1]

    embarked = cols[2]

    if pd.isna(age):

        return train[train['Pclass']==pclass][train['Embarked']==embarked]['Age'].mean()

    else:

        return age
def fill_age_test(cols):

    age = cols[0]

    pclass = cols[1]

    embarked = cols[2]

    if pd.isna(age):

        return test[test['Pclass']==pclass][test['Embarked']==embarked]['Age'].mean()

    else:

        return age
train['Age'] = train[['Age', 'Pclass','Embarked']].apply(fill_age_train, axis=1)
test['Age'] = test[['Age', 'Pclass','Embarked']].apply(fill_age_test, axis=1)
test[test['Fare'].isna()]
mean = test[test['Pclass']==3][test['Embarked']=='S']['Fare'].mean()

test['Fare'].fillna(mean, inplace=True)
train['missingAge']=train['Age'].apply(lambda x: 1 if pd.isna(x) else 0)

test['missingAge']=test['Age'].apply(lambda x: 1 if pd.isna(x) else 0)
train['Title']=train['Title'].apply(lambda x: x if x in ['Mr', 'Miss', 'Mrs'] else np.nan)

test['Title']=test['Title'].apply(lambda x: x if x in ['Mr', 'Miss', 'Mrs'] else np.nan)

trainTitledf = pd.get_dummies(data=train['Title'], prefix='Title')

testTitledf = pd.get_dummies(data=test['Title'], prefix='Title')

train = pd.concat([train, trainTitledf], axis=1)

test = pd.concat([test, testTitledf], axis=1)
train = pd.concat([train,pd.get_dummies(train['Sex'], prefix='Sex', drop_first=True)], axis=1)

test = pd.concat([test,pd.get_dummies(test['Sex'], prefix='Sex', drop_first=True)], axis=1)
train = pd.concat([train,pd.get_dummies(train['Embarked'], prefix='Embarked', drop_first=True)], axis=1)

test = pd.concat([test,pd.get_dummies(test['Embarked'], prefix='Embarked', drop_first=True)], axis=1)
train['missingCabin'] = train['Cabin'].apply(lambda x: 1 if not pd.isna(x) else 0)

test['missingCabin'] = test['Cabin'].apply(lambda x: 1 if not pd.isna(x) else 0)
X_train = np.array(train.drop(['PassengerId', 'Survived', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Title'], axis=1))

X_test = np.array(test.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Title'], axis=1))
y_train = np.array(train['Survived'])
def build_model():

    model = XGBClassifier(learning_rate=0.01, n_estimators=500, n_jobs=-1)

    return model
def KfoldAnalysis(X, y):

    models = []

    rkfold = RepeatedKFold(n_splits=5, n_repeats=10)

    for train_index, val_index in rkfold.split(X):

        model = build_model()

        model.fit(X[train_index], y[train_index])

        models.append(model)

    return models
models = KfoldAnalysis(X_train, y_train)
y_pred = np.array([model.predict(X_test) for model in models])
sns.distplot(np.mean(y_pred, 0).reshape(-1), bins=50)

plt.vlines(0.4, 0, 10)
ans = [1 if i>0.4 else 0 for i in np.mean(y_pred, 0).reshape(-1)]
np.mean(y_train)
np.mean(ans)
df = pd.DataFrame({'PassengerId' : test['PassengerId'], 'Survived' : pd.Series(ans)})

df.to_csv("submission.csv", index=False)