import pandas as pd

import numpy as np

import warnings

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')



from sklearn import preprocessing 

from sklearn.model_selection import GridSearchCV 

from sklearn.ensemble import RandomForestClassifier 

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn import cross_validation



# 在notobook里的命令，将那些用matplotlib绘制的图显示在页面里而不是弹出一个窗口

%matplotlib inline

plt.style.use(style='_classic_test')

plt.style.use(style='seaborn')

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
train.info()

print('-'*30)

test.info()
train['Age'].fillna(train['Age'].median(), inplace=True)

train['Embarked'].fillna('S', inplace=True)
train.info()
# Sex



survived_sex = train[train['Survived']==1]['Sex'].value_counts()

dead_sex = train[train['Survived']==0]['Sex'].value_counts()

df = pd.DataFrame([survived_sex,dead_sex])

df.index = ['Survived','Dead']

df.plot(kind='bar',stacked=True, figsize=(13,8))
# Age



figure = plt.figure(figsize=(13,8))

plt.hist([train[train['Survived']==1]['Age'],train[train['Survived']==0]['Age']], stacked=True, color = ['g','r'],

         bins = 30,label = ['Survived','Dead'])

plt.xlabel('Age')

plt.ylabel('Number of passengers')

plt.legend()
# Fare



figure = plt.figure(figsize=(13,8))

plt.hist([train[train['Survived']==1]['Fare'],train[train['Survived']==0]['Fare']], stacked=True, color = ['g','r'],

         bins = 30,label = ['Survived','Dead'])

plt.xlabel('Fare')

plt.ylabel('Number of passengers')

plt.legend()
# Fare & Age



plt.figure(figsize=(13,8))

ax = plt.subplot()

ax.scatter(train[train['Survived']==1]['Age'],train[train['Survived']==1]['Fare'],c='green',s=40)

ax.scatter(train[train['Survived']==0]['Age'],train[train['Survived']==0]['Fare'],c='red',s=40)

ax.set_xlabel('Age')

ax.set_ylabel('Fare')

ax.legend(('survived','dead'),scatterpoints=1,loc='upper right',fontsize=15,)
# Pclass



sns.factorplot('Pclass','Survived',order=[1,2,3], data=train,size=5)
# Embarked



sns.factorplot('Embarked','Survived', data=train,size=4,aspect=3)



fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

sns.countplot(x='Embarked', data=train, ax=axis1)

sns.countplot(x='Survived', hue="Embarked", data=train, order=[1,0], ax=axis2)

sns.barplot(x='Embarked', y='Survived', data=train,order=['S','C','Q'],ax=axis3)
# Parch & SibSp



train['Family_Size'] =  train["Parch"] + train["SibSp"]



figure = plt.figure(figsize=(13,8))

plt.hist([train[train['Survived']==1]['Family_Size'],train[train['Survived']==0]['Family_Size']], stacked=True, color = ['g','r'],

         bins = 30,label = ['Survived','Dead'])

plt.xlabel('Family_Size')

plt.ylabel('Number of passengers')

plt.legend()
# 合并测试集和数据集，方便数据处理



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



combined = train.drop('Survived', axis=1).append(test)

combined.head()
combined.shape
# Sex 



# 男性为1，女性为0

combined['Sex'] = combined['Sex'].apply(lambda x: 1 if x == 'male' else 0)
# Name



def Name_Title_Code(x):

    if x == 'Mr.':

        return 1

    if (x == 'Mrs.') or (x=='Ms.') or (x=='Lady.') or (x == 'Mlle.') or (x =='Mme'):

        return 2

    if x == 'Miss.':

        return 3

    if x == 'Rev.':

        return 4

    return 5



combined['Name_Title'] = combined['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])

combined['Name_Title'] = combined['Name_Title'].apply(Name_Title_Code)

del combined['Name']
# Age



combined['Age_Null_Flag'] = combined['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)  

data = combined[:891].groupby(['Name_Title', 'Pclass'])['Age']

combined['Age'] = data.transform(lambda x: x.fillna(x.mean()))
# SibSp, Parch



combined['Fam_Size'] = np.where((combined['SibSp']+combined['Parch']) == 0 , 'Singleton',

                           np.where((combined['SibSp']+combined['Parch']) <= 3,'Small', 'Big'))

del combined['SibSp']

del combined['Parch']
# Ticket



combined['Ticket_Lett'] = combined['Ticket'].apply(lambda x: str(x)[0])

combined['Ticket_Lett'] = combined['Ticket_Lett'].apply(lambda x: str(x))

combined['Ticket_Lett'] = np.where((combined['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), combined['Ticket_Lett'],

                                    np.where((combined['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),

                                            'Low_ticket', 'Other_ticket'))

combined['Ticket_Len'] = combined['Ticket'].apply(lambda x: len(x))

del combined['Ticket']
# Cabin



combined['Cabin_Letter'] = combined['Cabin'].apply(lambda x: str(x)[0])

combined['Cabin_num1'] = combined['Cabin'].apply(lambda x: str(x).split(' ')[-1][1:])

combined['Cabin_num1'].replace('an', np.NaN, inplace = True)

combined['Cabin_num1'] = combined['Cabin_num1'].apply(lambda x: int(x) if not pd.isnull(x) and x != '' else np.NaN)

combined['Cabin_num'] = pd.qcut(combined['Cabin_num1'][:891],3)

combined = pd.concat((combined, pd.get_dummies(combined['Cabin_num'], prefix = 'Cabin_num')), axis = 1)



del combined['Cabin']

del combined['Cabin_num']

del combined['Cabin_num1']
# Embarked



combined['Embarked'].fillna('S', inplace=True)
# Fare



combined['Fare'].fillna(combined['Fare'].mean(), inplace = True)
# dummies虚拟变量



columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Lett', 'Cabin_Letter', 'Name_Title', 'Fam_Size']

for column in columns:

    combined[column] = combined[column].apply(lambda x: str(x))

    good_cols = [column+'_'+i for i in combined[:891][column].unique() if i in combined[891:][column].unique()]

    combined = pd.concat((combined, pd.get_dummies(combined[column], prefix = column)[good_cols]), axis = 1)

    del combined[column]
# 数据分割



del combined['PassengerId']



targets = train.Survived

train = combined[:891]

test = combined[891:]
train.shape
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectFromModel

clf = ExtraTreesClassifier(n_estimators=200)

clf = clf.fit(train, targets)



features = pd.DataFrame()

features['feature'] = train.columns

features['importance'] = clf.feature_importances_

features.sort_values(['importance'],ascending=False)
model = SelectFromModel(clf, prefit=True)

train_new = model.transform(train)

train_new.shape
test_new = model.transform(test)

test_new.shape
train.describe()
from sklearn.preprocessing import StandardScaler



ss = StandardScaler()

train = ss.fit_transform(train)

test = ss.transform(test)
train[0]
from sklearn.cross_validation import StratifiedKFold

forest = RandomForestClassifier(max_features='sqrt')



parameter_grid = {

                 'max_depth' : [4,5,6,7,8],

                 'n_estimators': [200,210,240,250],

                 'criterion': ['gini','entropy']

                 }



cross_validation = StratifiedKFold(targets, n_folds=5)



grid_search = GridSearchCV(forest,

                           param_grid=parameter_grid,

                           cv=cross_validation)



grid_search.fit(train_new, targets)



print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))
from sklearn.ensemble import RandomForestClassifier

 

rf = RandomForestClassifier(criterion='gini', 

                             n_estimators=700,

                             min_samples_split=16,

                             min_samples_leaf=1,

                             max_features='auto',

                             oob_score=True,

                             random_state=1,

                             n_jobs=-1) 



rf.fit(train, targets)

print(rf.oob_score_)
output = rf.predict(test).astype(int)

submit = pd.DataFrame()

submit['PassengerId'] = pd.read_csv('../input/test.csv')['PassengerId']

submit['Survived'] = output

submit[['PassengerId','Survived']].to_csv('submit.csv',index=False)