import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train.shape, test.shape
train.head(10)
null_count = train.append(test, sort=True).isnull().sum()

null_count[null_count > 0]
train['Survived'].value_counts(normalize=True)
f, ax = plt.subplots(1,2,figsize=(10,4))

train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Survived')

ax[0].set_ylabel('')

sns.countplot('Survived',data=train,ax=ax[1])

ax[1].set_title('Survived')

plt.show()
train.groupby(['Pclass'])['Survived'].agg(['mean', 'count'])
print('unique Cabin count:', train['Cabin'].nunique())

train['Cabin'].unique()[:30]
train['Cabin'].str[0].unique()
fare_median = train['Fare'].median()

fare_median
embarked_mode = train['Embarked'].mode()[0]

embarked_mode
fill_dict = {'Fare': fare_median, 'Embarked': embarked_mode, 'Cabin': 'Z'}

train.fillna(fill_dict, inplace=True)

test.fillna(fill_dict, inplace=True)
dataset = train.append(test, sort=True)

dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)

pd.crosstab(dataset.Title, dataset.Sex).join(dataset.groupby('Title').mean()['Age'])
def put_title(data):

    data['Title'] = data.Name.str.extract('([A-Za-z]+)\.', expand=False)

    title_map = {

        'Mlle':'Miss', 

        'Mme':'Miss', 

        'Ms':'Miss',

        'Dr':'Mr', 

        'Major':'Mr', 

        'Capt':'Mr', 

        'Sir':'Mr', 

        'Don':'Mr',

        'Lady':'Mrs', 

        'Countess':'Mrs', 

        'Dona':'Mrs',

        'Jonkheer':'Other', 

        'Col':'Other', 

        'Rev':'Other'}

    data['Title'].replace(title_map, inplace=True)



put_title(train)

put_title(test)
age_mean = train.append(test, sort=True).groupby('Title')['Age'].mean()

age_mean
for i in age_mean.index:

    train.loc[train.Age.isnull() & (train.Title == i), 'Age'] = age_mean[i]

    test.loc[test.Age.isnull() & (test.Title == i), 'Age'] = age_mean[i]
pd.crosstab(train.Cabin.str[0], train.Survived, margins=True)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
def transDataFrame(data):

    data = data.copy()

    data.Embarked = data.Embarked.replace(['C', 'S', 'Q'], [0, 1, 2])

    data.Sex = data.Sex.replace(['male', 'female'], [0, 1])

    data.Cabin = le.fit_transform(data.Cabin.str[0])

    data['Title_Code'] = le.fit_transform(data['Title'])

    

    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    data['IsAlone'] = 1

    data.loc[(data.SibSp + data.Parch) > 1, 'IsAlone'] = 0

    

    data['Fare_Bin'] = pd.qcut(data['Fare'], 4)

    data['Fare_Code'] = le.fit_transform(data['Fare_Bin'])

    data['Age_Bin'] = pd.cut(data['Age'].astype(int), 5)

    data['Age_Code'] = le.fit_transform(data['Age_Bin'])

    data.drop(['Fare_Bin', 'Age_Bin'], axis=1, inplace=True)

    

    data.drop(['Name','Title','Ticket','Parch','SibSp','Age','Fare'], axis=1, inplace=True)

    return data
train_1 = transDataFrame(train).drop('PassengerId', axis=1)

train_1.describe()
corr = train_1.corr()



plt.figure(figsize=(15,6))

plt.title('Correlation of Features for Train Set')

sns.heatmap(corr, vmax=1.0, annot=True, cmap='coolwarm')

plt.show()
pd.DataFrame(corr.Survived.abs().sort_values(ascending=False)).T
for x in ['Sex', 'Pclass', 'Cabin', 'Embarked', 'IsAlone', 'Title_Code', 'FamilySize', 'Fare_Code', 'Age_Code']:

    print('Survival Correlation by:', x)

    print(train_1[[x, 'Survived']].groupby(x, as_index=False).mean())

    print('-'*10, '\n')
a = sns.pairplot(train_1[[u'Survived', u'Pclass', u'Sex', u'Age_Code', u'IsAlone', u'Fare_Code', u'Embarked', u'FamilySize', u'Title_Code']], 

                 hue='Survived', size=1.3, palette='seismic')

a.set(xticklabels=[])
#pd.crosstab(train_1.Age // 5 * 5, train_1.Survived).plot.area(stacked=False)

a = sns.FacetGrid(train, hue='Survived', aspect=4)

a.map(sns.kdeplot, 'Age', shade=True)

a.set(xlim=(0 , train['Age'].max()))

a.add_legend()
_train = transDataFrame(train)

_train.drop(['PassengerId'], axis=1, inplace=True)

_train.head()
_train.isnull().any()
_test = transDataFrame(test)

_test.head()
_test.isnull().any()
_train.head()
X_train = _train.drop("Survived", axis=1)

Y_train = _train["Survived"]

X_test  = _test.drop("PassengerId", axis=1).copy()



X_train.shape, Y_train.shape, X_test.shape
import lightgbm as lgb



params={'learning_rate': 0.05,

        'objective':'binary',

        'metric':'auc',

        'num_leaves': 31,

        'verbose': 1,

        'random_state':42,

        'bagging_fraction': 0.7,

        'feature_fraction': 0.7

       }



clf = lgb.LGBMClassifier(**params, n_estimators=1000)

clf.fit(X_train, Y_train)

result = clf.predict(X_test, num_iteration=clf.best_iteration_)
import shap



explainer = shap.TreeExplainer(model=clf, feature_dependence='tree_path_dependent', model_output='margin')

shap_values = explainer.shap_values(X=X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar")

shap.summary_plot(shap_values, X_train)

#shap.force_plot(base_value=explainer.expected_value, shap_values=shap_values, features=X_train)

#shap.force_plot(explainer.expected_value, shap_values[0,:], X_train.iloc[0,:])
submission = pd.DataFrame({

    "PassengerId": test["PassengerId"],

    "Survived": result

})

submission.to_csv("submission.csv", index=False)
submission.head()