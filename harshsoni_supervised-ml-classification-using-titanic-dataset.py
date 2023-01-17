import sys

print('Python: {}'.format(sys.version))

# numpy

import numpy

print('numpy: {}'.format(numpy.__version__))

# pandas

import pandas

print('pandas: {}'.format(pandas.__version__))

# matplotlib

import matplotlib

print('matplotlib: {}'.format(matplotlib.__version__))

# seaborn

import seaborn

print('seaborn: {}'.format(seaborn.__version__))

# scikit-learn

import sklearn

print('sklearn: {}'.format(sklearn.__version__))
%matplotlib inline



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import OrdinalEncoder

from sklearn.feature_selection import SelectKBest

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



import warnings

warnings.simplefilter(action='ignore')
sns.set(style='darkgrid')
train_path = '../input/train.csv'

test_path = '../input/test.csv'



train_df = pd.read_csv(train_path)

test_df = pd.read_csv(test_path)
train_df.shape
train_df.info()
train_df.describe()
train_df.head()
missing_data = train_df.isnull().sum()

percent_missing = round((missing_data / train_df.isnull().count())*100, 2)

missing_df = pd.concat(

    [missing_data.sort_values(ascending=False), percent_missing.sort_values(ascending=False)], 

    axis=1, keys=['Total', 'Percent']

)

missing_df.head(5)
train_df.columns
temp_df = train_df.copy()

temp_df['Cabin'] = temp_df['Cabin'].fillna('Unknown')



occ_cabins = temp_df['Cabin'].copy()

occ_cabins[occ_cabins != 'Unknown'] = 'Known'

temp_df['Cabin'] = occ_cabins



plt.figure(figsize=(5.5, 5))

sns.barplot(x='Cabin', y='Survived', data=temp_df)

sns.pointplot(x='Cabin', y='Survived', data=temp_df, color='k')
plt.figure(figsize=(12, 5))



plt.subplot(1, 2, 1)

sns.barplot(x='Sex', y='Survived', data=train_df)

plt.subplot(1, 2, 2)

sns.violinplot(x='Survived', y='Age', data=train_df, hue='Sex', split=True)
females = train_df[train_df['Sex'] == 'female']

males = train_df[train_df['Sex'] == 'male']



plt.figure(figsize=(12, 5))



plt.subplot(1, 2, 1)

ax = sns.distplot(males[males['Survived'] == 1]['Age'].dropna(), bins=10, kde=False, label='survived')

ax = sns.distplot(males[males['Survived'] == 0]['Age'].dropna(), bins=10, kde=False, label='not survived')

ax.legend()

ax.set_title('Male')



plt.subplot(1, 2, 2)

ax = sns.distplot(females[females['Survived'] == 1]['Age'].dropna(), kde=False, label='survived')

ax = sns.distplot(females[females['Survived'] == 0]['Age'].dropna(), kde=False, label='not survived')

ax.legend()

ax.set_title('Female')



plt.show()
plt.figure(figsize=(5.5, 5))

sns.boxplot(x='Survived', y='Age', data=train_df)
plt.figure(figsize=(5.5, 5))

sns.boxplot(x='Survived', y='Fare', data=train_df)
plt.figure(figsize=(5.5, 5))

sns.pointplot(x='Pclass', y='Survived', data=train_df)
plt.figure(figsize=(5.5, 5))

sns.barplot(x=(train_df['SibSp'] + train_df['Parch']), y=train_df['Survived'])
plt.figure(figsize=(5.5, 5))

sns.pointplot(x='Embarked', y='Survived', data=train_df)
pass_id = test_df['PassengerId']



X = train_df.drop(['PassengerId'], axis=1)

X_test = test_df.drop(['PassengerId'], axis=1)



y = X['Survived']

X = X.drop('Survived', axis=1)



features_train, features_valid, labels_train, labels_valid = train_test_split(

    X, y, test_size=0.2, random_state=7

)



pre_features_train = features_train.copy()



cols = features_train.columns

cols
missing_df
def missing_cabin(data):

    data['Cabin'].fillna('X01', inplace=True)



    

for data in [features_train, features_valid]:

    missing_cabin(data)
def missing_age(data):

    nan_ages = []

    

    mu = pre_features_train['Age'].mean()

    median = pre_features_train['Age'].median()

    sigma = pre_features_train['Age'].std()



    random_ages = np.random.randint(

            median-sigma, 

            median+sigma,

            data['Age'].isnull().sum()

        )



    nan_ages = data['Age'].copy() 

    nan_ages[nan_ages.isnull()] = random_ages

    data.loc[:, 'Age'] = nan_ages

    



for data in [features_train, features_valid]:

    missing_age(data)
impute_embark = SimpleImputer(strategy='most_frequent')



features_train['Embarked'] = impute_embark.fit_transform(features_train['Embarked'].values.reshape(-1, 1))

features_valid['Embarked'] = impute_embark.transform(features_valid['Embarked'].values.reshape(-1, 1))
features_train.info()
features_train.head()
cabin_data = np.array(features_train['Cabin'])

cabin_data_valid = np.array(features_valid['Cabin'])



cabin_data = pd.DataFrame([x[0] for x in cabin_data], index=features_train.index, columns=['Cabin'])

cabin_data_valid = pd.DataFrame([x[0] for x in cabin_data_valid], index=features_valid.index, columns=['Cabin'])



features_train.drop(['Cabin'], axis=1, inplace=True)

features_valid.drop(['Cabin'], axis=1, inplace=True)



features_train = pd.concat([features_train, cabin_data], axis=1)

features_valid = pd.concat([features_valid, cabin_data_valid], axis=1)



le_cabin = LabelEncoder()

features_train['Cabin'] = le_cabin.fit_transform(features_train['Cabin'])

features_valid['Cabin'] = le_cabin.transform(features_valid['Cabin'])
le_gender = LabelEncoder()

features_train['Sex'] = le_gender.fit_transform(features_train['Sex'])

features_valid['Sex'] = le_gender.transform(features_valid['Sex'])
oe_pclass = OrdinalEncoder(dtype='int64')

features_train['Pclass'] = oe_pclass.fit_transform(features_train['Pclass'].values.reshape(-1, 1))

features_valid['Pclass'] = oe_pclass.transform(features_valid['Pclass'].values.reshape(-1, 1))
le_embark = LabelEncoder()

integer_encoded = le_embark.fit_transform(features_train['Embarked'])

integer_encoded_valid = le_embark.transform(features_valid['Embarked'])



oh_embark = OneHotEncoder(handle_unknown='ignore', sparse='False', dtype='int64')

onehot_encoded = pd.DataFrame(

    oh_embark.fit_transform(integer_encoded.reshape(-1, 1)).toarray(), 

    columns=['Embarked_C', 'Embarked_Q', 'Embarked_S'], 

    index=features_train.index

)

onehot_encoded_valid = pd.DataFrame(

    oh_embark.transform(integer_encoded_valid.reshape(-1, 1)).toarray(), 

    columns=['Embarked_C', 'Embarked_Q', 'Embarked_S'], 

    index=features_valid.index

)



features_train = features_train.drop(['Embarked'], axis=1)

features_valid = features_valid.drop(['Embarked'], axis=1)



features_train = pd.concat([features_train, onehot_encoded], axis=1)

features_valid = pd.concat([features_valid, onehot_encoded_valid], axis=1)
features_train.head()
def scale_feature(data, feature):

    mu = pre_features_train[feature].mean()

    sigma = pre_features_train[feature].std()

    

    data.loc[:, feature] = round((data[feature] - mu) / sigma, 3)
for data in [features_train, features_valid]:

    scale_feature(data, 'Age')
for data in [features_train, features_valid]:

    scale_feature(data, 'Fare')
def add_rel(data):

    data.loc[:, 'relatives'] = data['SibSp'] + data['Parch']



for data in [features_train, features_valid]:

    add_rel(data)
def drop_features(data):

    drop_cols = ['Name', 'Ticket']

    data.drop(drop_cols, axis=1, inplace=True)

    

for data in [features_train, features_valid]:

    drop_features(data)
best_features = SelectKBest(k='all')

fit = best_features.fit(features_train, labels_train)



scores_df = pd.DataFrame(data=fit.scores_)

columns_df = pd.DataFrame(data=features_train.columns.values)



feature_scores_df = pd.concat([columns_df,scores_df],axis=1)

feature_scores_df.columns = ['Features','Score']



plt.figure(figsize=(5.5, 5))

sns.barplot(

    x='Score', y='Features', 

    order=feature_scores_df.nlargest(11,'Score')['Features'], 

    data=feature_scores_df, palette=sns.cubehelix_palette(n_colors=4, reverse=True)

)
def keep_best_features(data):

    drop_cols = ['relatives', 'SibSp', 'Embarked_Q']

    data.drop(drop_cols, axis=1, inplace=True)

    

for data in [features_train, features_valid]:

    keep_best_features(data)
features_train.head()
clf = [

    ('GNB', GaussianNB()),

    ('SVC', SVC(C=1000, kernel='rbf', gamma=0.3)),

    ('LReg', LogisticRegression(C=0.5, solver='lbfgs')),

    ('KNN', KNeighborsClassifier()),

    ('CART', DecisionTreeClassifier(min_samples_split=100)),

    ('BAG', BaggingClassifier()),

    ('RF', RandomForestClassifier(n_estimators=100)),

    ('AB', AdaBoostClassifier(n_estimators=100)),

    ('XGB', XGBClassifier(max_depth=10, learning_rate=0.03, n_estimators=100))

]



clf_names = [

    'GaussianNB', 'SVC', 'Logistic Reg', 'KNN', 'Decision Tree', 'Bagging', 'Random Forest', 'AdaBoost', 'XGboost'

]
result = []

for model in clf:

    score = cross_val_score(model[1], features_train, labels_train, cv=10)

    result.append(score)

    

result_df = pd.DataFrame({

    'Score' : [round(x.mean(), 3) for x in result],

    'Model' : clf_names

})



result_df = result_df.sort_values(by='Score', ascending=False)

result_df = result_df.set_index('Score')



result_df
plt.figure(figsize = (7, 4))

ax = sns.boxplot(data=pd.DataFrame(np.array(result).transpose(), columns=clf_names))

ax.set_xticklabels(ax.get_xticklabels(),rotation=30)

plt.show()
clf = XGBClassifier(

    max_depth=5, learning_rate=0.2,

    verbosity=1, silent=None, n_estimators=100,  

    objective='binary:logistic',booster='gbtree'

)



clf.fit(features_train, labels_train)

predict = clf.predict(features_valid)



print(round((accuracy_score(labels_valid, predict)*100), 2))
print(accuracy_score(labels_valid, predict))

print(confusion_matrix(labels_valid, predict))

print(classification_report(labels_valid, predict))
plt.figure(figsize=(4, 3))

ax = sns.heatmap(confusion_matrix(labels_valid, predict), annot=True)

ax.set_xlabel('Predicted Values')

ax.set_ylabel('True Values')



plt.show()
missing_cabin(X_test)

missing_age(X_test)

X_test['Embarked'] = impute_embark.transform(X_test['Embarked'].values.reshape(-1, 1))



X_test['Sex'] = le_gender.transform(X_test['Sex'])

X_test['Pclass'] = oe_pclass.transform(X_test['Pclass'].values.reshape(-1, 1))



cabin_data = np.array(X_test['Cabin'])

cabin_data = pd.DataFrame([x[0] for x in cabin_data], index=X_test.index, columns=['Cabin'])

X_test.drop(['Cabin'], axis=1, inplace=True)

X_test = pd.concat([X_test, cabin_data], axis=1)

X_test['Cabin'] = le_cabin.transform(X_test['Cabin'])



integer_encoded_test = le_embark.transform(X_test['Embarked'])

onehot_encoded_test = pd.DataFrame(

    oh_embark.transform(integer_encoded_test.reshape(-1, 1)).toarray(), 

    columns=['Embarked_C', 'Embarked_Q', 'Embarked_S'], 

    index=X_test.index

)



X_test.drop(['Embarked'], axis=1, inplace=True)

X_test = pd.concat([X_test, onehot_encoded_test], axis=1)

    

scale_feature(X_test, 'Age')

scale_feature(X_test, 'Fare')

add_rel(X_test)



drop_features(X_test)

keep_best_features(X_test)



predict = clf.predict(X_test)
output = pd.DataFrame(

    {

        'PassengerId': pass_id,

        'Survived': predict

    }

)



output.to_csv('gender_submission.csv', index=False)