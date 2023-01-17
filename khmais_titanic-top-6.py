# run these instructions only one time

!pip install git+https://github.com/fastai/fastai@2e1ccb58121dc648751e2109fc0fbf6925aa8887

!apt update && apt install -y libsm6 libxext6
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os

path = '../input/'

print(os.listdir(path))
train_data= pd.read_csv(path+'train.csv')

print(train_data.columns)

train_data.describe()
train_data.info()
print("number of row = ",train_data.shape[0])
train_data['Died'] = 1-train_data['Survived']

train_data.groupby('Sex').agg('sum')[['Survived','Died']].plot(

    kind='bar', stacked=True, figsize=(10,6));
fig, axis = plt.subplots(3,2,figsize=(20,15))

sns.barplot(x="Survived", y="Fare", ax=axis[0,0], data=train_data)

sns.boxplot(x="Survived", y="Fare", ax=axis[0,1], data=train_data)

sns.boxplot(train_data['Fare'], ax=axis[1][0])

sns.barplot(train_data['Fare'], ax=axis[(1,1)])

sns.kdeplot(train_data.Fare, ax=axis[(2,0)])

sns.distplot(train_data.Fare, ax=axis[(2,1)]);
train_data.head()
fig, axis = plt.subplots(1,2,figsize=(15,8))

sns.barplot(x="Embarked", y="Survived", hue="Sex", ax=axis[(0)], data=train_data);

sns.barplot(x="Pclass", y="Survived", hue="Sex", ax=axis[(1)], data=train_data);

plt.figure(figsize=(10,5))

sns.barplot(x="Parch", y="Survived", hue="Sex", data=train_data);
plt.figure(figsize=(10,6))

sns.violinplot(x='Sex',y='Age',hue='Survived', data=train_data, split=True);
plt.figure(figsize=(15,10))

plt.hist([train_data[train_data['Survived'] == 1]['Fare'], train_data[train_data['Died'] == 1]['Fare']],

        stacked=True, color=['g','r'], bins=70, label = ['Survived','Died'])

plt.xlabel('Fare')

plt.ylabel('Number of passnegers')

plt.legend()

plt.grid()
plt.figure(figsize=(25,10))

ax=plt.subplot()

ax.scatter(train_data['Age'], train_data['Fare'], s=100)

plt.xlabel('Age')

plt.ylabel('Fare');
plt.figure(figsize=(25,10))

ax=plt.subplot()



ax.scatter(train_data[train_data['Survived'] == 1]['Age'], train_data[train_data['Survived'] == 1]['Fare'],

          c='green', s=train_data[train_data['Survived'] == 1]['Fare'])

ax.scatter(train_data[train_data['Died'] == 1]['Age'], train_data[train_data['Died'] == 1]['Fare'],

          c='red', s=train_data[train_data['Died'] == 1]['Fare']);

plt.xlabel('Age')

plt.ylabel('Fare');
plt.figure(figsize=(25,15))

sns.boxplot(x=train_data.Pclass, y=train_data.Fare);
sns.kdeplot(train_data[train_data.Pclass==1]['Fare']);
ax=plt.subplot()

ax.set_ylabel('Average fare')

train_data.groupby('Pclass').mean()['Fare'].plot(kind='bar', ax=ax, figsize=(10,6) );

#the line above is the same as :

#train_data.groupby('Pclass').agg({'Fare':'mean'}).plot(kind='bar', ax=ax)
X_train = train_data.drop(['Survived','Died'],axis=1)

y_train = train_data['Survived']

X_test = pd.read_csv(path+'/test.csv')

full_data = pd.concat([X_train,X_test], ignore_index=True)

# otherwise : full_data=X_train.append(X_test)

print(X_train.shape[1])

print(full_data.shape[1])
def process_family(df):

    # introducing a new feature : the size of families (including the passenger)

    df['FamilySize'] = df['Parch'] + df['SibSp'] + 1

    

    # introducing other features based on the family size

    df['Singleton'] = df['FamilySize'].map(lambda s: 1 if s == 1 else 0)

    df['SmallFamily'] = df['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)

    df['LargeFamily'] = df['FamilySize'].map(lambda s: 1 if 5 <= s else 0)    

    df.drop(['FamilySize', 'Parch', 'SibSp'], axis=1, inplace=True)

    return df
full_data = process_family(full_data)

full_data.head()
print(full_data.Embarked.describe())

print('we have',full_data.Embarked.isna().sum(),'missing values in Emabrked column, which they are :')

full_data.loc[full_data.Embarked.isna()]
full_data[(full_data.Pclass==1) & (full_data.Sex=='female')]['Embarked'].describe()
def process_embarked(df):

    # two missing embarked values - filling them with the most frequent one in the train  set(S)

    df.Embarked.fillna('C', inplace=True)

    # dummy encoding 

    df_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')

    df = pd.concat([df, df_dummies], axis=1)

    df.drop('Embarked', axis=1, inplace=True)

#     status('embarked')

    return df
full_data = process_embarked(full_data)

full_data.head()
print('we have',full_data.Cabin.isna().sum(),'missing value in Cabin column.')

full_data.Cabin.describe()
from sklearn.preprocessing import LabelEncoder
def process_cabin(df):

    # replacing missing cabins with U (for Uknown)

    df.Cabin.fillna('U', inplace=True)

    

    # mapping each Cabin value with the cabin letter

    df['Cabin'] = df['Cabin'].apply(lambda x: x[0])

    

    #Label Encoding ...

    df['Cabin']= LabelEncoder().fit_transform(df['Cabin'])

    

    # dummy encoding ...

    #cabin_dummies = pd.get_dummies(df['Cabin'], prefix='Cabin')    

    #df = pd.concat([df, cabin_dummies], axis=1)



    #df.drop('Cabin', axis=1, inplace=True)

    return df
full_data = process_cabin(full_data)

full_data.head(3)
titles = set()

for name in full_data['Name']:

    titles.add(name.split(',')[1].split('.')[0].strip())

titles
def process_title (df):

    df['Title'] = df['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())

    Title_Dict = {}

    Title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))

    Title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))

    Title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))

    Title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))

    Title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))

    Title_Dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))

    df['Title'] = df['Title'].map(Title_Dict)

    df.drop(['Name'], axis=1, inplace=True)

    return df
full_data = process_title(full_data)

full_data.head(3)
# Title_Dictionary = {

#      'Capt':'Officier',

#      'Col':'Officier',

#      'Don':'Royalty',

#      'Dona':'Royalty',

#      'Dr':'Officier',

#      'Jonkheer':'Royalty',

#      'Lady':'Royalty',

#      'Major':'Officier',

#      'Master':'Master',

#      'Miss':'Miss',

#      'Mlle':'Miss',

#      'Mme':'Mrs',

#      'Mr':'Mr',

#      'Mrs':'Mrs',

#      'Ms':'Mrs',

#      'Rev':'Officier',

#      'Sir':'Royalty',

#      'the Countess':'Royalty'   

# }

# def passenger_title(df):

#     df['Title'] = df['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())

#     df['Title'] = df['Title'].apply( lambda x : Title_Dictionary[x])

#     df.drop(['Name'], axis=1, inplace=True)

#     return df
# full_data = passenger_title(full_data)

# full_data.head()
# grouped_median_train = full_data.groupby(['Sex','Pclass','Title']).agg({'Age':'median'}).reset_index()

# grouped_median_train.head()
#Adding the value of age for missing values based on the grouped_median_train
# def fill_age(row):

#     condition = (

#         (grouped_median_train['Sex'] == row['Sex']) & 

#         (grouped_median_train['Title'] == row['Title']) & 

#         (grouped_median_train['Pclass'] == row['Pclass'])

#     ) 

#     return grouped_median_train[condition]['Age'].values[0]



# def process_age(df):

#     # a function that fills the missing values of the Age variable

#     df['Age'] = df.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)

#     return df
# full_data = process_age(full_data)

# full_data.head()
from sklearn.ensemble import RandomForestRegressor

age_df = full_data[['Age', 'Pclass','Sex','Title']]

age_df = pd.get_dummies(age_df)

age_df.head(3)
known_age_df = age_df[age_df.Age.notna()]

unknown_age_df = age_df[age_df.Age.isna()]

print(known_age_df.head(3))

unknown_age_df.head(3)
X = known_age_df.drop(['Age'], axis=1)

y = known_age_df['Age']
rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)

rfr.fit(X, y)
predicted_ages = rfr.predict(unknown_age_df.drop(['Age'], axis=1))
full_data.loc[full_data.Age.isna(), 'Age']= predicted_ages
full_data.Age.isna().sum()
full_data.isna().sum()
full_data[full_data.Fare.isna()]
full_data[(full_data.Pclass == 3) & (full_data.Sex == 'male') & (full_data.Singleton==1) & (full_data.Cabin == 8)].Fare.describe()
full_data['Fare'].fillna(full_data[(full_data.Pclass == 3) & (full_data.Sex == 'male') & (full_data.Singleton==1) & (full_data.Cabin == 8)].Fare.mean(), inplace= True)
full_data.isna().sum()
full_data['Title'].describe()
def encode_title(df):    

    #dummification Title column

    titles_dummies = pd.get_dummies(df['Title'], prefix='Title')

    df = pd.concat([df, titles_dummies], axis=1)

    #removing the title column since we have its dummies

    df.drop('Title', axis=1, inplace=True)

    

    #lebel Encoding Title column:

#   df['Title'] = LabelEncoder().fit_transform(df['Title'])

    return df
full_data = encode_title(full_data)

full_data.head()
def encode_sex(df):    

    #dummification Sex column

    Sex_dummies = pd.get_dummies(df['Sex'])

    df = pd.concat([df, Sex_dummies], axis=1)

    

    #removing the Sex column since we have its dummies

    df.drop('Sex', axis=1, inplace=True)

    return df
full_data = encode_sex(full_data)

full_data.head()
full_data.Ticket.describe()
Ticket_count = dict(full_data['Ticket'].value_counts())

full_data['Ticket_Count_Group'] = full_data['Ticket'].apply(lambda x : Ticket_count[x])

full_data.drop(['Ticket'], axis=1, inplace=True)

full_data.head()
full_data.Ticket_Count_Group.unique().size
enc = LabelEncoder()

full_data['Ticket_Count_Group'] = enc.fit_transform(full_data['Ticket_Count_Group'])

full_data.head()
print('the size of the train set is',train_data.shape)
train_data = pd.concat([full_data[:891], train_data['Survived']], axis=1)
train_data.head(10)
plt.figure(figsize=(10,8))

sns.heatmap(train_data.corr());
plt.figure(figsize=(20,18))

corr_matrix = train_data.corr()

mask = np.zeros_like(corr_matrix, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr_matrix, mask=mask,cmap=cmap, vmax=1 , vmin=0, center=0.5,

            square=True, linewidths=.5, cbar_kws={"shrink": .5});
X_train = full_data[:891]

X_test = full_data[891:]

print(X_train.shape)

print(X_test.shape)
def split_vals(df,num_sample_to_train): return df[:num_sample_to_train], df[num_sample_to_train:]

valid_count =60

n_trn = len(X_train)-valid_count

X_train1, X_valid1 = split_vals(X_train, n_trn)

y_train1, y_valid1 = split_vals(y_train, n_trn)
X_train1.shape,y_train1.shape,X_valid1.shape,y_valid1.shape
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV



rfc = RandomForestClassifier(n_estimators=180,

                             min_samples_leaf=3,

                             max_features=0.5,

                             n_jobs=-1)

rfc.fit(X_train1,y_train1)

rfc.score(X_train1,y_train1)
y_predict=rfc.predict(X_valid1)

metrics.accuracy_score(y_valid1,y_predict)
print(metrics.classification_report(y_valid1,y_predict))
print(metrics.confusion_matrix(y_valid1,y_predict))
from fastai.imports import *

from fastai.structured import *

from pandas_summary import DataFrameSummary

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier,GradientBoostingClassifier

from IPython.display import display

from sklearn import metrics

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

import seaborn as sns

import pylab as plot
fi = rf_feat_importance(rfc, X_train1);

fi
def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)

plot_fi(fi);
to_keep = fi[fi.imp>0.01].cols; 

print(len(to_keep))

to_keep
X_train = X_train[to_keep]

X_train.head()
rfc = RandomForestClassifier(n_estimators=150,min_samples_leaf=3,max_features=0.5,n_jobs=-1)

rfc.fit(X_train,y_train)

rfc.score(X_train,y_train)
X_test = X_test[to_keep]

X_test.Fare.fillna(14, inplace=True)

output=rfc.predict(X_test)
data_test = pd.read_csv(path+'/test.csv')

df_output = pd.DataFrame()

df_output['PassengerId'] = data_test['PassengerId']

df_output['Survived'] = output

df_output[['PassengerId','Survived']].to_csv('submission11.csv', index=False)
df_output.head(10)
X = X_train

y = train_data['Survived']
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectKBest



pipe=Pipeline([('select',SelectKBest(k='all')), 

               ('classify', RandomForestClassifier(random_state = 10, max_features = 'sqrt'))])



param_test = {'classify__n_estimators':list(range(20,50,2)), 

              'classify__max_depth':list(range(3,60,3))}

gsearch = GridSearchCV(estimator = pipe, param_grid = param_test, scoring='roc_auc', cv=10)

gsearch.fit(X,y)

print(gsearch.best_params_, gsearch.best_score_)
from sklearn.pipeline import make_pipeline

select = SelectKBest(k = 'all')

clf = RandomForestClassifier(random_state = 10, warm_start = True, 

                                  n_estimators = 26,

                                  max_depth = 6, 

                                  max_features = 'sqrt')

pipeline = make_pipeline(select, clf)

pipeline.fit(X, y)
from sklearn import metrics

from sklearn.model_selection import cross_val_score

cv_score = cross_val_score(pipeline, X, y, cv= 10)

print("CV Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_score), np.std(cv_score)))
predictions = pipeline.predict(X_test)

submission = pd.DataFrame({"PassengerId": pd.read_csv(path+'test.csv')['PassengerId'], "Survived": predictions.astype(np.int32)})

submission.to_csv("submission11.csv", index=False)

submission.info()
X_train  = full_data[:891]

y_train = train_data['Survived']

X_test = full_data[891:]
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV



clf = RandomForestClassifier()

parameters = {'n_estimators': [4, 6, 10, 12], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [2, 3, 5, 7, 10], 

              'min_samples_split': [2, 3, 5],

              'min_samples_leaf': [1,5,8]

             }



acc_scorer = make_scorer(accuracy_score)



grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)

grid_obj = grid_obj.fit(X_train, y_train)



clf = grid_obj.best_estimator_



clf.fit(X_train, y_train)
predictions = clf.predict(X_train)

print(accuracy_score(y_train, predictions))
predictions = clf.predict(X_test)

submission = pd.DataFrame({"PassengerId": pd.read_csv(path+'test.csv')['PassengerId'], "Survived": predictions.astype(np.int32)})

submission.to_csv("submission12.csv", index=False)

submission.info()
from sklearn.preprocessing import scale
X_train = scale(X_train)

X_test = scale(X_test)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(X_train, y_train)
y_predict=lr.predict(X_train)

metrics.accuracy_score(train_data['Survived'],y_predict)
predictions = lr.predict(X_test)

submission = pd.DataFrame({"PassengerId": pd.read_csv(path+'test.csv')['PassengerId'], "Survived": predictions.astype(np.int32)})

submission.to_csv("submission13.csv", index=False)

submission.info()
from sklearn.neighbors import KNeighborsClassifier
X_train = scale(full_data[:891].drop(['PassengerId'], axis=1))

y_train = train_data['Survived']
knn = KNeighborsClassifier()

knn.fit( X_train , y_train)
y_predicted = knn.predict(X_train)

metrics.accuracy_score(y_train, y_predicted)
X_test.shape, X_train.shape
predictions = knn.predict(np.delete(X_test, -1, axis=1))

submission = pd.DataFrame({"PassengerId": pd.read_csv(path+'test.csv')['PassengerId'], "Survived": predictions.astype(np.int32)})

submission.to_csv("submission14.csv", index=False)

submission.info()