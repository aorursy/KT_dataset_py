import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#Data Visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("white")
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from category_encoders import CatBoostEncoder
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")
train.head()
print('Concise summary of training data:\n')

print(train.info())

print('-'*50,'\nConcise summary of testing data:\n')

print(test.info())
#dropping PassengerId as it provides no purpose.

train.drop('PassengerId',1,inplace=True)

PID = test.PassengerId

test.drop('PassengerId',1,inplace=True)

#We will work through rest of the columns.
# categorical data

train.select_dtypes("object").columns
print("Survival Percentage in Titanic:")

print(round(train.Survived.sum()*100/len(train),2))

sns.countplot(y=train.Survived, orient='h', palette='Accent')
sns.countplot(train.Pclass, hue=train.Survived)
#let's extract class 3 and drop the column Pclass.

train['isPclass3'] = train['Pclass']==3

test['isPclass3'] = test['Pclass']==3
print(train.Name.sample(10))

print('\nTotal no of unique names:',train.Name.nunique())
#Extracting title from name

title = train.Name.str.extract('([A-Za-z]+)\.')

ttitle = test.Name.str.extract('([A-Za-z]+)\.')

plt.xticks(rotation=90)

sns.countplot(title[0], palette='tab20', hue=train.Survived)

plt.xlabel('Title')
title = title[0].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'Countess', 'Dona'], 'Miss/Mrs/Ms')

ttitle = ttitle[0].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'Countess', 'Dona'], 'Miss/Mrs/Ms')



title = title.replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Misc')

ttitle = ttitle.replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Misc')

train = train.join(title).rename(columns = {0:'Title'}) #append the generated feature to our dataset

test = test.join(ttitle).rename(columns = {0:'Title'}) #append the generated feature to our dataset
plt.xticks(rotation=90)

sns.countplot('Title',data=train,hue='Survived', palette='Set2')
sns.countplot(train.Sex, hue=train.Survived)
train['isMale'] = train.Sex=='male'

test['isMale'] = test.Sex=='male'
Ticket = train.Ticket.str.strip().str[0]

Ticket_test = test.Ticket.str.strip().str[0]
plt.figure(figsize=(14,6))



sns.countplot('Ticket',data=train.drop('Ticket',1).join(Ticket),hue='Survived', palette="Accent")
train = train.drop('Ticket',1).join(Ticket).rename(columns={'Ticket':'Ticket'})

test = test.drop('Ticket',1).join(Ticket_test).rename(columns={'Ticket':'Ticket'})
train['Ticket_preferred']=train.Ticket.apply(lambda s: s in ['3','A','S','C','7','W','4'])

test['Ticket_preferred']=test.Ticket.apply(lambda s: s in ['3','A','S','C','7','W','4'])
print("Total no of null values in Cabin column:",train.Cabin.isnull().sum())
Cabin = train.Cabin.str[0]

Cabin_test = test.Cabin.str[0]

Cabin.value_counts()
#let's append it to our dataset

train = train.drop('Cabin',1).join(Cabin)

test = test.drop('Cabin',1).join(Cabin_test)
sns.countplot(train.Cabin, hue=train.Survived)
train['Cabin_preferred']=train.Cabin.apply(lambda s: s in ['C','E','D','B','F'])

test['Cabin_preferred']=test.Cabin.apply(lambda s: s in ['C','E','D','B','F'])
sns.countplot(train.Embarked, hue=train.Survived, palette='seismic')
print('Total no of missing values in Embarked:',train.Embarked.isna().sum())
train.Embarked.fillna('S',inplace=True)
train['ifEmbarkedS'] = train.Embarked =='S'

test['ifEmbarkedS'] = test.Embarked =='S'
train.drop('Pclass',1, inplace = True)

test.drop('Pclass',1, inplace = True)



train.drop('Name',1, inplace = True)

test.drop('Name',1, inplace = True)



train.drop('Title',1, inplace = True)

test.drop('Title',1, inplace = True)



train.drop('Sex',1, inplace = True)

test.drop('Sex',1, inplace = True)



train = train.drop('Ticket',1)

test = test.drop('Ticket',1)



train = train.drop('Cabin',1)

test = test.drop('Cabin',1)



train = train.drop('Embarked',1)

test = test.drop('Embarked',1)
#descriptive statistics of numerical data

train.drop(['Survived'],1).describe()
print('Total no of missing values in Age:',train.Age.isna().sum())
train.loc[:,'Age'] = train.loc[:,'Age'].fillna(train.groupby(['isMale','Ticket_preferred'])['Age'].transform('median'))

train.loc[:,'Age'] = train.loc[:,'Age'].fillna(train['Age'].median())
test.loc[:,'Age'] = test.loc[:,'Age'].fillna(test.groupby(['isMale','Ticket_preferred'])['Age'].transform('median'))

test.loc[:,'Age'] = test.loc[:,'Age'].fillna(test['Age'].median())
sns.distplot(train[train.Survived==0].Age,color='r')

sns.distplot(train[train.Survived==1].Age,color='g')
train[['isPclass3','Age']].groupby('isPclass3').mean()
train['Family_size'] = train['SibSp'] + train['Parch'] + 1

test['Family_size'] = test['SibSp'] + test['Parch'] + 1
sns.distplot(train[train.Survived==0].Family_size,color='r')

sns.distplot(train[train.Survived==1].Family_size,color='g')
train['isAlone']= train.Family_size==1

test['isAlone']= test.Family_size==1
#let's drop sibsp, parch and family_size as their work is done.

train.drop(['SibSp','Parch','Family_size'],1,inplace=True)

test.drop(['SibSp','Parch','Family_size'],1,inplace=True)
sns.boxplot(train.Fare, hue=train.Survived)
train[train.Fare>400].Fare = np.nan

train.loc[:,'Fare'] = train.loc[:,'Fare'].fillna(train.groupby(['isMale','Ticket_preferred'])['Fare'].transform('mean'))

test.loc[:,'Fare'] = test.loc[:,'Fare'].fillna(test.groupby(['isMale','Ticket_preferred'])['Fare'].transform('mean'))
fig, ax = plt.subplots(1,2, figsize=(16,6))

sns.distplot(train[train.Survived==0].Fare,color='r', bins=50, ax=ax[0])

sns.distplot(train[train.Survived==1].Fare,color='g', bins=100, ax=ax[0])

ax[0].axis(xmin=-10,xmax=55)



sns.distplot(train[train.Survived==0].Fare,color='r', bins=75, ax=ax[1])

sns.distplot(train[train.Survived==1].Fare,color='g', bins=100, ax=ax[1])

ax[1].axis(xmin=45,xmax=150)
#binning age

train=train.drop('Age',1).join(pd.cut(train.Age, range(0,81,10), True, range(8), ).astype('int64')).rename(columns={'Age':'Age_bin'})

test=test.drop('Age',1).join(pd.cut(test.Age, range(0,81,10), True, range(8), ).astype('int64')).rename(columns={'Age':'Age_bin'})
n=9

sns.countplot(pd.qcut(train.Fare, n, range(n)).astype('int64'), hue=train.Survived)
#binning Fare

train.Fare, bins = pd.qcut(train.Fare, n, range(n), True)

test.Fare = pd.cut(test.Fare, bins, True, range(n))
train.Fare = train.Fare.astype('int64')

test.Fare = test.Fare.fillna(method='bfill').astype('int64')
# Custom Label Encoder for handling unknown values

class LabelEncoderExt(object):

    def __init__(self):

        self.label_encoder = LabelEncoder()



    def fit(self, data):

        self.label_encoder = self.label_encoder.fit(list(data) + ['Unknown'])

        self.classes_ = self.label_encoder.classes_

        return self



    def transform(self, data):

        new_data = list(data)

        for unique_item in np.unique(data):

            if unique_item not in self.label_encoder.classes_:

                new_data = ['Unknown' if x==unique_item else x for x in new_data]

        return self.label_encoder.transform(new_data)
# from itertools import combinations



# object_cols = train.select_dtypes("object").columns

# object_cols_test = test.select_dtypes("object").columns



# low_cardinality_cols = [col for col in object_cols if train[col].nunique() < 15]



# interactions = pd.DataFrame(index=train.index)

# interactions_test = pd.DataFrame(index=test.index)



# # Iterate through each pair of features, combine them into interaction features

# for features in combinations(low_cardinality_cols,2):

    

#     new_interaction = train[features[0]].map(str)+"_"+train[features[1]].map(str)

#     new_interaction_test = test[features[0]].map(str)+"_"+test[features[1]].map(str)

    

#     encoder = LabelEncoderExt()

#     encoder.fit(new_interaction)

#     interactions["_".join(features)] = encoder.transform(new_interaction)

#     interactions_test["_".join(features)] = encoder.transform(new_interaction_test)
# train = train.join(interactions) #append to the dataset

# test = test.join(interactions_test) #append to the dataset
train.info()
test.info()
print(train.isna().sum(), test.isna().sum())
X_train = train.drop('Survived',1)

y_train = train.Survived

X_test = test
seed = 42
sampler = SMOTE(random_state = seed)

X_train, y_train = sampler.fit_resample(X_train, y_train)
y_train.value_counts()
print(X_train.shape,X_test.shape)
# pcorr = X_train.corrwith(y_train)

# imp_corr_cols = pcorr[(pcorr>0.1) | (pcorr<-0.1)].index



# X_train = X_train[imp_corr_cols]

# X_test = X_test[imp_corr_cols]
from numpy import mean

from numpy import std



from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn.metrics import classification_report, confusion_matrix



from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, StackingClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from sklearn.model_selection import GridSearchCV
models = {

    'LR':make_pipeline(StandardScaler(),LogisticRegression(random_state=seed)),

    'SVC':make_pipeline(StandardScaler(),SVC(random_state=seed)),

    'AB':AdaBoostClassifier(random_state=seed),

    'ET':ExtraTreesClassifier(random_state=seed),

    'GB':GradientBoostingClassifier(random_state=seed),

    'RF':RandomForestClassifier(random_state=seed),

    'XGB':XGBClassifier(random_state=seed),

    'LGBM':LGBMClassifier(random_state=seed)

    }
# evaluate a give model using cross-validation

def evaluate_model(model):

    cv = StratifiedKFold(shuffle=True, random_state=seed)

    scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

    return scores
# evaluate the models and store results

results, names = list(), list()

for name, model in models.items():

    scores = evaluate_model(model)

    results.append(scores)

    names.append(name)

    print('*%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
plt.boxplot(results, labels=names, showmeans=True)

plt.show()
feat_imp=[]

for name, model in models.items():

    model.fit(X_train, y_train)

    if name not in ['LR', 'SVC', 'KNN']: #since they do not have feature importance

        feat_imp.append(pd.Series(model.feature_importances_, index=X_train.columns))
feat_imp[-1]=feat_imp[-1].apply(lambda x: x/1000)

avg_feat_imp = pd.DataFrame(feat_imp).mean()

plt.figure(figsize=(16,6))

plt.xticks(rotation=90)

plt.xlabel('Average Feature Importance')

plt.plot(avg_feat_imp.sort_values(ascending=False))
# impcols = avg_feat_imp.sort_values(ascending=False).index[:15]

# X_train = X_train[impcols]

# X_test = X_test[impcols]
models = {

    'SVC':make_pipeline(StandardScaler(),SVC(random_state=seed)),

    'AB':AdaBoostClassifier(random_state=seed),

    'ET':ExtraTreesClassifier(n_jobs=-1, random_state=seed),

    'GB':GradientBoostingClassifier(random_state=seed),

    'RF':RandomForestClassifier(n_jobs=-1, random_state=seed),

    'XGB':XGBClassifier(n_jobs=-1, random_state=seed),

    'LGBM':LGBMClassifier(n_jobs=-1, random_state=seed)

    }
sns.heatmap(X_train.join(y_train).corr().apply(abs), cmap = 'bwr')
for name, model in models.items():

    print(name,'parameters:')

    print(model.get_params())

    print('='*140)
params = {

    'SVC':{'svc__gamma':[0.01,0.02,0.05,0.08,0.1], 'svc__C':range(1,8)},

    

    'AB':{'learning_rate': [0.05, 0.1, 0.2, 0.5], 'n_estimators': range(50,501,100)},

    

    'ET':{'max_depth':[5,8,10,12], 'min_samples_split': [5,8,10,12],

          'n_estimators': [500,1000,1500,2000]},

    

    'GB':{'learning_rate': [0.1, 0.2, 0.5], 'max_depth':[3,5,8,10],

          'min_samples_split': [5,8,10,12], 'n_estimators': [50,100,200,500],

          'subsample':[0.5,0.7,0.9]},

    

    'RF':{'max_depth':[3,5,10,12,15], 'n_estimators': [50,100,500,1000],

          'min_samples_split': [4,8,10]},

    

    'XGB':{'max_depth':range(3,10,2), 'n_estimators': range(50,201,50),

           'learning_rate': [0.05, 0.08, 0.1, 0.15], 'subsample':[0.5,0.7,0.9]},

    

    'LGBM':{'max_depth':range(3,10,2), 'n_estimators': range(50,201,50),

            'learning_rate': [0.05, 0.08, 0.1, 0.15, 0.2],'subsample':[0.5,0.7,0.9],

           'num_leaves': range(15,51,10)}

}
# evaluate the models and store results

best_params = params

names= list()

for name, param_grid, model in zip(params.keys(), params.values(), models.values()):

    gscv = GridSearchCV(model, param_grid, n_jobs=-1, verbose=3, cv=4)

    gscv.fit(X_train,y_train)

    names.append(name)

    best_params[name] = gscv.best_params_

    print(name)

    print("best score:",gscv.best_score_)

    print("best params:",gscv.best_params_)
base_models = [

    ('SVC',make_pipeline(StandardScaler(),SVC(random_state=seed))),

    ('AB',AdaBoostClassifier(random_state=seed)),

    ('ET',ExtraTreesClassifier(random_state=seed)),

    ('GB',GradientBoostingClassifier(random_state=seed)),

    ('RF',RandomForestClassifier(random_state=seed)),

    ('XGB',XGBClassifier(random_state=seed)),

    ('LGBM',LGBMClassifier(random_state=seed))

]
for model, param in zip(base_models,best_params.values()):

    model[1].set_params(**param)
clf = StackingClassifier(estimators=base_models)
score = evaluate_model(clf)
print(score)

print(mean(score))
clf.fit(X_train,y_train)
y_preds = clf.predict(X_test)
Submission = pd.DataFrame({ 'PassengerId': PID,'Survived': y_preds })

Submission.to_csv('Submission.csv', index = False)