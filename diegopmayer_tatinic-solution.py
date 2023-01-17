# !pip install scikit-optimize

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





 

import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

%pylab inline



from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier



from sklearn.linear_model import LogisticRegression

from lightgbm import LGBMClassifier

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB



from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC



from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score

from sklearn.feature_selection import RFE

from skopt import dummy_minimize, plots



from sklearn.metrics import classification_report, f1_score, roc_auc_score

from sklearn.metrics import log_loss



import warnings

warnings.filterwarnings('ignore')

pd.options.display.max_columns = None
ls
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/train.csv')
col_obj = df_train.columns[df_train.dtypes == object]

col_num = df_train.columns[df_train.dtypes != object]
df_train[col_num].head()
df_train[col_obj].head()
print(f'Shape train -->> {df_train.shape}')

print(f'Null columns--------------\n{df_train.isnull().sum()}')



# Age      = 177 null values

# Cabin    = 687 null values

# Embarked = 2 null value
print(f'{df_train.Pclass.value_counts()}\n{df_train.Pclass.value_counts() / df_train.shape[0]}')
sns.countplot(x='Pclass', data=df_train, hue='Survived')
plt.figure(figsize=(15,10))

sns.countplot(x='Age', data=df_train, hue='Pclass')
print(f'Mean Age from Class_3: {df_train.Age[df_train.Pclass == 3].median()}')

print(f'Mean Age from Class_2: {df_train.Age[df_train.Pclass == 2].median()}')

print(f'Mean Age from Class_1: {df_train.Age[df_train.Pclass == 1].median()}')
print(f'Clas_3 fem - sur - count: 47 - med: {df_train.Age[(df_train.Pclass == 3) & (df_train.Sex == "female") & (df_train.Survived == 1)].median()}') # 47

print(f'Clas_2 fem - sur - count: 68 - med: {df_train.Age[(df_train.Pclass == 2) & (df_train.Sex == "female") & (df_train.Survived == 1)].median()}') # 68

print(f'Clas_1 fem - sur - count: 82 - med: {df_train.Age[(df_train.Pclass == 1) & (df_train.Sex == "female") & (df_train.Survived == 1)].median()}') # 82

print(f'Clas_3 fem - die - count: 55 - med: {df_train.Age[(df_train.Pclass == 3) & (df_train.Sex == "female") & (df_train.Survived == 0)].median()}') # 55

print(f'Clas_2 fem - die - count:  6 - med: {df_train.Age[(df_train.Pclass == 2) & (df_train.Sex == "female") & (df_train.Survived == 0)].median()}') # 6

print(f'Clas_1 fem - die - count:  3 - med: {df_train.Age[(df_train.Pclass == 1) & (df_train.Sex == "female") & (df_train.Survived == 0)].median()}') # 3
df_train.Age[(df_train.Pclass == 3) & (df_train.Sex == 'female')].mean(), df_train.Age[(df_train.Pclass == 3) & (df_train.Sex == 'female')].median()
df_train.Age[(df_train.Pclass == 3) & (df_train.Sex == 'female') & (df_train.Survived == 1)].hist(bins=25)

df_train.Age[(df_train.Pclass == 3) & (df_train.Sex == 'female') & (df_train.Survived == 0)].hist(bins=25, color='red')
df_train.Age[(df_train.Pclass == 2) & (df_train.Sex == 'female')].mean(), df_train.Age[(df_train.Pclass == 2) & (df_train.Sex == 'female')].median()
df_train.Age[(df_train.Pclass == 2) & (df_train.Sex == 'female') & (df_train.Survived == 1)].hist(bins=25)

df_train.Age[(df_train.Pclass == 2) & (df_train.Sex == 'female') & (df_train.Survived == 0)].hist(bins=25, color='red')
df_train.Age[(df_train.Pclass == 1) & (df_train.Sex == 'female')].mean(), df_train.Age[(df_train.Pclass == 1) & (df_train.Sex == 'female')].median()
df_train.Age[(df_train.Pclass == 1) & (df_train.Sex == 'female') & (df_train.Survived == 1)].hist(bins=25)

df_train.Age[(df_train.Pclass == 1) & (df_train.Sex == 'female') & (df_train.Survived == 0)].hist(bins=25, color='red')
df_train.Age[(df_train.Pclass == 3) & (df_train.Sex == 'male')].mean(), df_train.Age[(df_train.Pclass == 3) & (df_train.Sex == 'male')].median()
df_train.Age[(df_train.Pclass == 3) & (df_train.Sex == 'male') & (df_train.Survived == 0)].hist(bins=25, color='red')

df_train.Age[(df_train.Pclass == 3) & (df_train.Sex == 'male') & (df_train.Survived == 1)].hist(bins=25)
df_train.Age[(df_train.Pclass == 2) & (df_train.Sex == 'male')].mean(), df_train.Age[(df_train.Pclass == 2) & (df_train.Sex == 'male')].median()
df_train.Age[(df_train.Pclass == 2) & (df_train.Sex == 'male') & (df_train.Survived == 0)].hist(bins=25, color='red')

df_train.Age[(df_train.Pclass == 2) & (df_train.Sex == 'male') & (df_train.Survived == 1)].hist(bins=25)
df_train.Age[(df_train.Pclass == 1) & (df_train.Sex == 'male')].mean(), df_train.Age[(df_train.Pclass == 1) & (df_train.Sex == 'male')].median()
df_train.Age[(df_train.Pclass == 1) & (df_train.Sex == 'male') & (df_train.Survived == 0)].hist(bins=25, color='red')

df_train.Age[(df_train.Pclass == 1) & (df_train.Sex == 'male') & (df_train.Survived == 1)].hist(bins=25)
df_train.Age[(df_train.Pclass == 3) & (df_train.Sex == 'male') & (df_train.Survived == 0)].hist(bins=25)

df_train.Age[(df_train.Pclass == 2) & (df_train.Sex == 'male') & (df_train.Survived == 0)].hist(bins=25)

df_train.Age[(df_train.Pclass == 1) & (df_train.Sex == 'male') & (df_train.Survived == 0)].hist(bins=25)
class_3_died = df_train[(df_train.Pclass == 3) & (df_train.Survived == 0)]

class_2_died = df_train[(df_train.Pclass == 2) & (df_train.Survived == 0)]

class_1_died = df_train[(df_train.Pclass == 1) & (df_train.Survived == 0)]

class_3_surv = df_train[(df_train.Pclass == 3) & (df_train.Survived == 1)]

class_2_surv = df_train[(df_train.Pclass == 2) & (df_train.Survived == 1)]

class_1_surv = df_train[(df_train.Pclass == 1) & (df_train.Survived == 1)]
print(f""" 

Died  Class 3 - Mean: {round(class_3_died.Age.mean(), ndigits=2)}   Median: {class_3_died.Age.median()}

Died  Class 2 - Mean: {round(class_2_died.Age.mean(), ndigits=2)}   Median: {class_2_died.Age.median()}

Died  Class 1 - Mean: {round(class_1_died.Age.mean(), ndigits=2)}    Median: {class_1_died.Age.median()}

Lived Class 3 - Mean: {round(class_3_surv.Age.mean(), ndigits=2)}   Median: {class_3_surv.Age.median()}

Lived Class 2 - Mean: {round(class_2_surv.Age.mean(), ndigits=2)}    Median: {class_2_surv.Age.median()}

Lived Class 1 - Mean: {round(class_1_surv.Age.mean(), ndigits=2)}   Median: {class_1_surv.Age.median()}

""")
class_1_died.Age.fillna(45).hist(bins=25), class_1_surv.fillna(35).Age.hist(bins=25)

# < 4

# > 10 and < 42

class_3_died.fillna(25).Age.hist(bins=25), class_3_surv.fillna(22).Age.hist(bins=25)

# < 5

# > 18 and < 32

class_2_died.fillna(30).Age.hist(bins=25), class_2_surv.fillna(28).Age.hist(bins=25)

# < 10

# > 18 and < 25
df_train.SibSp.hist(), df_train.Parch.hist()
#sns.countplot(x='SibSp', data=df_train, hue='Survived')
#sns.countplot(x='Parch', data=df_train, hue='Survived')
df_train.Fare.value_counts()

df_train.Fare.mean(), df_train.Fare.median()
df_train.Fare[df_train.Fare <300].hist(bins=25)
#df_train.Fare[df_train.Fare > 100].hist()

df_train.Fare[df_train.Fare < 100].hist(bins=25)
df_train.Fare.describe()

# Q3 - Q1 = FIQ

# 31 - 7.91 = 23.09

# Q3 +(FIQ*1.5) = > 65.63 there are many outliers

# Q1 -(FIQ*1.5) = < -26.72 no outlier
df_train.Fare.plot(kind='box')
print(df_train.Embarked.value_counts())

print('NaN: ', df_train.Embarked.isnull().sum())
surname = df_train.Name.str.extract(r'(\w+)')

surname.columns = ['Surname']
df = df_train.copy()

df = pd.concat([df, surname], axis=1).copy()



mapping = df.Surname.value_counts()



df.Surname = df.Surname.map(mapping).copy()

df.head()
surname = ['Mr.', 'Mrs', 'Miss', 'Master', 'Rev.', 'Dr', 'Col.', 'Sir.', 'Major', 'Don.', 'Capt']

for rep in surname:

    for name in df['Name'][df.Name.str.contains(rep, regex=False)]:

        if rep in ['Col.', 'Major', 'Capt']:

            df.replace(name, 'Army', inplace=True)

        elif rep in ['Sir.', 'Don.']:

            df.replace(name, 'Mr.', inplace=True)

        else:

            df.replace(name, rep, inplace=True)



for title in df.Name:

    if title not in ['Mr.', 'Miss', 'Mrs', 'Master', 'Dr', 'Rev', 'Army']:

        df.replace(title, 'Miss', inplace=True)
sns.countplot('Surname', hue='Survived', data=df)
sns.countplot('Name', hue='Survived', data=df)
sns.catplot(x='Surname', y='Surname', hue='Survived', data=df)
import re
cab = pd.DataFrame()
cab_ = df_train.Cabin.str.extract((r'([A-Za-z])'), ).fillna(0).copy()

cab = pd.concat([cab, cab_], axis=1).copy()

cab.columns = ['cabin']
cab.cabin.value_counts()
sns.countplot('cabin', data=cab[cab.cabin != 0])
df_train.Ticket.head()
df_train.Ticket.str.extract(r'([A-Za-z])').head()
df_train[df_train.Ticket == '1601']
mapping_ticket = df_train['Ticket'].value_counts().copy()

ticket = df_train['Ticket'].copy()

ticket = ticket.map(mapping_ticket)
ticket.head(10)
train = pd.DataFrame()

# Create new variable for Data Preparation

train['Survived'] = df_train['Survived'].copy()

train['Pclass'] = df_train['Pclass'].copy()

train['Sex'] = df_train.Sex.map({

    'female':1,

    'male':0

}).copy()

#--------------------------------------------- get mean to NaN in Age

train['Age'] = df_train['Age'].copy()

med_cl_fem_3 = train.Age[(train.Pclass == 3) & (train.Sex == 1)].median()

med_cl_fem_2 = train.Age[(train.Pclass == 2) & (train.Sex == 1)].median()

med_cl_fem_1 = train.Age[(train.Pclass == 1) & (train.Sex == 1)].median()

med_cl_mal_3 = train.Age[(train.Pclass == 3) & (train.Sex == 0)].median()

med_cl_mal_2 = train.Age[(train.Pclass == 2) & (train.Sex == 0)].median()

med_cl_mal_1 = train.Age[(train.Pclass == 1) & (train.Sex == 0)].median()

train.Age[(train.Pclass == 3) & (train.Sex == 1)] = train.Age[(train.Pclass == 3) & (train.Sex == 1)].fillna(med_cl_fem_3)

train.Age[(train.Pclass == 2) & (train.Sex == 1)] = train.Age[(train.Pclass == 2) & (train.Sex == 1)].fillna(med_cl_fem_2)

train.Age[(train.Pclass == 1) & (train.Sex == 1)] = train.Age[(train.Pclass == 1) & (train.Sex == 1)].fillna(med_cl_fem_1)

train.Age[(train.Pclass == 3) & (train.Sex == 0)] = train.Age[(train.Pclass == 3) & (train.Sex == 0)].fillna(med_cl_mal_3)

train.Age[(train.Pclass == 2) & (train.Sex == 0)] = train.Age[(train.Pclass == 2) & (train.Sex == 0)].fillna(med_cl_mal_2)

train.Age[(train.Pclass == 1) & (train.Sex == 0)] = train.Age[(train.Pclass == 1) & (train.Sex == 0)].fillna(med_cl_mal_1)

#--------------------------------------------- Slices of age base in histogram

age_0 = train.Age[train.Age <=4].copy()

age_1 = train.Age[(train.Age >=5) & (train.Age <=16)].copy()

age_2 = train.Age[(train.Age >=17) & (train.Age <=26)].copy()

age_3 = train.Age[(train.Age >=27) & (train.Age <=36)].copy()

age_4 = train.Age[(train.Age >=37) & (train.Age <=41)].copy()

age_5 = train.Age[(train.Age >=42) & (train.Age <=62)].copy()

age_6 = train.Age[train.Age >=63].copy()

age_0.name, age_1.name, age_2.name = 'age_0', 'age_1', 'age_2'

age_3.name, age_4.name, age_5.name = 'age_3', 'age_4', 'age_5'

age_6.name = 'age_6'

train = pd.concat([train, age_0], axis=1).fillna(0)

train = pd.concat([train, age_1], axis=1).fillna(0)

train = pd.concat([train, age_2], axis=1).fillna(0)

train = pd.concat([train, age_3], axis=1).fillna(0)

train = pd.concat([train, age_4], axis=1).fillna(0)

train = pd.concat([train, age_5], axis=1).fillna(0)

train = pd.concat([train, age_6], axis=1).fillna(0)

train = train.drop(['Age'], axis=1)



#--------------------------------------Members from family

train['Family_size'] = df_train.SibSp + df_train.Parch

train['SibSp'] = df_train['SibSp'].copy()

train['Parch'] = df_train['Parch'].copy()

#------------------------------------------Separated Values out of normally 'outlier'

outlier = np.mean(df_train.Fare) + (2 * np.std(df_train.Fare))

fare_out = df_train.Fare[df_train.Fare >= outlier].copy()

fare = df_train.Fare[df_train.Fare < outlier].copy()

fare_out.name, fare.name = 'Fare_out', 'Fare'

train = pd.concat([train, fare_out], axis=1).fillna(0)

train = pd.concat([train, fare], axis=1).fillna(0)

#-------------------------------------------------Embarked sliced

train['Embarked'] = df_train['Embarked'].fillna('S').copy()

embark = pd.get_dummies(train['Embarked'], prefix='Embark')

train = pd.concat([train, embark], axis=1).copy()

train = train.drop('Embarked', axis=1).copy()

#---------------------------------------get the titles from persons and separated

train['Name'] = df_train.Name.copy()

surname = ['Mr.', 'Mrs', 'Miss', 'Master', 'Rev.', 'Dr', 'Col.', 'Sir.', 

           'Major', 'Don.', 'Capt']

for rep in surname:

    for name in train['Name'][train.Name.str.contains(rep, regex=False)]:

        if rep in ['Col.', 'Major', 'Capt']:

            train.replace(name, 'Army', inplace=True)

        elif rep in ['Sir.', 'Don.']:

            train.replace(name, 'Mr.', inplace=True)

        else:

            train.replace(name, rep, inplace=True)

for title in train.Name:

    if title not in ['Mr.', 'Miss', 'Mrs', 'Master', 'Dr', 'Rev', 'Army']:

        train.replace(title, 'Miss', inplace=True)

name_dum = pd.get_dummies(train.Name, prefix='name').copy()

train = pd.concat([train, name_dum], axis=1)

train = train.drop('Name', axis=1)

#----------------------------------------------- Cabin information

cab = df_train.Cabin.str.extract((r'([A-Za-z])'), ).fillna(0).copy()

cab_dum = pd.get_dummies(cab, prefix='Deck')

train = pd.concat([train, cab_dum], axis=1).copy()

train = train.drop(['Deck_0'], axis=1).copy()

null_cabin = df_train['Cabin'][df_train['Cabin'].isnull()]

null_cabin.fillna(1, inplace=True)

null_cabin.name = 'Null_Cabin'

train = pd.concat([train, null_cabin], axis=1).fillna(0).copy()

#--------------------------------------------------------get the surname, last name

surname = df_train.Name.str.extract(r'(\w+)').copy()

surname.columns = ['Surname']

train = pd.concat([train, surname], axis=1).copy()

mapping = train.Surname.value_counts()

train.Surname = train.Surname.map(mapping).copy()
mapping_ticket = df_train['Ticket'].value_counts().copy()

train['Ticket'] = df_train['Ticket'].copy()

train['Ticket'] = train['Ticket'].map(mapping_ticket)
features = train.columns.drop('Survived')

features
y_true_bas = train.Survived[train.Sex == 1]

y_pred_bas = train.Sex[train.Sex ==1]



bas_acc = np.mean(y_true_bas == y_pred_bas)

bas_f1 = f1_score(y_true_bas, y_pred_bas)
print(f'Baseline Accuracy: {round(bas_acc, 3)}')

print(f'      Baseline F1: {round(bas_f1, 3)}')
X, y = train[features].copy(), train['Survived'].copy()

Xtrain, Xvalid, ytrain, yvalid = train_test_split(X, y, test_size=0.5, stratify=None)
models = [

    KNeighborsClassifier(),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    LGBMClassifier(),

    GaussianNB(),

    SVC(),

    ExtraTreeClassifier(),

    LogisticRegression(max_iter=300),

    GradientBoostingClassifier(),

    AdaBoostClassifier(),

    ExtraTreesClassifier()]

scores = []
# This loop is to get diferents splits and get mean().

# The Stratify is None for not stratify the data, because the Kaggle don't do that.

def models_valid(feature):

    for model in models:

        for ranges in range(200):

            Xtrain, Xvalid, ytrain, yvalid = train_test_split(

                X[feature], y, test_size=0.5, stratify=None) 



            mdl = model

            mdl.fit(Xtrain, ytrain)

            pred = mdl.predict(Xvalid)

            acc = np.mean(yvalid == pred)

            f1 = f1_score(yvalid, pred)

            mod_name = str(model)[:9]

            scores.append([mod_name, acc, f1])



    df_scores = pd.DataFrame(scores, columns=['Model', 'Acc', 'F1'])

    print(Xtrain.columns, df_scores.groupby('Model').mean())
models_valid(features)

# Baseline Accuracy: 0.742

#       Baseline F1: 0.852
# Tryed with LogisticRegression GradientBoosting and RandomForestClassifier, Random was better

scor = []

for feat in range(len(features), 0, -1):

    mdl = RandomForestClassifier(random_state=0)

    #mdl = GradientBoostingClassifier()

    

    rfe = RFE(mdl, step=1, n_features_to_select=feat)

    rfe.fit(Xtrain, ytrain)

    feat_sel = Xtrain.columns[rfe.support_]

    pred = rfe.predict(Xvalid)



    f1 = f1_score(yvalid, pred)

    acc = np.mean(yvalid == pred)

    scor.append([feat_sel, acc, f1])

df_score_sel = pd.DataFrame(scor, columns=['features', 'acc', 'f1'])

index_feat = df_score_sel[df_score_sel.acc == df_score_sel['acc'].max()].index[0] # get index from max()

feat_selected = df_score_sel.features[index_feat].copy() # put the index number to filter and get features

print(f'{feat_selected} \n\n ACC Score: {df_score_sel.acc.max()}')
# reprint the results, now, with features selected

models_valid(feat_selected)
max_depth = [int(x) for x in np.linspace(1, 300)]

max_depth.append(None)

random_state = (0, 301)

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 10000, num = 1000)]

max_features = ['auto', 'sqrt']

min_samples_split = [2, 3, 4, 5, 6, 7, 10, 12, 15]

min_samples_leaf = [1, 2, 4, 6, 8, 10, 13, 14]

bootstrap = [True, False]



learning_rate = (1e-3, 1e-1, 'log-uniform')

num_leaves = (2, 128)

min_child_samples = (1, 100)

subsample = (0.05, 1.0)

colsample_bytree = (0.1, 1.0)



space_LGBM = [learning_rate, num_leaves, min_child_samples, subsample,

               colsample_bytree]



space_RF = [max_depth, n_estimators, max_features,

            min_samples_split, min_samples_leaf, bootstrap]
feat_selected = features
def model_LGBM(params, cross_val=False):

    learning_rate = params[0]

    num_leaves = params[1]

    min_child_samples = params[2]

    subsample = params[3]

    colsample_bytree = params[4]

    

    mdl = LGBMClassifier(learning_rate=learning_rate, num_leaves=num_leaves,

                         min_child_samples=min_child_samples,

                         subsample=subsample, colsample_bytree=colsample_bytree,

                         random_state=0, subsample_freq=1, 

                         n_estimators=100)

    

    Xtrain, Xvalid, ytrain, yvalid = train_test_split(

        X[feat_selected], y, test_size=0.5, stratify=None) 

    

    mdl.fit(Xtrain, ytrain)



    pred = mdl.predict(Xvalid)

    yscore = mdl.predict_proba(Xvalid)[:, 1]



    f1 = f1_score(yvalid, pred)

    roc = roc_auc_score(yvalid, yscore)

    acc = np.mean(yvalid == pred)

    log = log_loss(yvalid, pred)

    

    if cross_val:

        return acc, f1

    else:

        return -acc
result_params_LGBM = dummy_minimize(model_LGBM, space_LGBM,

                                    n_calls=4500, verbose=1,

                                    random_state=0)
params_LGBM = result_params_LGBM.x

print(result_params_LGBM.fun)

plots.plot_convergence(result_params_LGBM)
def model_RF(params, cross_val=False):



    max_depth = params[0]

    n_estimators = params[1]

    max_features = params[2]

    min_samples_split = params[3]

    min_samples_leaf = params[4]

    bootstrap = params[5]



    mdl = RandomForestClassifier(max_depth=max_depth, 

                                 random_state=0,

                                 n_estimators=n_estimators,

                                 max_features=max_features, 

                                 min_samples_split=min_samples_split,

                                 min_samples_leaf=min_samples_leaf,

                                 bootstrap=bootstrap, n_jobs=1)

 

    Xtrain, Xvalid, ytrain, yvalid = train_test_split(

        X[feat_selected], y, test_size=0.5, stratify=None) 

    

    mdl.fit(Xtrain, ytrain)

    pred = mdl.predict(Xvalid)

    yscore = mdl.predict_proba(Xvalid)[:, 1]



    f1 = f1_score(yvalid, pred)

    roc = roc_auc_score(yvalid, yscore)

    acc = np.mean(yvalid == pred)

    log = log_loss(yvalid, pred)

    

    if cross_val:

        return acc, f1

    else:

        return -acc
result_params_RF = dummy_minimize(model_RF, space_RF, n_calls=30,

                                  verbose=1, random_state=0)
params_RF = result_params_RF.x

print(result_params_RF.fun)

plots.plot_convergence(result_params_RF)
features.shape
max_features = (1, 34)

learning_rate = (1e-3, 1e-1, 'log-uniform')

n_estimators = [int(x) for x in np.linspace(1, 32, 32)]

max_depth = [int(x) for x in np.linspace(1, 4, 100)]

min_samples_split = (0.05, 0.9)

min_samples_leaf = (0.05, 0.3)



space_gradient = [max_features, learning_rate, n_estimators,

                 max_depth, min_samples_split, min_samples_leaf]
features.shape
feat_selected.shape
def model_Gradient_Boosting(params, cross_val=False):



    max_features = params[0]

    learning_rate = params[1]

    n_estimators = params[2]

    max_depth = params[3]

    min_samples_split = params[4]

    min_samples_leaf = params[5]



    mdl = GradientBoostingClassifier(max_features=max_features,

                                     learning_rate=learning_rate,

                                     n_estimators=n_estimators,

                                     max_depth=max_depth,

                                     min_samples_split=min_samples_split,

                                     min_samples_leaf=min_samples_leaf)

 

    Xtrain, Xvalid, ytrain, yvalid = train_test_split(

        X[features], y, test_size=0.5, stratify=None) 

    

    mdl.fit(Xtrain, ytrain)

    pred = mdl.predict(Xvalid)

    yscore = mdl.predict_proba(Xvalid)[:, 1]



    f1 = f1_score(yvalid, pred)

    roc = roc_auc_score(yvalid, yscore)

    acc = np.mean(yvalid == pred)

    log = log_loss(yvalid, pred)

    

    if cross_val:

        return acc, f1

    else:

        return -acc
result_params_Gradient = dummy_minimize(model_Gradient_Boosting,

                                        space_gradient, n_calls=3000,

                                        verbose=1, random_state=0)
params_Gradient = result_params_Gradient.x

print(result_params_Gradient.fun)

print(params_Gradient)

plots.plot_convergence(result_params_Gradient)
plots.plot_convergence(result_params_Gradient)

plt.show()

plots.plot_convergence(result_params_LGBM)

plt.show()

plots.plot_convergence(result_params_RF)

plt.show()
plot_data = [-result_params_RF.fun,-result_params_LGBM.fun,-result_params_Gradient.fun]

legends = ['RandomForest', 'LGBM', 'Gradient']
plt.bar(legends, plot_data, width=0.9)
def features_select(params):

    max_depth = params[0]

    n_estimators = params[1]

    max_features = params[2]

    min_samples_split = params[3]

    min_samples_leaf = params[4]

    bootstrap = params[5]



    mdl = RandomForestClassifier(max_depth=max_depth, 

                                 random_state=0,

                                 n_estimators=n_estimators,

                                 max_features=max_features, 

                                 min_samples_split=min_samples_split,

                                 min_samples_leaf=min_samples_leaf,

                                 bootstrap=bootstrap, n_jobs=1)





    Xtrain, Xvalid, ytrain, yvalid = train_test_split(

        X[features], y, test_size=0.5, stratify=None, random_state=0)

    scor = []

    for feat in range(len(features), 0, -1):



        rfe = RFE(mdl, step=3, n_features_to_select=feat)

        rfe.fit(Xtrain, ytrain)

        feat_sel = Xtrain.columns[rfe.support_]

        pred = rfe.predict(Xvalid)



        f1 = f1_score(yvalid, pred)

        acc = np.mean(yvalid == pred)

        scor.append([feat_sel, acc, f1])

    df_score_sel = pd.DataFrame(scor, columns=['features', 'acc', 'f1'])

    index_feat = df_score_sel[df_score_sel.acc == df_score_sel['acc'].max()].index[0] # get index from max()

    feat_selected = df_score_sel.features[index_feat].copy() # put the index number to filter and get features

    print(f'{feat_selected} \n\n ACC Score: {df_score_sel.acc.max()}')

    return feat_selected
feat_selected = features_select(params_RF)
def kFold_RF(params):

    max_depth = params[0]

    n_estimators = params[1]

    max_features = params[2]

    min_samples_split = params[3]

    min_samples_leaf = params[4]

    bootstrap = params[5]



    mdl = RandomForestClassifier(max_depth=max_depth, 

                                 random_state=0,

                                 n_estimators=n_estimators,

                                 max_features=max_features, 

                                 min_samples_split=min_samples_split,

                                 min_samples_leaf=min_samples_leaf,

                                 bootstrap=bootstrap, n_jobs=1)

 

    k_fold = RepeatedKFold(n_splits=2, n_repeats=200, random_state=0) 

    scoring = "accuracy"

    score = cross_val_score(mdl, X[feat_selected], y, cv=k_fold, n_jobs=1, scoring=scoring)

    return score
kf_RF = kFold_RF(params_RF)

kf_RF.mean()
def kFold_LGBM(params):

    learning_rate = params[0]

    num_leaves = params[1]

    min_child_samples = params[2]

    subsample = params[3]

    colsample_bytree = params[4]

    

    mdl = LGBMClassifier(learning_rate=learning_rate, num_leaves=num_leaves,

                         min_child_samples=min_child_samples,

                         subsample=subsample, colsample_bytree=colsample_bytree,

                         random_state=0, subsample_freq=1, 

                         n_estimators=100)

 

    k_fold = RepeatedKFold(n_splits=2, n_repeats=200, random_state=0) 

    scoring = "accuracy"

    score = cross_val_score(mdl, X[feat_selected], y, cv=k_fold, n_jobs=1, scoring=scoring)

    return score
kf_LGBM = kFold_LGBM(params_LGBM)

kf_LGBM.mean()
def kFold_Gradient(params):

    max_features = params[0]

    learning_rate = params[1]

    n_estimators = params[2]

    max_depth = params[3]

    min_samples_split = params[4]

    min_samples_leaf = params[5]



    mdl = GradientBoostingClassifier(max_features=max_features,

                                     learning_rate=learning_rate,

                                     n_estimators=n_estimators,

                                     max_depth=max_depth,

                                     min_samples_split=min_samples_split,

                                     min_samples_leaf=min_samples_leaf)

 

    k_fold = RepeatedKFold(n_splits=2, n_repeats=200, random_state=0) 

    scoring = "accuracy"

    score = cross_val_score(mdl, X[feat_selected], y, cv=k_fold, n_jobs=1, scoring=scoring)

    return score
kf_Gradient = kFold_Gradient(params_Gradient)

kf_Gradient.mean()
data_kf_plot = [kf_RF, kf_LGBM, kf_Gradient]

legend_kf = ['Random', 'LGBM', 'Gradient']
print(f'Mean RF: {kf_RF.mean()}')

print(f'Mean LGBM: {kf_LGBM.mean()}')

print(f'Mean Gradient: {kf_Gradient.mean()}')

plt.figure(figsize=(14,6))

plt.hist(kf_RF)

plt.hist(kf_LGBM, alpha=0.7, color='green')

plt.hist(kf_Gradient, alpha=0.7, color='yellow')

plt.show()
test = pd.DataFrame()



test['Pclass'] = df_test['Pclass'].copy()

test['Sex'] = df_test.Sex.map({

    'female':1,

    'male':0

}).copy()

#---------------------------------------

test['Age'] = df_test['Age'].copy()



med_cl_fem_3 = test.Age[(test.Pclass == 3) & (test.Sex == 1)].median()

med_cl_fem_2 = test.Age[(test.Pclass == 2) & (test.Sex == 1)].median()

med_cl_fem_1 = test.Age[(test.Pclass == 1) & (test.Sex == 1)].median()

med_cl_mal_3 = test.Age[(test.Pclass == 3) & (test.Sex == 0)].median()

med_cl_mal_2 = test.Age[(test.Pclass == 2) & (test.Sex == 0)].median()

med_cl_mal_1 = test.Age[(test.Pclass == 1) & (test.Sex == 0)].median()



test.Age[(test.Pclass == 3) & (test.Sex == 1)] = test.Age[(test.Pclass == 3) & (test.Sex == 1)].fillna(med_cl_fem_3)

test.Age[(test.Pclass == 2) & (test.Sex == 1)] = test.Age[(test.Pclass == 2) & (test.Sex == 1)].fillna(med_cl_fem_2)

test.Age[(test.Pclass == 1) & (test.Sex == 1)] = test.Age[(test.Pclass == 1) & (test.Sex == 1)].fillna(med_cl_fem_1)

test.Age[(test.Pclass == 3) & (test.Sex == 0)] = test.Age[(test.Pclass == 3) & (test.Sex == 0)].fillna(med_cl_mal_3)

test.Age[(test.Pclass == 2) & (test.Sex == 0)] = test.Age[(test.Pclass == 2) & (test.Sex == 0)].fillna(med_cl_mal_2)

test.Age[(test.Pclass == 1) & (test.Sex == 0)] = test.Age[(test.Pclass == 1) & (test.Sex == 0)].fillna(med_cl_mal_1)



#---------------------------------------

age_0 = test.Age[test.Age <=4].copy()

age_1 = test.Age[(test.Age >=5) & (test.Age <=16)].copy()

age_2 = test.Age[(test.Age >=17) & (test.Age <=26)].copy()

age_3 = test.Age[(test.Age >=27) & (test.Age <=36)].copy()

age_4 = test.Age[(test.Age >=37) & (test.Age <=41)].copy()

age_5 = test.Age[(test.Age >=42) & (test.Age <=62)].copy()

age_6 = test.Age[test.Age >=63].copy()

age_0.name, age_1.name, age_2.name = 'age_0', 'age_1', 'age_2'

age_3.name, age_4.name, age_5.name = 'age_3', 'age_4', 'age_5'

age_6.name = 'age_6'

test = pd.concat([test, age_0], axis=1).fillna(0)

test = pd.concat([test, age_1], axis=1).fillna(0)

test = pd.concat([test, age_2], axis=1).fillna(0)

test = pd.concat([test, age_3], axis=1).fillna(0)

test = pd.concat([test, age_4], axis=1).fillna(0)

test = pd.concat([test, age_5], axis=1).fillna(0)

test = pd.concat([test, age_6], axis=1).fillna(0)

test = test.drop(['Age'], axis=1)



#---------------------------------------



test['Family_size'] = df_test.SibSp + df_test.Parch



#---------------------------------------



test['SibSp'] = df_test['SibSp'].copy()

test['Parch'] = df_test['Parch'].copy()



#---------------------------------------



outlier = np.mean(df_test.Fare) + (2 * np.std(df_test.Fare))

fare_out = df_test.Fare[df_test.Fare >= outlier].copy()

fare = df_test.Fare[df_test.Fare < outlier].copy()

fare_out.name, fare.name = 'Fare_out', 'Fare'



test = pd.concat([test, fare_out], axis=1).fillna(0)

test = pd.concat([test, fare], axis=1).fillna(0)



#---------------------------------------



test['Embarked'] = df_test['Embarked'].fillna('S').copy()

embark = pd.get_dummies(test['Embarked'], prefix='Embark')

test = pd.concat([test, embark], axis=1).copy()

test = test.drop('Embarked', axis=1).copy()



#---------------------------------------



test['Name'] = df_test.Name.copy()

surname = ['Mr.', 'Mrs', 'Miss', 'Master', 'Rev.', 'Dr', 'Col.', 'Sir.', 'Major', 'Don.', 'Capt']

for rep in surname:

    for name in test['Name'][test.Name.str.contains(rep, regex=False)]:

        if rep in ['Col.', 'Major', 'Capt']:

            test.replace(name, 'Army', inplace=True)

        elif rep in ['Sir.', 'Don.']:

            test.replace(name, 'Mr.', inplace=True)

        else:

            test.replace(name, rep, inplace=True)



for title in test.Name:

    if title not in ['Mr.', 'Miss', 'Mrs', 'Master', 'Dr', 'Rev', 'Army']:

        test.replace(title, 'Miss', inplace=True)

        

name_dum = pd.get_dummies(test.Name, prefix='name').copy()

test = pd.concat([test, name_dum], axis=1)



test = test.drop('Name', axis=1)



#---------------------------------------



cab = df_test.Cabin.str.extract((r'([A-Za-z])'), ).fillna(0).copy()

cab_dum = pd.get_dummies(cab, prefix='Deck')

test = pd.concat([test, cab_dum], axis=1).copy()

test = test.drop(['Deck_0'], axis=1).copy()



#---------------------------------------



test['Deck_T'] = 0



#---------------------------------------



surname = df_test.Name.str.extract(r'(\w+)').copy()

surname.columns = ['Surname']



test = pd.concat([test, surname], axis=1).copy()

mapping = test.Surname.value_counts()

test.Surname = test.Surname.map(mapping).copy()



#---------------------------------------

null_cabin = df_test['Cabin'][df_test['Cabin'].isnull()]

null_cabin.fillna(1, inplace=True)

null_cabin.name = 'Null_Cabin'

test = pd.concat([test, null_cabin], axis=1).fillna(0).copy()



#---------------------------------------

mapping_ticket = df_test['Ticket'].value_counts().copy()

test['Ticket'] = df_test['Ticket'].copy()

test['Ticket'] = test['Ticket'].map(mapping_ticket)
def sub_RF_params(params):



    max_depth = params[0]

    n_estimators = params[1]

    max_features = params[2]

    min_samples_split = params[3]

    min_samples_leaf = params[4]

    bootstrap = params[5]



    mdl = RandomForestClassifier(max_depth=max_depth, 

                                 random_state=0,

                                 n_estimators=n_estimators,

                                 max_features=max_features, 

                                 min_samples_split=min_samples_split,

                                 min_samples_leaf=min_samples_leaf,

                                 bootstrap=bootstrap, n_jobs=1)

    mdl.fit(X[feat_selected], y)



    pred = mdl.predict(test[feat_selected])

    mod_name = str(mdl)[:4]



    sub = pd.DataFrame({

    'PassengerId':df_test['PassengerId'],

    'Survived':pred

    })



    sub.to_csv(f"feat_sel_{mod_name}.csv", index=False)

sub_RF_params(params_RF)
def sub_LGBM_params(params):

    

    learning_rate = params[0]

    num_leaves = params[1]

    min_child_samples = params[2]

    subsample = params[3]

    colsample_bytree = params[4]

    

    mdl = LGBMClassifier(learning_rate=learning_rate, num_leaves=num_leaves,

                         min_child_samples=min_child_samples,

                         subsample=subsample, colsample_bytree=colsample_bytree,

                         random_state=0, subsample_freq=1, n_estimators=100)

    

    mdl.fit(Xtrain[feat_selected], ytrain)

    pred = mdl.predict(test[feat_selected])

    mod_name = str(mdl)[:4]



    sub = pd.DataFrame({

    'PassengerId':df_test['PassengerId'],

    'Survived':pred

    })



    sub.to_csv(f"feat_sel_{mod_name}.csv", index=False)

sub_LGBM_params(params_LGBM)
def sub_Gradient(params, cross_val=False):



    max_features = params[0]

    learning_rate = params[1]

    n_estimators = params[2]

    max_depth = params[3]

    min_samples_split = params[4]

    min_samples_leaf = params[5]



    mdl = GradientBoostingClassifier(max_features=max_features,

                                     learning_rate=learning_rate,

                                     n_estimators=n_estimators,

                                     max_depth=max_depth,

                                     min_samples_split=min_samples_split,

                                     min_samples_leaf=min_samples_leaf)

 

    

    mdl.fit(Xtrain[feat_selected], ytrain)

    pred = mdl.predict(test[feat_selected])

    mod_name = str(mdl)[:4]



    sub = pd.DataFrame({

    'PassengerId':df_test['PassengerId'],

    'Survived':pred

    })



    sub.to_csv(f"feat_sel_{mod_name}.csv", index=False)

sub_Gradient(params_Gradient)