import numpy as np

import pandas as pd
data_train = pd.read_csv('../input/train.csv')

data_train.head()
data_test = pd.read_csv('../input/test.csv')

data_test.head()
df = data_train.append(data_test, sort = True)

df.shape
df.info()

print('-------------------------------------')

print(pd.isnull(df).sum())
df.describe()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



plt.style.use('bmh')

plt.rc('font', family='DejaVu Sans', size=13)
cat_list = ['Cabin', 'Embarked', 'Name', 'Parch', 'Pclass', 'Sex', 'SibSp', 'Ticket']

for n, i in enumerate(cat_list):

    cat_num = len(df[i].value_counts().index)

    print('The feature "%s" has %d values.' % (i, cat_num))
f, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize = (20, 5))

sns.countplot(x = 'Sex', hue = 'Survived', data = data_train, ax = ax1)

sns.countplot(x = 'Pclass', hue = 'Survived', data = data_train, ax = ax2)

sns.countplot(x = 'Embarked', hue = 'Survived', data = data_train, ax = ax3)

f.suptitle('Nominal/Ordinal feature', size = 20, y = 1.1)



f, [ax1, ax2] = plt.subplots(1, 2, figsize = (20, 5))

sns.countplot(x = 'SibSp', hue = 'Survived', data = data_train, ax = ax1)

sns.countplot(x = 'Parch', hue = 'Survived', data = data_train, ax = ax2)



plt.show()
grid = sns.FacetGrid(df, col = 'Pclass', hue = 'Sex', palette = 'seismic', height = 5)

grid.map(sns.countplot, 'Embarked', alpha = 0.8)

grid.add_legend()
grid = sns.FacetGrid(data_train, row = 'Sex', col = 'Pclass', hue = 'Survived', palette = 'seismic', height = 5)

grid.map(sns.countplot, 'Embarked', alpha = 0.8)

grid.add_legend()
f, ax = plt.subplots(figsize = (10, 5))

sns.kdeplot(data_train.loc[data_train.Survived == 0, 'Age'], color = 'gray', shade = True, label = 'dead')

sns.kdeplot(data_train.loc[data_train.Survived == 1, 'Age'], color = 'green', shade = True, label = 'survived')

plt.title('Survival rate in different age')

plt.xlabel('Age')

plt.ylabel('Frequency')
f, [ax1, ax2] = plt.subplots(1, 2, figsize = (20, 6))

sns.boxplot(x = 'Pclass', y = 'Age', data = data_train, ax = ax1)

sns.swarmplot(x = 'Pclass', y = 'Age', data = data_train, ax = ax1)

sns.kdeplot(data_train.loc[data_train.Pclass == 3, 'Age'], color = 'b', shade = True, label = 'Pclass 3', ax = ax2)

sns.kdeplot(data_train.loc[data_train.Pclass == 2, 'Age'], color = 'g', shade = True, label = 'Pclass 2', ax = ax2)

sns.kdeplot(data_train.loc[data_train.Pclass == 1, 'Age'], color = 'r', shade = True, label = 'Pclass 1', ax = ax2)

ax1.set_title('Pclass-Age box-plot')

ax2.set_title('Pclass-Age kde-plot')

f.show()
grid = sns.FacetGrid(data_train, row = 'Sex', col = 'Pclass', hue = 'Survived', palette = 'seismic', height = 3.5)

grid.map(plt.scatter, 'PassengerId', 'Age', alpha = 0.8)

grid.add_legend()
grid = sns.FacetGrid(data_train, row = 'Sex', col = 'Embarked', hue = 'Survived', palette = 'seismic', height = 3.5)

grid.map(plt.scatter, 'PassengerId', 'Age', alpha = 0.8)

grid.add_legend()
grid = sns.FacetGrid(data_train, row = 'Sex', col = 'SibSp', hue = 'Survived', palette = 'seismic', height = 3.5)

grid.map(plt.scatter, 'PassengerId', 'Age', alpha = 0.8)

grid.add_legend()
grid = sns.FacetGrid(data_train, row = 'Sex', col = 'Parch', hue = 'Survived', palette = 'seismic', height = 3.5)

grid.map(plt.scatter, 'PassengerId', 'Age', alpha = 0.8)

grid.add_legend()
f, ax = plt.subplots(figsize = (10, 5))

sns.kdeplot(data_train.loc[data_train.Survived == 0, 'Fare'], color = 'gray', shade = True, label = 'dead')

sns.kdeplot(data_train.loc[data_train.Survived == 1, 'Fare'], color = 'green', shade = True, label = 'survived')

plt.title('Survival rate in different fare')

plt.xlabel('Fare')

plt.ylabel('Frequency')
f, [ax1, ax2] = plt.subplots(1, 2, figsize = (20, 6))

sns.boxplot(x = 'Pclass', y = 'Fare', data = data_train, ax = ax1)

sns.swarmplot(x = 'Pclass', y = 'Fare', data = data_train, ax = ax1)

sns.kdeplot(data_train.loc[data_train.Pclass == 3, 'Fare'], color = 'b', shade = True, label = 'Pclass 3', ax = ax2)

sns.kdeplot(data_train.loc[data_train.Pclass == 2, 'Fare'], color = 'g', shade = True, label = 'Pclass 2', ax = ax2)

sns.kdeplot(data_train.loc[data_train.Pclass == 1, 'Fare'], color = 'r', shade = True, label = 'Pclass 1', ax = ax2)

ax1.set_title('Pclass-Fare box-plot')

ax2.set_title('Pclass-Fare kde-plot')

f.show()
grid = sns.FacetGrid(data_train, row = 'Sex', col = 'Pclass', hue = 'Survived', palette = 'seismic', height = 3.5)

grid.map(plt.scatter, 'Age', 'Fare', alpha = 0.8)

grid.add_legend()
g = sns.pairplot(data_train[['Survived', 'Pclass', 'Sex', 'Age', 'Parch', 'Fare', 'Embarked']], hue = 'Survived', palette = 'seismic', height = 4, diag_kind = 'kde', diag_kws = dict(shade = True), plot_kws = dict(s = 50, alpha = 0.8))

g.set(xticklabels = [])
df.isnull().sum()
df[df.Fare.isnull()]
df[(df.Pclass == 3) & (df.Age > 60) & (df.Sex == 'male')]
fare_mean = df[(df.Pclass == 3) & (df.Age > 60) & (df.Sex == 'male')].Fare.mean()

df.loc[df.PassengerId == 1044, 'Fare'] = fare_mean

df[df.PassengerId == 1044]
df[df.Embarked.isnull()]
df.Embarked = df.Embarked.fillna('C')
df.Cabin = df.Cabin.fillna('0')

len(df.Cabin.value_counts().index)
df['CabinCat'] = pd.Categorical(df.Cabin.apply(lambda x : x[0])).codes
f, ax = plt.subplots(figsize = (10, 5))

sns.countplot('CabinCat', hue = 'Survived', data = df, ax = ax)

f.show()
import re

from sklearn.preprocessing import LabelEncoder
# Title

df['Title'] = df.Name.apply(lambda x : re.search(' ([a-zA-Z]+)\.', x).group(1))

title_mapping = {'Mr' : 1, 'Miss' : 2, 'Mrs' : 3, 'Master' : 4, 'Dr' : 5, 'Rev' : 6, 'Major' : 7, 'Col' : 7, 'Mlle' : 2, 'Mme' : 3, 'Don' : 9, 'Dona' : 9, 'Lady' : 10, 'Countess' : 10, 'Jonkheer' : 10, 'Sir' : 9, 'Capt' : 7, 'Ms' : 2}

df['TitleCat'] = df.Title.map(title_mapping)



# FamilySize

df['FamilySize'] = df.SibSp + df.Parch + 1



# FamilyName

df['FamilyName'] = df.Name.apply(lambda x : str.split(x, ',')[0])



# IsAlone

df['IsAlone'] = 0

df.loc[df.FamilySize == 1, 'IsAlone'] = 1



# NameLength

le = LabelEncoder()

df['NameLength'] = df.Name.apply(lambda x : len(x))

df['NameLengthBin'] = pd.qcut(df.NameLength, 5)

df['NameLengthBinCode'] = le.fit_transform(df.NameLengthBin)



# Embarked

df['Embarked'] = pd.Categorical(df.Embarked).codes



# Sex

df = pd.concat([df, pd.get_dummies(df.Sex)], axis = 1)



# Ticket

table_ticket = pd.DataFrame(df.Ticket.value_counts())

table_ticket.rename(columns = {'Ticket' : 'TicketNum'}, inplace = True)

table_ticket['TicketId'] = pd.Categorical(table_ticket.index).codes

table_ticket.loc[table_ticket.TicketNum < 3, 'TicketId'] = -1

df = pd.merge(left = df, right = table_ticket, left_on = 'Ticket', right_index = True, how = 'left', sort = False)

df['TicketCode'] = list(pd.cut(df.TicketId, bins = [-2, 0, 500, 1000], labels = [0, 1, 2]))



# CabinNum

regex = re.compile('\s*(\w+)\s*')

df['CabinNum'] = df.Cabin.apply(lambda x : len(regex.findall(x)))
from sklearn.ensemble import ExtraTreesRegressor
classers = ['Fare','Parch','Pclass','SibSp','TitleCat', 'CabinCat','female','male', 'Embarked', 'FamilySize', 'IsAlone', 'NameLengthBinCode','TicketNum','TicketCode']



etr = ExtraTreesRegressor(n_estimators = 200, random_state = 0)

age_X_train = df[classers][df.Age.notnull()]

age_y_train = df.Age[df.Age.notnull()]

age_X_test = df[classers][df.Age.isnull()]



etr.fit(age_X_train, np.ravel(age_y_train))

age_pred = etr.predict(age_X_test)

df.loc[df.Age.isnull(), 'Age'] = age_pred
age_X_test['Age'] = age_pred



f, ax = plt.subplots(figsize = (10, 5))

sns.boxplot('Pclass', 'Age', data = age_X_test, ax = ax)

sns.swarmplot('Pclass', 'Age', data = age_X_test, ax = ax)
# Identity

childAge = 18

def getIdentity(passenger):

    age, sex = passenger

    

    if age < childAge:

        return 'child'

    elif sex == 'male':

        return 'male_adult'

    else:

        return 'female_adult'



df = pd.concat([df, pd.DataFrame(df[['Age', 'Sex']].apply(getIdentity, axis = 1), columns = ['Identity'])], axis = 1)

df = pd.concat([df, pd.get_dummies(df.Identity)], axis = 1)
# FamilySurvival

DEFAULT_SURVIVAL_VALUE = 0.5

df['FamilySurvival'] = DEFAULT_SURVIVAL_VALUE



for _, grp_df in df.groupby(['FamilyName', 'Fare']):

    if len(grp_df) != 1 :

        for index, row in grp_df.iterrows():

            smax = grp_df.drop(index).Survived.max()

            smin = grp_df.drop(index).Survived.min()

            pid = row.PassengerId

            

            if smax == 1:

                df.loc[df.PassengerId == pid, 'FamilySurvival'] = 1.0

            elif smin == 0:

                df.loc[df.PassengerId == pid, 'FamilySurvival'] = 0.0

for _, grp_df in df.groupby(['Ticket']):

    if len(grp_df != 1):

        for index, row in grp_df.iterrows():

            if (row.FamilySurvival == 0.0 or row.FamilySurvival == 0.5):

                smax = grp_df.drop(index).Survived.max()

                smin = grp_df.drop(index).Survived.min()

                pid = row.PassengerId

                

                if smax == 1:

                    df.loc[df.PassengerId == pid, 'FamilySurvival'] = 1.0

                elif smin == 0:

                    df.loc[df.PassengerId == pid, 'FamilySurvival'] = 0.0

                    

df.FamilySurvival.value_counts()
# FareBinCode

df['FareBin'] = pd.qcut(df.Fare, 5)



le = LabelEncoder()

df['FareBinCode'] = le.fit_transform(df.FareBin)
# AgeBinCode

df['AgeBin'] = pd.qcut(df.Age, 4)



le = LabelEncoder()

df['AgeBinCode'] = le.fit_transform(df.AgeBin)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
target = data_train['Survived'].values

select_features = ['AgeBinCode', 'Embarked', 'FareBinCode', 'Parch', 'Pclass', 'SibSp', 'CabinCat', 'TitleCat', 'FamilySize', 'IsAlone', 'FamilySurvival', 'NameLengthBinCode', 'female', 'male', 'TicketNum', 'TicketCode', 'CabinNum', 'child', 'female_adult', 'male_adult']
#unimportant_features = ['Parch', 'Embarked', 'child', 'CabinNum']

#select_features = list(set(select_features).difference(unimportant_features))
#scaler = MinMaxScaler()

scaler = StandardScaler()



#df_scaled = df[select_features]

df_scaled = scaler.fit_transform(df[select_features])



train = df_scaled[0:891].copy()

test = df_scaled[891:].copy()
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, len(select_features))

selector.fit(train, target)

scores = -np.log10(selector.pvalues_)

indices = np.argsort(scores)[::-1]



print('Features importance:')

for i in range(len(scores)):

    print('%.2f %s' % (scores[indices[i]], select_features[indices[i]]))
df_corr = df[select_features].copy()



colormap = plt.cm.RdBu

plt.figure(figsize = (16, 16))

sns.heatmap(df_corr.corr(), linewidths = 0.1, vmax = 1.0, square = True, cmap = colormap, linecolor = 'white', annot = True)

plt.show()
from sklearn.model_selection import KFold, cross_val_score

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



kf = KFold(n_splits = 5, random_state = 1)
from sklearn.ensemble import RandomForestClassifier



rfc_parameters = {'max_depth' : [5], 'n_estimators' : [500], 'min_samples_split' : [9], 'random_state' : [1], 'n_jobs' : [-1]}

rfc = RandomForestClassifier()

clf_rfc = GridSearchCV(rfc, rfc_parameters, n_jobs = -1, cv = kf, scoring = 'roc_auc')



clf_rfc.fit(train, target)
rfc2_parameters = {'max_depth' : [2, 5, 8, 10, 20, 50], 'n_estimators' : [10, 50, 100, 200, 500, 1000, 2000], 'min_samples_split' : [2, 3, 5, 9, 20]}

rfc2 = RandomForestClassifier(random_state = 1, n_jobs = -1)

clf_rfc2 = RandomizedSearchCV(rfc2, rfc2_parameters, n_jobs = -1, cv = kf, scoring = 'roc_auc')



clf_rfc2.fit(train, target)
importance = clf_rfc.best_estimator_.feature_importances_

indices = np.argsort(importance)[::-1]



print(clf_rfc.best_score_)

print(clf_rfc.score(train, target))

print(clf_rfc.best_params_)

print('\nFeature importance:')

for i in range(len(select_features)):

    print('%.2f %s' % (importance[indices[i]], select_features[indices[i]]))
from sklearn.linear_model import LogisticRegression



lr_paramaters = {'C' : [0.05, 0.1, 0.2], 'random_state' : [1]}

lr = LogisticRegression()



clf_lr = GridSearchCV(lr, lr_paramaters, n_jobs = -1, cv = kf, scoring = 'roc_auc')

clf_lr.fit(train, target)
print(clf_lr.best_score_)

print(clf_lr.score(train, target))

print(clf_lr.best_params_)
from sklearn.svm import SVC



svc_paramaters = {'C' : [5.5, 6, 6.5], 'kernel' : ['linear', 'rbf'], 'gamma' : ['auto', 'scale'], 'random_state' : [1]}

svc = SVC()



clf_svc = GridSearchCV(svc, svc_paramaters, n_jobs = -1, cv = kf, scoring = 'roc_auc')

clf_svc.fit(train, target)
print(clf_svc.best_score_)

print(clf_svc.score(train, target))

print(clf_svc.best_params_)
from sklearn.ensemble import GradientBoostingClassifier



gbdt_parameters = {'subsample' : [1], 'min_samples_leaf' : [3], 'learning_rate' : [0.1], 'n_estimators' : [50], 'min_samples_split' : [2], 'max_depth' : [3], 'random_state' : [1]}

gbdt = GradientBoostingClassifier()



clf_gbdt = GridSearchCV(gbdt, gbdt_parameters, n_jobs = -1, cv = kf, scoring = 'roc_auc')

clf_gbdt.fit(train, target)
print(clf_gbdt.best_score_)

print(clf_gbdt.score(train, target))

print(clf_gbdt.best_params_)
from xgboost import XGBClassifier



xgb_paramaters = {'subsample' : [0.7], 'min_child_weight' : [1], 'max_depth' : [3], 'learning_rate' : [0.1], 'n_estimators' : [100], 'n_jobs' : [-1], 'random_state' : [1]}

xgb = XGBClassifier()



clf_xgb = GridSearchCV(xgb, xgb_paramaters, n_jobs = -1, cv = kf, scoring = 'roc_auc')

clf_xgb.fit(train, target)
print(clf_xgb.best_score_)

print(clf_xgb.score(train, target))

print(clf_xgb.best_params_)
prediction = clf_rfc2.predict(test)
submission = pd.DataFrame({'Survived' : prediction}, index = data_test.PassengerId)

submission.to_csv('submission.csv', index_label = ['PassengerId'])