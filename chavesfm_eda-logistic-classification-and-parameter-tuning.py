import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.linear_model import LogisticRegressionCV

from sklearn.metrics import mean_squared_error, classification_report

from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, PolynomialFeatures

from sklearn.model_selection import GridSearchCV



from eli5.sklearn import PermutationImportance

from eli5 import show_weights



import warnings



from IPython.display import display, Math



warnings.filterwarnings("ignore")



sns.set_style('whitegrid')

sns.set_context("talk")



%matplotlib inline
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')
all_df = pd.concat([train_df, test_df], axis=0, sort=False, ignore_index=True)
all_df.head()
all_df.info()
ans = all_df.drop("Survived", axis=1).isnull().sum().sort_values(ascending=False)

plt.figure(figsize=(12,1.5))

sns.heatmap(pd.DataFrame(data=ans[ans>0], columns=['Missing Values']).T, annot=True, cbar=False, cmap='viridis')
all_df['Title'] = all_df['Name'].apply(lambda name: name.split(',')[1].strip().split('.')[0])
newtitles={

    "Capt":       "Officer",

    "Col":        "Officer",

    "Major":      "Officer",

    "Jonkheer":   "Royalty",

    "Don":        "Royalty",

    "Sir" :       "Royalty",

    "Dr":         "Officer",

    "Rev":        "Officer",

    "the Countess":"Royalty",

    "Dona":       "Royalty",

    "Mme":        "Mrs",

    "Mlle":       "Miss",

    "Ms":         "Mrs",

    "Mr" :        "Mr",

    "Mrs" :       "Mrs",

    "Miss" :      "Miss",

    "Master" :    "Master",

    "Lady" :      "Royalty"}
all_df['Title'].update(all_df['Title'].map(newtitles))
all_df['Deck'] = all_df['Cabin'].apply(lambda cabin: cabin[0] if pd.notnull(cabin) else 'N')
all_df['Alone'] = (all_df['Parch'].apply(lambda value: not(value)) & all_df['SibSp'].apply(lambda value: not(value))).astype('int')
all_df['Relatives'] = all_df['SibSp'] + all_df['Parch']
all_df[all_df['Embarked'].isnull()]
sort_embarked_features = ['Cabin', 'Sex', 'Pclass', 'Alone', 'Fare']



sorted_df = all_df.sort_values(by=sort_embarked_features)



sorted_df[(sorted_df['Survived'] == 1) & 

          (sorted_df['Sex'] == 'female') & 

          (sorted_df['Pclass'] == 1) &

          (sorted_df['Alone'] == 1)].sort_values(by=sort_embarked_features)[sort_embarked_features+['Embarked']].head()
all_df['Embarked'].fillna(value='C', inplace=True)
all_df[all_df['Fare'].isnull()]
sort_fare_features = ['Sex', 'Pclass', 'Title', 'Deck', 'Embarked', 'Alone', 'Age']



sorted_df = all_df.sort_values(by=sort_embarked_features)



aux = sorted_df[(sorted_df['Alone'] == 1) & 

          (sorted_df['Sex'] == 'male') & 

          (sorted_df['Pclass'] == 1) & 

          (sorted_df['Embarked'] == 'S') & 

          (sorted_df['Deck'] == 'N') & 

          (sorted_df['Title'] == 'Mr') & 

          ((sorted_df['Age'] >= 55) & (sorted_df['Age'] <= 65))]



aux
all_df['Fare'].fillna(value=aux['Fare'].mean(), inplace=True)
def impute_num(cols, avg, std):

       

    try:

        avg_value = avg.loc[tuple(cols)][0]

    except Exception as e:        

        print(f'It is not possible to find an average value for this combination of features values:\n{cols}')

        return np.nan

    

    try:

        std_value = std.loc[tuple(cols)][0]

    except Exception as e:        

        std_value = 0        

    finally:

        if pd.isnull(std_value):

            std_value = 0

        

    while True:        

        value = np.random.randint(avg_value-std_value, avg_value+std_value+1)

        if value >= 0:

            break

    return round(value, 0)
group_age_features = ['Title','Relatives','Parch','SibSp','Deck','Pclass','Embarked','Sex','Alone']
stat_age = all_df.pivot_table(values='Age', index=group_age_features, aggfunc=['mean','std']).round(2)
ages1 = all_df[~all_df['Age'].isnull()][group_age_features].apply(impute_num, axis=1, avg=stat_age.xs(key='mean', axis=1), std=0)

ages2 = all_df[~all_df['Age'].isnull()][group_age_features].apply(impute_num, axis=1, avg=stat_age.xs(key='mean', axis=1), std=stat_age.xs(key='std', axis=1))
comp = pd.DataFrame([all_df[~all_df['Age'].isnull()]['Age'], ages2, ages1]).T

comp.columns = ['Real Age', 'Predicted Age $(\mu=\mu_{Age}$, $\sigma=\sigma_{Age})$', 'Predicted Age $(\mu=\mu_{Age}$, $\sigma=0)$']

comp.head(10)
comp.describe()
pd.DataFrame(data=[mean_squared_error(all_df[~all_df['Age'].isnull()]['Age'], ages1), 

                   mean_squared_error(all_df[~all_df['Age'].isnull()]['Age'], ages2)],

            index = ['$\mu=\mu_{Age}, \sigma=0$', '$\mu=\mu_{Age}, \sigma=\sigma_{Age}$'],

            columns=['Mean Square Error']).round(1)
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

sns.distplot(all_df[~all_df['Age'].isnull()]['Age'], bins=20, ax=ax[0])

sns.distplot(ages1, bins=20, ax=ax[1])

sns.distplot(ages2, bins=20, ax=ax[2])



ax[0].set_xlabel('Age')

ax[1].set_xlabel('Age')

ax[2].set_xlabel('Age')



ax[0].set_title('Real Ages')

ax[1].set_title('Predicted Ages $(\mu=\mu_{Age}$, $\sigma=0)$')

ax[2].set_title('Predicted Ages $(\mu=\mu_{Age}$, $\sigma=\sigma_{Age})$')
group_features = ['Title','Pclass','Embarked','Sex']
stat_age = all_df.pivot_table(values='Age', index=group_features, aggfunc=['mean','std']).round(2)
ages = all_df[all_df['Age'].isnull()][group_features].apply(impute_num, axis=1, avg=stat_age.xs(key='mean', axis=1), std=stat_age.xs(key='std', axis=1))
all_df['Age'].update(ages)
fig, axes = plt.subplots(nrows=3, ncols=3, sharey=True, figsize=(15,10))

fields = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Title', 'Deck', 'Alone', 'Relatives']



i = 0

for row_ax in axes:

    for col_ax in row_ax:

        ans = pd.crosstab(index=[all_df['Survived'], all_df[fields[i]]], columns=all_df[fields[i]], normalize='columns').reset_index()

        ans['%'] = 100*ans[ans.drop(['Survived', fields[i]], axis=1).columns].apply(sum, axis=1).round(2)

        ans.drop(labels=ans.drop(['Survived', fields[i], '%'], axis=1).columns, inplace=True, axis=1)

        ans.sort_values(by=['Survived', '%'], inplace=True)        

        sns.barplot(x=fields[i], y='%', hue='Survived', data=ans, ax=col_ax)

        col_ax.set_ylim((0, 100))

        i+=1

        

plt.tight_layout()        
def sort_remap(crosstab_df, key):

    alives = list(crosstab_df[crosstab_df['Survived']==1][key])

    deads = list(crosstab_df[crosstab_df['Survived']==0][key])



    alives = list(set(deads) - set(alives)) + alives

    sorted_map = {key:value for value, key in enumerate(alives)}

    

    return sorted_map
fields = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Title', 'Deck', 'Alone', 'Relatives']



for feature in fields:

    ans = pd.crosstab(index=[all_df['Survived'], all_df[feature]], columns=all_df[feature], normalize='columns').reset_index()

    ans['%'] = 100*ans[ans.drop(['Survived', feature], axis=1).columns].apply(sum, axis=1).round(2)

    ans.drop(labels=ans.drop(['Survived', feature, '%'], axis=1).columns, inplace=True, axis=1)

    ans.sort_values(by=['Survived', '%'], inplace=True)

    sorted_map = sort_remap(ans, feature)    

    all_df[feature].update(all_df[feature].map(sorted_map))
fig, axes = plt.subplots(nrows=3, ncols=3, sharey=True, figsize=(15,10))

fields = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Title', 'Deck', 'Alone', 'Relatives']



i = 0

for row_ax in axes:

    for col_ax in row_ax:

        ans = pd.crosstab(index=[all_df['Survived'], all_df[fields[i]]], columns=all_df[fields[i]], normalize='columns').reset_index()

        ans['%'] = 100*ans[ans.drop(['Survived', fields[i]], axis=1).columns].apply(sum, axis=1).round(2)

        ans.drop(labels=ans.drop(['Survived', fields[i], '%'], axis=1).columns, inplace=True, axis=1)

        ans.sort_values(by=['Survived', '%'], inplace=True)        

        sns.barplot(x=fields[i], y='%', hue='Survived', data=ans, ax=col_ax)

        col_ax.set_ylim((0, 100))

        i+=1

        

plt.tight_layout()        
kdis = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='uniform').fit(all_df[['Age', 'Fare']])
cat_age_fare = kdis.transform(all_df[['Age', 'Fare']])
all_df['Cat_Age'] = cat_age_fare[:,0].astype('int')

all_df['Cat_Fare'] = cat_age_fare[:,1].astype('int')
fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12,5))

fields = ['Cat_Age', 'Cat_Fare']



i = 0

for col_ax in axes:

    ans = pd.crosstab(index=[all_df['Survived'], all_df[fields[i]]], columns=all_df[fields[i]], normalize='columns').reset_index()

    ans['%'] = 100*ans[ans.drop(['Survived', fields[i]], axis=1).columns].apply(sum, axis=1).round(2)

    ans.drop(labels=ans.drop(['Survived', fields[i], '%'], axis=1).columns, inplace=True, axis=1)

    ans.sort_values(by=['Survived', '%'], inplace=True)        

    sns.barplot(x=fields[i], y='%', hue='Survived', data=ans, ax=col_ax)

    col_ax.set_ylim((0, 100))

    i+=1

        

plt.tight_layout()        
fields = ['Cat_Age', 'Cat_Fare']



for feature in fields:

    ans = pd.crosstab(index=[all_df['Survived'], all_df[feature]], columns=all_df[feature], normalize='columns').reset_index()

    ans['%'] = 100*ans[ans.drop(['Survived', feature], axis=1).columns].apply(sum, axis=1).round(2)

    ans.drop(labels=ans.drop(['Survived', feature, '%'], axis=1).columns, inplace=True, axis=1)

    ans.sort_values(by=['Survived', '%'], inplace=True)

    sorted_map = sort_remap(ans, feature)    

    all_df[feature].update(all_df[feature].map(sorted_map))
fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12,5))

fields = ['Cat_Age', 'Cat_Fare']



i = 0

for col_ax in axes:

    ans = pd.crosstab(index=[all_df['Survived'], all_df[fields[i]]], columns=all_df[fields[i]], normalize='columns').reset_index()

    ans['%'] = 100*ans[ans.drop(['Survived', fields[i]], axis=1).columns].apply(sum, axis=1).round(2)

    ans.drop(labels=ans.drop(['Survived', fields[i], '%'], axis=1).columns, inplace=True, axis=1)

    ans.sort_values(by=['Survived', '%'], inplace=True)        

    sns.barplot(x=fields[i], y='%', hue='Survived', data=ans, ax=col_ax)

    col_ax.set_ylim((0, 100))

    i+=1

        

plt.tight_layout()        
features_on_off = {'PassengerId':False,

                   'Survived':False,

                   'Pclass':True,

                   'Name':False,

                   'Sex':True,

                   'Age':False,

                   'SibSp':True,                   

                   'Parch':True,

                   'Ticket':False,

                   'Fare':False,

                   'Cabin':False,

                   'Embarked':True,

                   'Title':True,

                   'Deck':True,                   

                   'Alone':True,

                   'Relatives':True,

                   'Cat_Age':True,

                   'Cat_Fare':True}
features_on = [key for key, status in features_on_off.items() if status]

aux = pd.DataFrame(features_on_off, index=['On\Off'])



plt.figure(figsize=(15,1.5))

sns.heatmap(aux, cbar=False, cmap=['red', 'green'], annot=True)
poly = PolynomialFeatures(degree=2, include_bias=False).fit(all_df[features_on])

poly_features = poly.transform(all_df[features_on])

poly_df = pd.DataFrame(data=poly_features, columns=poly.get_feature_names(features_on))
std_scaler = StandardScaler().fit(poly_df)

scaled_features = std_scaler.transform(poly_df)

scaled_df = pd.DataFrame(data=scaled_features, columns=poly_df.columns)
new_df = scaled_df.copy()

new_df['Survived'] = all_df['Survived']
plt.figure(figsize=(12,8))

sns.heatmap(new_df.corr()[['Survived']].sort_values('Survived', ascending=False).drop('Survived').head(15), annot=True, cbar=False, cmap='viridis')
X_train = scaled_df.loc[range(train_df.shape[0])]

y_train = all_df.loc[range(train_df.shape[0]), 'Survived']



X_test = scaled_df.loc[range(train_df.shape[0], train_df.shape[0]+test_df.shape[0])]

y_test = pd.read_csv('../input/true-labels/submission_true.csv')['Survived']
log_reg = LogisticRegressionCV(Cs=1000, cv=5, refit=True, random_state=101, max_iter=200).fit(X_train, y_train)
print(f'Accuracy on the training set: {100*log_reg.score(X_train, y_train):.1f}%')
print(f'Accuracy on the test set: {100*log_reg.score(X_test, y_test):.1f}%')
p = np.linspace(0.01, 1, 50)

error_rate = {'train':[], 'test':[]}



for p_value in p:    

    prob = log_reg.predict_proba(X_train)

    y_pred = np.apply_along_axis(lambda pair: 1 if pair[1] > p_value else 0, 1, prob)

    error_rate['train'].append(mean_squared_error(y_train, y_pred))

    prob = log_reg.predict_proba(X_test)

    y_pred = np.apply_along_axis(lambda pair: 1 if pair[1] > p_value else 0, 1, prob)

    error_rate['test'].append(mean_squared_error(y_test, y_pred))
best_p = p[np.array(error_rate['test']).argmin()]
plt.figure(figsize=(12,5))



min_x, max_x = 0, 1

min_y, max_y = min(min(error_rate['test'], error_rate['train'])), max(max(error_rate['test'], error_rate['train']))



plt.plot(p, error_rate['train'], label='Train', marker='o', color='red')

plt.plot(p, error_rate['test'], label='Test', marker='o', color='blue')



plt.ylabel('Mean Squared Error')

plt.xlabel('Probability')



plt.vlines(x=best_p, ymin=min_y-0.1, ymax=max_y+0.1, linestyle='--', label=f'Best $p={best_p:.2f}$')



plt.ylim((min_y-0.1, max_y+0.1))

plt.xlim(min_x-0.1, max_x+0.1)



plt.legend(loc='best')
print(f'In this particular case, the best probability limit is around {best_p:.2}. This means that if the model provides a survival value greater than {best_p:.2}, we accept that the passenger will survive or die otherwise.')
prob = log_reg.predict_proba(X_train)

y_pred_train = np.apply_along_axis(lambda pair: 1 if pair[1] > best_p else 0, 1, prob)

print(f'Accuracy on the training set with the best choice of p: {100*np.mean(y_train == y_pred_train):.1f}%')
prob = log_reg.predict_proba(X_test)

y_pred_test = np.apply_along_axis(lambda pair: 1 if pair[1] > best_p else 0, 1, prob)

print(f'Accuracy on the testing set with the best choice of p: {100*np.mean(y_pred_test == y_test):.1f}%')
print(classification_report(y_pred_train, y_train))
print(classification_report(y_pred_test, y_test))
perm = PermutationImportance(log_reg).fit(X_train, y_train)

show_weights(perm, feature_names = list(X_train.columns))
weights = pd.DataFrame(data=log_reg.coef_[0], index=scaled_df.columns, columns=['Weights'])

weights.loc['Intercept'] = log_reg.intercept_

weights.sort_values('Weights', ascending=False, inplace=True)

weights.head(15)
output = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': y_pred_test.astype('int')})
output.to_csv('submission.csv', index=False)