import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split
data = pd.read_csv('../input/adult-census-income-data/adult_data.csv')   ##import data

data_descr = pd.read_csv('../input/adult-census-income-data/adult_descr.csv', sep=':')
data.head()
data_descr   ## data descrption.
data_names = data_descr.tail(15)

names = list(data_names.index)



## move the first column on the last position

names.append(names[0])

names = names[1:]
data = pd.read_csv('../input/adult-census-income-data/adult_data.csv', names=names)

data = data.rename(columns={'>50K, <=50K.': 'money'})

data.head()
money_group = data.groupby(by='money').count()

less_than_50k = str(np.round(money_group.age[0]/len(data) * 100, 2))+' %'

greater_than_50k = str(np.round(money_group.age[1]/len(data) * 100, 2))+' %'
print('Due to the data', less_than_50k, 'of people earn less than 50K, and only', greater_than_50k, 'do more than 50K.')
age_count = data.groupby(by='age').count()

age_unique_list = list(age_count.index)
sns.jointplot(x=age_unique_list, y=age_count.values[:, 0], height=10, marginal_kws=dict(bins=len(age_count), rug=True), kind='scatter')
educ_money = data[['education', 'money']]

less = educ_money[educ_money['money']==' <=50K'].groupby('education', as_index=False).count()

more = educ_money[educ_money['money']==' >50K'].groupby('education', as_index=False).count()

merge = more.merge(less, on='education', how='right', sort=True)

merge = merge.rename(columns={'education': 'Education', 'money_x': 'More than 50K', 'money_y': 'Less than 50K'})

#merge
merge_melted = merge.melt(id_vars='Education').rename(columns=str.title)

#merge_melted
plt.figure(figsize=(20, 8))

sns.barplot(x='Education', y='Value', data=merge_melted, hue='Variable', palette='vlag')
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 10))

f.suptitle('Capital - Marital Status - Race', fontsize=20)

sns.violinplot(data=data,

               x="marital-status",

               y="age",

               inner="quartile",

               hue='money',

               scale="count",

               split=True, palette='vlag', linewidth=.8, ax=ax1, cut=2)

sns.violinplot(data=data,

               x="race",

               y="age",

               inner="quartile",

               hue='money',

               scale="count",

               split=True, palette='vlag', linewidth=.8, ax=ax2)
f, ax = plt.subplots(1, figsize=(26, 10))

f.suptitle('Capital - Relationship', fontsize=20)

sns.violinplot(data=data,

               x="relationship",

               y="age",

               inner="quartile",

               hue='money',

               scale="count",

               split=True, palette='vlag', linewidth=.8, ax=ax, cut=2)
train = data.drop(columns=['relationship', 'education', 'education-num', 'marital-status', 'workclass'])

labels = data['money']
train.head()
train['Relationship_Num'] = data['relationship'].map(dict(zip(list(np.unique(data['relationship'])), [0, 1, 2, 3, 4, 0])))

train['Education_Num'] = data['education'].map(dict(zip(list(np.unique(data['education'])), [0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 3, 0, 4, 0, 5, 0])))

train['Marital_Status_Num'] = data['marital-status'].map(dict(zip(list(np.unique(data['marital-status'])), [0, 1, 1, 1, 2, 0, 3])))

train['Workclass_Num'] = data['workclass'].map(dict(zip(list(np.unique(data['workclass'])), [np.nan, 0, 0, 1, 2, 2, 2, 0, 1])))

train['sex'] = data['sex'].map(dict(zip(list(np.unique(data['sex'])), [0, 1])))  #1 - male, 0 - female

train['race'] = data['race'].map(dict(zip(list(np.unique(data['race'])), list(range(len(data.groupby(by=['race']).count().iloc[:,0]))))))

train['Occupation_Num'] = data['occupation'].map(dict(zip(list(np.unique(data['occupation'])), list(range(len(data.groupby(by=['occupation']).count().iloc[:,0]))))))

train['Occupation_Num'] = train['Occupation_Num'].replace({0: np.nan})
train['HrsBand'] = pd.cut(data['hours-per-week'], 5)

train['HrsBand_Num'] = pd.cut(data['hours-per-week'], 5, labels=list(range(len(train.groupby(by=['HrsBand']).count().iloc[:,0]))))

train['AgeBand'] = pd.cut(data['age'], 5)

train['Age_Num'] = pd.cut(data['age'], 5, labels=list(range(len(train.groupby(by=['AgeBand']).count().iloc[:,0]))))
countries= np.unique(train['native-country'])

countries = countries.tolist()
indexes = ['O', 'A', 'Am-N', 'A', 'Am-S', 'Am-M', 'Am-M', 'Am-S', 'Am-M', 'E', 'E', 'E', 'E', 'Am-M', 'Am-M', 'E', 'Am-M', 'A', 'E', 'A', 'A', 'E', 'E', 'Am-M', 'A', 'A', 'Am-M', 'Am-M', 'O', 'Am-S', 'A', 'E', 'E', 'Am-M', 'E', 'O', 'A', 'A', 'Am-S', 'Am-N', 'A', 'E']
## quick check that we did not miss anything

print(len(countries)==len(indexes))   
## create dictionary that assign each country to its continent

native_countries_dictionary = {a: b for a, b in zip(countries, indexes)}   



## crate new column with assigned continents

train['Continents'] = train['native-country'].map(native_countries_dictionary)  



## create new column with numerical values - each number represents assinged continent

train['Continents_Num'] = train['Continents'].map(dict(zip(list(np.unique(train['Continents'])), list(range(len(train.groupby(by=['Continents']).count().iloc[:,0]))))))    
## we delete all non-numerical columns

train_drop = train.drop(columns=['occupation', 'native-country', 'Continents', 'AgeBand', 'age', 'hours-per-week', 'HrsBand'])



## check length of the data before deleting NaN's

len(train_drop)
## drop NaN's and check the lenght again

train_drop_na = train_drop.dropna()

len(train_drop_na)
## change money features into numerical values: 0 - less than 50K, 1 - more than 50K

train_drop_na.money = train_drop_na.money.map(dict(zip(list(np.unique(labels)), [0, 1])))  
## change numerical values to int32 type

train_drop_na = train_drop_na.astype('int32')
train_drop_na.head()
male_money = train_drop_na[train_drop_na.sex==1].groupby(by='money').count()

female_money = train_drop_na[train_drop_na.sex==0].groupby(by='money').count()



male_percent_low = np.round(male_money.sex.values[0]/male_money.sex.sum(),2)

male_percent_high = np.round(male_money.sex.values[1]/male_money.sex.sum(),2)

female_percent_low = np.round(female_money.sex.values[0]/female_money.sex.sum()*100,2)

female_percent_high = np.round(female_money.sex.values[1]/female_money.sex.sum()*100,2)



title = str(male_percent_high) + '% of males gain more than 50K, ' + str(male_percent_low) + '& males gain less. For women ' + str(female_percent_low) + '% of them gain less than 50K, ' + str(female_percent_high) + '% gain more.'

print(title)
best_attributies = train_drop_na[train_drop_na.sex==0]

best_attributies.shape
## take only wifes and husbands - they are assigned to 0 in column Relationship_Num

best_attributies = best_attributies[best_attributies.Relationship_Num==0]



## take only Masters - they are assigned to 4

best_attributies = best_attributies[best_attributies.Education_Num==4]
female_money = best_attributies.groupby(by='money').count()



if len(female_money.sex.values)==2:

    female_percent_low = np.round(female_money.sex.values[0]/female_money.sex.sum() * 100,2)

    female_percent_high = np.round(female_money.sex.values[1]/female_money.sex.sum() *100 ,2)

    title = str(female_percent_low) + '% of women gain less than 50K, ' + str(female_percent_high) + '% gain more.'

    print(title)

else:

    print('100% of women gain more than 50K/year')
f, (ax) = plt.subplots(1, figsize=(20, 10))

f.suptitle('Capital - Race', fontsize=20)

sns.violinplot(data=best_attributies,

               x="Workclass_Num",

               y="Age_Num",

               inner="quartile",

               hue='money',

               scale="area",

               split=True, palette='vlag', linewidth=.8, ax=ax)
print(np.unique(data.workclass)[2])
best_attributies = best_attributies[best_attributies.Workclass_Num==2]
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 10))

f.suptitle('Capital - Occupation - Race', fontsize=20)

sns.violinplot(data=best_attributies,

               x="Occupation_Num",

               y="Age_Num",

               inner="quartile",

               hue='money',

               scale="count",

               split=True, palette='vlag', linewidth=.8, ax=ax1, cut=2)

sns.violinplot(data=best_attributies,

               x="race",

               y="Age_Num",

               inner="quartile",

               hue='money',

               scale="count",

               split=True, palette='vlag', linewidth=.8, ax=ax2)
print(np.unique(data.occupation)[10])

best_attributies = best_attributies[best_attributies.Occupation_Num==10]
female_money = best_attributies.groupby(by='money').count()



if len(female_money.sex.values)==2:

    female_percent_low = np.round(female_money.sex.values[0]/female_money.sex.sum() * 100,2)

    female_percent_high = np.round(female_money.sex.values[1]/female_money.sex.sum() *100 ,2)

    title = str(female_percent_low) + '% of women gain less than 50K, ' + str(female_percent_high) + '% gain more.'

    print(title)

else:

    print('100% of women gain more than 50K/year')
f, (ax) = plt.subplots(1, figsize=(26, 10))

f.suptitle('Capital - Race', fontsize=20)

sns.violinplot(data=best_attributies,

               x="race",

               y="Age_Num",

               inner="quartile",

               hue='money',

               scale="count",

               split=True, palette='vlag', linewidth=.8, ax=ax)
print(np.unique(data.race)[4])

best_attributies = best_attributies[best_attributies.race==4]
female_money = best_attributies.groupby(by='money').count()



if len(female_money.sex.values)==2:

    female_percent_low = np.round(female_money.sex.values[0]/female_money.sex.sum() * 100,2)

    female_percent_high = np.round(female_money.sex.values[1]/female_money.sex.sum() *100 ,2)

    title = str(female_percent_low) + '% of women gain less than 50K, ' + str(female_percent_high) + '% gain more.'

    print(title)

else:

    print('100% of women gain more than 50K/year')
f, (ax) = plt.subplots(1, figsize=(26, 10))

f.suptitle('Capital - Hrs/Week', fontsize=20)

sns.violinplot(data=best_attributies,

               x="HrsBand_Num",

               y="Age_Num",

               inner="quartile",

               hue='money',

               scale="count",

               split=True, palette='vlag', linewidth=.8, ax=ax)
print(np.unique(train.AgeBand)[1])

best_attributies = best_attributies[best_attributies.HrsBand_Num==1]
female_money = best_attributies.groupby(by='money').count()



if len(female_money.sex.values)==2:

    female_percent_low = np.round(female_money.sex.values[0]/female_money.sex.sum() * 100,2)

    female_percent_high = np.round(female_money.sex.values[1]/female_money.sex.sum() *100 ,2)

    title = str(female_percent_low) + '% of women gain less than 50K, ' + str(female_percent_high) + '% gain more.'

    print(title)

else:

    print('100% of women gain more than 50K/year')
## standarization for fnlwgt column

fnlwgt_norm = (train_drop_na.fnlwgt - train_drop_na.fnlwgt.mean())/train_drop_na.fnlwgt.std()

train_drop_na['fnlwgt'] = train_drop_na['fnlwgt'].map(dict(zip(list(train_drop_na.fnlwgt), fnlwgt_norm)))
## standarization for capital-gain column

capital_gain_norm = (train_drop_na['capital-gain'] - train_drop_na['capital-gain'].mean())/train_drop_na['capital-gain'].std()

train_drop_na['capital-gain'] = train_drop_na['capital-gain'].map(dict(zip(list(train_drop_na['capital-gain']), capital_gain_norm)))
## standarization for capital-loss column

capital_loss_norm = (train_drop_na['capital-loss'] - train_drop_na['capital-loss'].mean())/train_drop_na['capital-loss'].std()

train_drop_na['capital-loss'] = train_drop_na['capital-loss'].map(dict(zip(list(train_drop_na['capital-loss']), capital_gain_norm)))
train_drop_na.head()
## splitting features and labels

train_only = train_drop_na.drop(columns=['money'])

labels_only = train_drop_na['money']
corr = train_only.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



#cmap = sns.diverging_palette(220, 10, as_cmap=True)



f, ax = plt.subplots(figsize=(11, 9))

sns.heatmap(corr, cbar=True, annot=True, mask=mask, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
## create a dictionary with integers and assined race features

race_dict = dict(zip(list(range(len(data.groupby(by=['race']).count().iloc[:,0]))), list(np.unique(data['race']))))

relationship_dict = dict(zip(list(range(len(train_drop_na.groupby(by=['Relationship_Num']).count().iloc[:,0]))), list(np.unique(data['relationship']))))

education_dict = dict(zip(list([0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 3, 0, 4, 0, 5, 0]), list(np.unique(data['education'])) ))

marital_dict = dict(zip(list([0, 1, 1, 1, 2, 0, 3]), list(np.unique(data['marital-status'])) ))

workclass_dict = dict(zip(list([0, 0, 1, 2, 2, 2, 0, 1]), list(np.unique(data['workclass'])) ))

occupation_dict = dict(zip(list(range(len(data.groupby(by=['occupation']).count()))), list(np.unique(data['occupation']))))

continents_dict = dict(zip(list(range(len(train_drop_na.groupby(by=['Continents_Num']).count().iloc[:,0]))), list(np.unique(train['Continents']))))

print(race_dict)

print(relationship_dict)

print(education_dict)

print(marital_dict)

print(workclass_dict)

print(occupation_dict)

print(continents_dict)
## race

race_dummy = pd.get_dummies(train_drop_na.race)

race_dummy = race_dummy.rename(columns=race_dict)

## relationship

relationship_dummy = pd.get_dummies(train_drop_na.Relationship_Num)

relationship_dummy = relationship_dummy.rename(columns=relationship_dict)

## education

education_dummy = pd.get_dummies(train_drop_na.Education_Num)

education_dummy = education_dummy.rename(columns=education_dict)

## marital-status

marital_dummy = pd.get_dummies(train_drop_na.Marital_Status_Num)

marital_dummy = marital_dummy.rename(columns=marital_dict)

## workclass

workclass_dummy = pd.get_dummies(train_drop_na.Workclass_Num)

workclass_dummy = workclass_dummy.rename(columns=workclass_dict)

## occupation

occupation_dummy = pd.get_dummies(train_drop_na.Occupation_Num)

occupation_dummy = occupation_dummy.rename(columns=occupation_dict)

## continents

continents_dummy = pd.get_dummies(train_drop_na.Continents_Num)

continents_dummy = continents_dummy.rename(columns=continents_dict)
concat_list = [train_only, race_dummy, relationship_dummy, education_dummy, marital_dummy, workclass_dummy, occupation_dummy, continents_dummy]
train_with_dummies = pd.concat(concat_list, axis=1)

train_with_dummies = train_with_dummies.drop(columns=['race', 'Relationship_Num', 'Education_Num', 'Marital_Status_Num', 'Workclass_Num', 'Occupation_Num', 'Continents_Num'])
train_with_dummies.head()
X_train, X_val, y_train, y_val = train_test_split(train_with_dummies, labels_only, test_size=0.2)
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score

from sklearn.ensemble import AdaBoostClassifier

import xgboost as xgb

from sklearn.svm import SVC, LinearSVC

from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV
classificator_names, acc_values = [], []
## we start with basic classification algorithm that is logistic regression

classificator_names.append('Logistic Regression')



logreg_clf = LogisticRegression(random_state=42)

logreg_clf.fit(X_train, y_train)

acc_logreg = round(logreg_clf.score(X_val, y_val) * 100, 2)

print (acc_logreg)

acc_values.append(acc_logreg)
## next try is taking Logistic Regression model with Stochastic Gradient Decent optimization

classificator_names.append('Logistic Regression with Stochastic Gradient Decent Optimizer')



sgd_log_clf = SGDClassifier(loss='log', random_state=42)

sgd_log_clf.fit(X_train, y_train)

acc_log_sgd = round(sgd_log_clf.score(X_val, y_val) * 100, 2)

print (acc_log_sgd)

acc_values.append(acc_log_sgd)
## DecisionTreeClassifier with default hyperparameters

classificator_names.append('Decision Tree')



dec_tree_clf = DecisionTreeClassifier(random_state=42)

dec_tree_clf.fit(X_train, y_train)

acc_decision_tree = round(dec_tree_clf.score(X_val, y_val) * 100, 2)

print (acc_decision_tree)

acc_values.append(acc_decision_tree)
## using GridSearch model we can find best hyperparameters:

classificator_names.append('Tuned Decision Tree')



tuned_dec_tree_clf = DecisionTreeClassifier(random_state=42)

grid_values = {'max_depth': [2, 4, 8, 10, 12], 'max_leaf_nodes': [2, 6, 8, 10, 12]}

grid_clf_acc = GridSearchCV(tuned_dec_tree_clf, param_grid = grid_values, scoring = 'accuracy', cv=5)

grid_clf_acc.fit(X_train, y_train)

acc_grid_decision_tree = round(grid_clf_acc.score(X_val, y_val) * 100, 2)

print('The accuracy using hyperparameters:', grid_clf_acc.best_params_, 'is equal to:', acc_grid_decision_tree)

acc_values.append(acc_grid_decision_tree)
## RandomForestClassifier with default hyperparameters

classificator_names.append('Random Forest')



rand_for_clf = RandomForestClassifier(random_state=42, n_estimators=100)

rand_for_clf.fit(X_train, y_train)

acc_random_forest = round(rand_for_clf.score(X_val, y_val) * 100, 2)

print (acc_random_forest)

acc_values.append(acc_random_forest)
## again we are looking for best parameters

classificator_names.append('Tuned Random Forest')



tuned_rand_for_clf = RandomForestClassifier(random_state=42, n_estimators=100)

grid_values = {'max_depth': [5, 10, 15], 'max_leaf_nodes': [8, 12, 16]}

grid_clf_acc = GridSearchCV(tuned_rand_for_clf, param_grid = grid_values, scoring = 'accuracy', cv=5)

grid_clf_acc.fit(X_train, y_train)

acc_grid_random_forest = round(grid_clf_acc.score(X_val, y_val) * 100, 2)

print('The accuracy using hyperparameters:', grid_clf_acc.best_params_, 'is equal to:', acc_grid_random_forest)

acc_values.append(acc_grid_random_forest)
## Support Vector Machine with defauls hyperparamteres

classificator_names.append('Support Vector Machine')



svm_clf = SVC(random_state=42, gamma='scale')

svm_clf.fit(X_train, y_train)

acc_svm = round(svm_clf.score(X_val, y_val) * 100, 2)

print(acc_svm)

acc_values.append(acc_svm)
## SVM with GridSearch tuning

classificator_names.append('Tuned Support Vector Machine')



tuned_svm_clf = SVC(kernel='rbf', random_state=42, gamma='scale')

grid_values = {'C': [1, 10, 100]}

grid_clf_acc = GridSearchCV(tuned_svm_clf, param_grid = grid_values, scoring = 'accuracy', cv=5)

grid_clf_acc.fit(X_train, y_train)

acc_grid_svm = round(grid_clf_acc.score(X_val, y_val) * 100, 2)

print('The accuracy using hyperparameters:', grid_clf_acc.best_params_, 'is equal to:', acc_grid_svm)

acc_values.append(acc_grid_svm)
## SVM with SGD Optimizer

classificator_names.append('Support Vector Machine with SGD Optimizer')



sgd_svm_clf = SGDClassifier(loss='hinge', random_state=42)

sgd_svm_clf.fit(X_train, y_train)

acc_svm_sgd = round(sgd_svm_clf.score(X_val, y_val) * 100, 2)

print (acc_svm_sgd)

acc_values.append(acc_svm_sgd)
## K-Nearest Neighbors

classificator_names.append('K-Nearest Neighbors')



knn_clf = KNeighborsClassifier(algorithm='auto', n_neighbors=5)

knn_clf.fit(X_train, y_train)

acc_knn = round(knn_clf.score(X_val, y_val) * 100, 2)

print(acc_knn)

acc_values.append(acc_knn)
## Gaussian Naive Bayes

classificator_names.append('Gaussian Naive Bayes')



gauss_clf = GaussianNB()

gauss_clf.fit(X_train, y_train)   

acc_gauss = round(gauss_clf.score(X_val, y_val) * 100, 2)

print (acc_gauss)

acc_values.append(acc_gauss)
classifiers = pd.DataFrame({

    'Model': classificator_names, 'Score': acc_values

})

classifiers = classifiers.sort_values(by='Score', ascending=False).reset_index(drop=True)

classifiers
boosting_list, boosting_acc = [], []
boosting_list.append('Gradient Boosting Classifier')



gbrt = GradientBoostingClassifier(max_depth = 2, n_estimators = 150, learning_rate = .2)

gbrt.fit(X_train, y_train)

acc_gbrt = round(gbrt.score(X_val, y_val) * 100, 2)

print(acc_gbrt)

boosting_acc.append(acc_gbrt)
boosting_list.append('AdaBoost Classifier')



ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 2, max_leaf_nodes=9), n_estimators = 150, algorithm = 'SAMME.R', learning_rate = 0.2)

ada_clf.fit(X_train, y_train)

acc_ada = round(ada_clf.score(X_val, y_val) * 100, 2)

print(acc_ada)

boosting_acc.append(acc_ada)
boosting_list.append('XGBoost Classifier')



model = xgb.XGBClassifier(objective='binary:logistic', learning_rate= 1, colsample_bynode= 0.8, subsample= 0.8, num_parallel_tree=100, eval_metric = 'auc')

model.fit(X_train, y_train)

acc_xgb = round(model.score(X_val, y_val) * 100, 2)

print(acc_xgb)

boosting_acc.append(acc_xgb)
## for XGBoost we can create plot of most important features

f, ax = plt.subplots(1, figsize=(26, 10))

xgb.plot_importance(model, importance_type='weight', ax=ax)
boosting_list.append('Bagging Classifier')



bag_clf = BaggingClassifier(DecisionTreeClassifier(splitter = 'random', max_leaf_nodes = 16), n_estimators = 500, bootstrap = True, n_jobs = -1)

bag_clf.fit(X_train, y_train)

acc_bag_clf = round(bag_clf.score(X_val, y_val) * 100, 2)

print(acc_bag_clf)

boosting_acc.append(acc_bag_clf)
boosting_classifiers = pd.DataFrame({

    'Model': boosting_list, 'Score': boosting_acc

})

boosting_classifiers = boosting_classifiers.sort_values(by='Score', ascending=False).reset_index(drop=True)

boosting_classifiers
test_data = pd.read_csv('../input/adult-census-income-data/adult_test.csv', names=names)

test_data = test_data.rename(columns={'>50K, <=50K.': 'money'})

test_data = test_data.iloc[1:]
train_on_test = test_data.drop(columns=['relationship', 'education', 'education-num', 'marital-status', 'workclass'])

test_labels = test_data['money']
train_on_test['Relationship_Num'] = test_data['relationship'].map(dict(zip(list(np.unique(test_data['relationship'])), [0, 1, 2, 3, 4, 0])))

train_on_test['Education_Num'] = test_data['education'].map(dict(zip(list(np.unique(test_data['education'])), [0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 3, 0, 4, 0, 5, 0])))

train_on_test['Marital_Status_Num'] = test_data['marital-status'].map(dict(zip(list(np.unique(test_data['marital-status'])), [0, 1, 1, 1, 2, 0, 3])))

train_on_test['Workclass_Num'] = test_data['workclass'].map(dict(zip(list(np.unique(test_data['workclass'])), [np.nan, 0, 0, 1, 2, 2, 2, 0, 1])))

train_on_test['sex'] = test_data['sex'].map(dict(zip(list(np.unique(test_data['sex'])), [0, 1])))  #0 - male, 1 - female

train_on_test['race'] = test_data['race'].map(dict(zip(list(np.unique(test_data['race'])), list(range(len(test_data.groupby(by=['race']).count().iloc[:,0]))))))

train_on_test['Occupation_Num'] = test_data['occupation'].map(dict(zip(list(np.unique(test_data['occupation'])), list(range(len(test_data.groupby(by=['occupation']).count().iloc[:,0]))))))

train_on_test['Occupation_Num'] = train_on_test['Occupation_Num'].replace({0: np.nan})

train_on_test['HrsBand'] = pd.cut(test_data['hours-per-week'], 5)

train_on_test['HrsBand_Num'] = pd.cut(test_data['hours-per-week'], 5, labels=list(range(len(train_on_test.groupby(by=['HrsBand']).count().iloc[:,0]))))

train_on_test['AgeBand'] = pd.cut(test_data.age.astype('int32'), 5)  ## in the test file values of age column have string type

train_on_test['Age_Num'] = pd.cut(test_data.age.astype('int32'), 5, labels=list(range(len(train_on_test.groupby(by=['AgeBand']).count().iloc[:,0]))))
train_on_test['Continents'] = train_on_test['native-country'].map(native_countries_dictionary)  

train_on_test['Continents_Num'] = train_on_test['Continents'].map(dict(zip(list(np.unique(train_on_test['Continents'])), list(range(len(train_on_test.groupby(by=['Continents']).count().iloc[:,0]))))))    
train_on_test_drop = train_on_test.drop(columns=['occupation', 'native-country', 'Continents', 'AgeBand', 'age', 'hours-per-week', 'HrsBand'])
train_on_test_drop_na = train_on_test_drop.dropna()

print('There is', len(train_on_test_drop) - len(train_on_test_drop_na),"Nan rows in the test dataset, it's", np.round((len(train_on_test_drop) - len(train_on_test_drop_na))/len(train_on_test_drop) * 100, 2),'% that needs to be deleted.')
train_on_test_drop_na.money = train_on_test_drop_na.money.map(dict(zip(list(np.unique(test_labels)), [0, 1])))

train_on_test_drop_na = train_on_test_drop_na.astype('int32')
## standarization for fnlwgt column

fnlwgt_norm = (train_on_test_drop_na.fnlwgt - train_on_test_drop_na.fnlwgt.mean())/train_on_test_drop_na.fnlwgt.std()

train_on_test_drop_na['fnlwgt'] = train_on_test_drop_na['fnlwgt'].map(dict(zip(list(train_on_test_drop_na.fnlwgt), fnlwgt_norm)))

## standarization for capital-gain column

capital_gain_norm = (train_on_test_drop_na['capital-gain'] - train_on_test_drop_na['capital-gain'].mean())/train_on_test_drop_na['capital-gain'].std()

train_on_test_drop_na['capital-gain'] = train_on_test_drop_na['capital-gain'].map(dict(zip(list(train_on_test_drop_na['capital-gain']), capital_gain_norm)))

## standarization for capital-loss column

capital_loss_norm = (train_on_test_drop_na['capital-loss'] - train_on_test_drop_na['capital-loss'].mean())/train_on_test_drop_na['capital-loss'].std()

train_on_test_drop_na['capital-loss'] = train_on_test_drop_na['capital-loss'].map(dict(zip(list(train_on_test_drop_na['capital-loss']), capital_gain_norm)))
## race

test_race_dummy = pd.get_dummies(train_on_test_drop_na.race)

test_race_dummy = test_race_dummy.rename(columns=race_dict)

## relationship

test_relationship_dummy = pd.get_dummies(train_on_test_drop_na.Relationship_Num)

test_relationship_dummy = test_relationship_dummy.rename(columns=relationship_dict)

## education

test_education_dummy = pd.get_dummies(train_on_test_drop_na.Education_Num)

test_education_dummy = test_education_dummy.rename(columns=education_dict)

## marital-status

test_marital_dummy = pd.get_dummies(train_on_test_drop_na.Marital_Status_Num)

test_marital_dummy = test_marital_dummy.rename(columns=marital_dict)

## workclass

test_workclass_dummy = pd.get_dummies(train_on_test_drop_na.Workclass_Num)

test_workclass_dummy = test_workclass_dummy.rename(columns=workclass_dict)

## occupation

test_occupation_dummy = pd.get_dummies(train_on_test_drop_na.Occupation_Num)

test_occupation_dummy = test_occupation_dummy.rename(columns=occupation_dict)

## continents

test_continents_dummy = pd.get_dummies(train_on_test_drop_na.Continents_Num)

test_continents_dummy = test_continents_dummy.rename(columns=continents_dict)
train_on_test_only = train_on_test_drop_na.drop(columns=['money'])

test_labels_only = train_on_test_drop_na['money']
test_concat_list = [train_on_test_only, test_race_dummy, test_relationship_dummy, test_education_dummy, test_marital_dummy, test_workclass_dummy, test_occupation_dummy, test_continents_dummy]
train_on_test_with_dummies = pd.concat(test_concat_list, axis=1)

train_on_test_with_dummies = train_on_test_with_dummies.drop(columns=['race', 'Relationship_Num', 'Education_Num', 'Marital_Status_Num', 'Workclass_Num', 'Occupation_Num', 'Continents_Num'])
print(len(train_on_test_with_dummies.columns)==len(train_with_dummies.columns)) ## check if number of columns in train and test datasets are same
print('We can see that boosting classifiers work better during the training than the previous ones, the best boosting classifier is', 

      boosting_classifiers.values[0][0],'while the worst regular one is',classifiers.values[len(classifiers)-1][0],'. Let us check how they behave on the test one.')
test_acc_ada = round(ada_clf.score(train_on_test_with_dummies, test_labels_only) * 100, 2)

print('Accuracy for AdaBoost Classifier on the test set is:', test_acc_ada)

test_acc_gauss = round(gauss_clf.score(train_on_test_with_dummies, test_labels_only) * 100, 2)

print ('Accuracy for Gaussian Naive Bayes on the test set is:', test_acc_gauss)
test_acc_gbrt = round(gbrt.score(train_on_test_with_dummies, test_labels_only) * 100, 2)

print('Best accuracy using XGBoost model:',test_acc_gbrt)