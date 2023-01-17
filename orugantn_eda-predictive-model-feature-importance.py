# Import libraries necessary for this project

import numpy as np #math

import pandas as pd



# libraries for visualization

import matplotlib.pyplot as plt

import seaborn as sns



# Igore warning statements

import warnings

warnings.filterwarnings('ignore')



# Pretty display for notebooks

%matplotlib inline
# load data

df = pd.read_csv('../input/S08A1_livestock.csv')



# Success - Display the first record

df.head(n = 1)
# list of all features

print(df.columns.values)
# Check for any null datapoint and its data type

df_inspect = pd.concat([df.isnull().any(), df.dtypes], axis=1, keys=['Check null?', 'data type'])

print(df_inspect)
# Determine percent of survey records has missing values per label



nullLabels = df_inspect[df_inspect['Check null?'] == True].index.tolist() #Filter null labels

n_records = df.shape[0] #Total survey records

print('Total number of records: {}\n'.format(n_records))

percent_null_records = []

for idx in nullLabels:

    percent_null_records.append((df[idx].isnull().sum()/n_records)*100)



df_nulls = pd.DataFrame({'nullLabels':nullLabels, 'percent_null_records':percent_null_records})

df_nulls.sort_values(by = 'percent_null_records', ascending=False, inplace=True)





plt.figure(figsize=(16, 6))

sns.barplot(x = 'nullLabels', y = 'percent_null_records', data = df_nulls, color='salmon')

plt.xlabel('Label with missing data', fontsize = 15)

plt.ylabel('Missing data %', fontsize = 15)

plt.tick_params(axis='both', labelsize = 12)

plt.xticks(rotation = 90)

df_nulls.T
nonnullLabels = df_inspect[df_inspect['Check null?'] == False].index.tolist() #Filter null labels

data = df[nonnullLabels]
data.head(n=1)
print('Province ({}): {} \n'.format(data['PROVINCE'].nunique(), data['PROVINCE'].unique().tolist()))

print('District ({}): {} \n'.format(data['DISTRICT'].nunique(), data['DISTRICT'].unique().tolist()))

print('URB2002 ({}): {} \n'.format(data['URB2002'].nunique(), data['URB2002'].unique().tolist()))

print('QUINTILE ({}): {} \n'.format(data['QUINTILE'].nunique(), data['QUINTILE'].unique().tolist()))

print('POVERTY ({}): {} \n'.format(data['URB2002'].nunique(), data['POVERTY'].unique().tolist()))

print('ITEM ({}): {} \n'.format(data['ITEM'].nunique(), data['ITEM'].unique().tolist()))

print('S8A1Q2 ({}): {} \n'.format(data['S8A1Q2'].nunique(), data['S8A1Q2'].unique().tolist()))
plt.figure(figsize=(10,4))

sns.countplot(x = 'PROVINCE', data = data, hue = 'URB2002', palette='coolwarm')



population = []

percent_rural_province = []



for i in data['PROVINCE'].unique():

    pop = data[(data['PROVINCE'] == i)]['PROVINCE'].count()

    val = data[(data['PROVINCE'] == i) & (data['URB2002'] == 'Rural')]['PROVINCE'].count()/pop

    

    percent_rural_province.append(100*val)

    population.append(pop)



p = pd.DataFrame({'percent_rural':percent_rural_province, 'population':population})

p.set_index(data['PROVINCE'].unique(), inplace=True)

p.sort_values(by = 'percent_rural', ascending=False, inplace=True)

p.T
sns.countplot(x = 'URB2002', data = data)

urb_ratio = data[data['URB2002'] == 'Rural']['URB2002'].count()/n_records

print('Percentage of URB2002 are rural : {:.2f}%'.format(urb_ratio*100))
plt.figure(figsize=(16,4))

sns.countplot(x = 'DISTRICT', data = data, hue = 'URB2002', palette='Set2')

plt.rcParams.update({'font.size': 22})
plt.figure(figsize=(15,6))

sns.countplot(x = 'PROVINCE', data = data, hue = 'ITEM', palette='Set2');
plt.figure(figsize=(15,6))

sns.countplot(x = 'PROVINCE', data = data, hue = 'POVERTY', palette='Set2');



plt.figure(figsize=(15,6))

sns.countplot(x = 'QUINTILE', data = data, hue = 'POVERTY', palette='Set2');
f, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (16,5))

ax1.plot(data['CLUSTER'], data['HH_WT']);

ax1.set_xlabel('Cluster', fontsize = 14)

ax1.set_ylabel('Household weight (HH_WT)', fontsize = 14)

ax1.tick_params(labelsize = 14)



ax2.plot(data['CLUSTER'], data['HH_WT']);

ax2.set_xlabel('Cluster', fontsize = 14)

ax2.set_ylabel('Household weight (HH_WT)', fontsize = 14)

ax2.tick_params(labelsize = 14)

ax2.set_ylim(-1, 250);
plt.figure(figsize = (10,5))

sns.kdeplot(data[data['POVERTY'] == 'Non-poor']['HH_WT'])

sns.kdeplot(data[data['POVERTY'] == 'Poor']['HH_WT'])

sns.kdeplot(data[data['POVERTY'] == 'Extremely poor']['HH_WT'])

plt.title('Household weight distribution by poverty level')

plt.legend(data['POVERTY'].unique(), fontsize = 12);
print('There are {} responses that users responded 9 for "Purchased article in last 12 months (discrete)"'.format(data[data['S8A1Q2'] == 9].count()[0]))



# S8A1Q2 is a yes/no (1/2) question. Response 9 is invalied. 

# It is convenient/reasonable to replace 9 with more common response (2)



data['S8A1Q2'].replace(to_replace=9, value = 2, inplace = True)

sns.countplot(x='S8A1Q2', data = data);
data.head(1)
features = data.drop('POVERTY', axis = 1)

poverty = data['POVERTY']



features_encoded = pd.get_dummies(features)
print("{} total features after one-hot encoding.\n".format(len(features_encoded.columns.values)))

print(features_encoded.columns.values)
# 0: Non-poor | 1 - Poor ! 2 - Extermely poor



poverty_factorize = pd.DataFrame({'poverty': poverty.factorize()[0].tolist()})

poverty_factorize['poverty'].unique()
#Split test train data 80 to 20 ratio

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(features_encoded, 

                                                    poverty_factorize, 

                                                    test_size=0.20, 

                                                    random_state=101)



print('Train sample size: {}'.format(y_train.size))

print('Test sample size: {}'.format(y_test.size))
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, classification_report

from sklearn.model_selection import ShuffleSplit, validation_curve



target_names=data['POVERTY'].unique()



cv = ShuffleSplit(n_splits = 10, test_size = 0.20, train_size= None, random_state=0)

clf = DecisionTreeClassifier()



# Vary the max_depth parameter from 1 to 10

max_depth = np.arange(1,30, 4)



train_scores, test_scores =  validation_curve(clf, X_train, y_train, cv=cv, 

                                              param_name = "max_depth", param_range = max_depth, scoring = 'accuracy')



train_mean = np.mean(train_scores, axis=1)

train_std = np.std(train_scores, axis=1)



test_mean = np.mean(test_scores, axis=1)

test_std = np.std(test_scores, axis=1)



# Plot the validation curve

plt.figure(figsize=(7, 5))

plt.title('Decision Tree Classifier Complexity Performance')

plt.plot(max_depth, train_mean, 'o-', color = 'r', label = 'Training Score')

plt.plot(max_depth, test_mean, 'o-', color = 'g', label = 'Validation Score')

plt.fill_between(max_depth, train_mean - train_std, train_mean + train_std, alpha = 0.15, color = 'r')

plt.fill_between(max_depth, test_mean - test_std, test_mean + test_std, alpha = 0.15, color = 'g')



# Visual aesthetics

plt.legend(loc = 'lower right')

plt.xlabel('Maximum Depth')

plt.ylabel('Score')

plt.ylim([-0.05,1.05])

plt.show()
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import ShuffleSplit



def fit_model(X, y):

    

    cv = ShuffleSplit(n_splits = 10, test_size = 0.20, train_size=None, random_state=0)



    params = {'max_depth': range(1, 30)}

    scoring_fnc = make_scorer(accuracy_score)



    grid = GridSearchCV(DecisionTreeClassifier(), params, cv=cv, scoring=scoring_fnc).fit(X, y)



    return grid.best_estimator_
# Fit the training data to the model using grid search

clf = fit_model(X_train, y_train)



# Produce the value for 'max_depth'

print("Parameter 'max_depth' is {} for the optimal model.\n".format(clf.get_params()['max_depth']))

print('---------------------------------------------------------------------------\n')

print(clf)

print('---------------------------------------------------------------------------\n')
pred = clf.predict(X_test)

print('Model prediction accuracy: {:.2f}%'.format(accuracy_score(y_test, pred)*100))
from sklearn.ensemble import ExtraTreesClassifier



forest = ExtraTreesClassifier(n_estimators=250, random_state=101)



forest.fit(features_encoded, poverty_factorize)



importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



feature_name = []

feature_num = []

importance_val = []



for (i, j) in enumerate(features_encoded.columns.values):

    feature_num.append(indices[i])

    feature_name.append(features_encoded.columns.values[indices[i]])

    importance_val.append(importances[indices[i]]*100)



df_features_importance = pd.DataFrame({'importance_val %': importance_val,

                                       'feature_name': feature_name,

                                        'feature_num':feature_num})



df_features_importance
# Plot the feature importances of the forest

plt.figure(figsize=(15,6))

plt.title("Feature importances with forests of trees")

sns.barplot(df_features_importance['feature_name'], df_features_importance['importance_val %'])

plt.xlabel('Feature importance (%)')

plt.ylabel('Feature name')

plt.xticks(rotation = 90)

plt.show()