import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

%matplotlib inline
#Lets load the training and testing data

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#Will merge training and testing data so that it will be easier in future if any transformation or scaling etc is applied, we dont miss out on one another

all_data = pd.concat([train, test], axis=0).reset_index(drop=True)
#Just checking how many columns and column names

print (train.columns)
print (test.columns)
#Just checking how the datasets are distributed

print (train.describe())
print ('--'*40)
print (train.describe(include=['O']))
#Are there any nulls in the data and if so what is the percent of nulls in respective column

nulls_df = pd.concat([all_data.isnull().sum(), train.isnull().sum(), train.isnull().sum()/train.shape[0]*100], axis=1)
nulls_df.columns = ['all_data nulls', 'train nulls', '% of train nulls']
print (nulls_df)
#Before we do anything with data, the columns must be in proper datatypes

all_data['Age'] = pd.to_numeric(all_data['Age'], errors='coerce')
all_data['Fare'] = pd.to_numeric(all_data['Fare'], errors='coerce')
all_data['Survived'] = pd.to_numeric(all_data['Survived'], errors='coerce')

#Why reverse order of categories? coz we need 3<2<1 so specify in that order
all_data['Pclass'] = pd.Categorical(all_data['Pclass'], ordered = True, categories=[3,2,1])

all_data.info()
#Replace all cabin values with 1st char, if its unknown then set to "?"

for index, val in all_data['Cabin'].iteritems():
    if not pd.isnull(val):
        all_data.loc[index, 'Cabin'] = val[0]
    else:
        all_data.loc[index, 'Cabin'] = '?'
#Fill NaN with most occurring value
all_data['Embarked'].fillna(all_data['Embarked'].mode()[0], inplace=True)
#Fill NaN mean value
all_data['Fare'].fillna(all_data['Fare'].mean(), inplace=True)
'''
For filling Age values, we can use multiple ways like Mode, Mean, ffill or bfill.
But better approach would be to predict Age based on other params.
So will create a linear regression model to predict the missing Age
'''

X_age_independent = all_data[['Parch', 'SibSp', 'Pclass', 'Age']]
X_age_independent.dropna(axis=0, inplace=True)

y_age_dependent = X_age_independent[['Age']]
X_age_independent.drop(['Age'], axis=1, inplace=True)

#Generate the model
lin_reg_age = LinearRegression()
lin_reg_age = lin_reg_age.fit(X_age_independent, y_age_dependent)
age_nulls_df = all_data[['Parch', 'SibSp', 'Pclass']][all_data["Age"].isnull()]

#Filling NaN in Age column
if age_nulls_df.shape[0] != 0:
    pred_age = lin_reg_age.predict(age_nulls_df)
    all_data['Age'][all_data['Age'].isnull()] = pred_age

#for null_age_index in all_data["Age"][all_data["Age"].isnull()].index:
#    all_data.iloc[null_age_index]['Age'] = lin_reg_age.predict(all_data.iloc[null_age_index][['Parch', 'SibSp', 'Pclass']].reshape(1, -1))
#Keep only 1st char of ticket

ticket = []
for tkt in all_data['Ticket']:
    tmp_tkt = tkt.replace("/", "").replace(".", "")[0]
    tmp_tkt = '?' if tmp_tkt.isdigit() else tmp_tkt
    ticket.append(tmp_tkt)

all_data['Ticket'] = ticket
#Lets checout the correlation among the variables

fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(train.corr(), annot=True, ax=ax)
#Relation between Gender and Survival
tmp_df = train[['Sex', 'Survived']]
tmp_df.groupby('Sex', as_index=False).mean()
#Relation between Pclass and Survival
tmp_df = train[['Pclass', 'Survived']]
tmp_df.groupby('Pclass', as_index=False).mean()
#Relation between Sibling/Spouse and Survival
tmp_df = train[['SibSp', 'Survived']]
tmp_df.groupby('SibSp', as_index=False).mean()
#Relation between Parent/Child and Survival
tmp_df = train[['Parch', 'Survived']]
tmp_df.groupby('Parch', as_index=False).mean()
#How fare is distributed between gender and survival rate
g = sns.FacetGrid(train, col="Survived")
g.map(sns.barplot, 'Sex', 'Fare')
#Just checking outliers in the fare data

fig, ax = plt.subplots(figsize=(10,5))
sns.boxplot(x='Sex', y='Fare', hue='Survived', data=all_data, orient='v', ax=ax)
#sns.boxplot(all_data['Fare'], orient='v')

#all_data[all_data['Fare'] > 400]
#all_data.drop(all_data[all_data['Fare'] > 400].index, axis=0, inplace=True) 
#fig, ax = plt.subplots(figsize=(10,5))
print ('Skew of fare distribution', train['Fare'].skew())
#sns.kdeplot(all_data['Fare'])
sns.distplot(all_data['Fare'], rug=True, hist=False)
#Normalize the data to gaussian curve
all_data['Fare'] = np.log(all_data['Fare'])
sns.distplot(all_data['Fare'], rug=True, hist=False)
#Break fare into 4 groups
all_data['FareGroup'] = pd.qcut(all_data['Fare'], 4, labels=[0, 1, 2, 3])
#K-Kid, Y-Young, M-Mid, O-Old

all_data['AgeGroup'] = pd.cut(all_data['Age'], 4, labels=['K', 'Y', 'M', 'O'])
all_data['AgeGroup'].value_counts()
#Break the family into 3 groups
all_data['FamilySize'] = all_data['Parch'] + all_data['SibSp'] + 1
all_data['FamilyType'] = pd.cut(all_data['FamilySize'], 3 , labels=['S', 'M', 'L'])
all_data.isnull().sum()
all_data.head(3)
all_data.info()
#Work on the copy of cleaned and transformed data

final_data = all_data.copy()

#Create dummies for categorical columns

final_data = final_data.join(pd.get_dummies(final_data[['Cabin', 'Embarked', 'Sex', 'Ticket', 'Pclass', 'FareGroup', 'AgeGroup', 'FamilyType']]))
final_data.shape
final_data.columns
y_train = train['Survived']
all_PassengerId = final_data['PassengerId']

#Drop unwantedcolumns and also columns which were used for dummy creation as they are not needed any more
final_data.drop(['Age', 'Cabin', 'Embarked', 'Fare', 'Name', 'Parch', 'PassengerId', 'Pclass', 'Sex', 'SibSp', 'Survived', 'Ticket', 'FareGroup', 'AgeGroup', 'FamilySize', 'FamilyType'], axis=1, inplace=True)
final_data.columns
#Divide the data back into training and testing from cleaned and transformed data

X_train = final_data[:len(train)]
X_test = final_data[len(train):]
X_test_PassengerId = all_PassengerId[len(train):]

print (X_train.shape)
print (X_test.shape)
#split = StratifiedShuffleSplit(n_splits=5)
split = StratifiedKFold(n_splits=5)

rf = RandomForestClassifier()
dec_tree = DecisionTreeClassifier()
nb = GaussianNB()
knn = KNeighborsClassifier()
gboost = GradientBoostingClassifier()
adaboost = AdaBoostClassifier()

classifiers = {
    'rf' : rf,
    'dec_tree' : dec_tree,
    #'nb' : nb,
    'knn' : knn,
    'gradient': gboost,
    'ada': adaboost
}
cv_mean_scores = []
X_train_2nd = pd.DataFrame()

for cls_name, classifier in classifiers.items():
    scores = cross_val_score(classifier, X_train, y_train, cv=split)
    cv_mean_scores.append(scores.mean() * 100)
    
    classifier.fit(X_train, y_train)
    X_train_2nd[cls_name] = classifier.predict(X_train)

sns.barplot(list(classifiers.keys()), cv_mean_scores)
adaboost_model_2nd = AdaBoostClassifier(n_estimators=50, learning_rate=0.75)
adaboost_model_2nd = adaboost_model_2nd.fit(X_train_2nd, y_train)
cv_score_2nd = cross_val_score(adaboost_model_2nd, X_train_2nd, y_train, cv=split)
cv_score_2nd.mean()*100
X_test_2nd = pd.DataFrame()

for cls_name, classifier in classifiers.items():
    X_test_2nd[cls_name] = classifier.predict(X_test)

sns.barplot(list(classifiers.keys()), cv_mean_scores)
predict_test_2nd = adaboost_model_2nd.predict(X_test_2nd)
#submission
submission = pd.DataFrame({
        "PassengerId": X_test_PassengerId,
        "Survived": predict_test_2nd
    })

submission.to_csv('titanic_results.csv', index=False)

#Hyper parameter tuning of RandomForest.

gs_random_forest_params = [{
    'n_estimators':[50],
    'criterion': ['entropy'],
    'max_features':[5, 'auto', 'log2'],
    'min_samples_split':[5, 10, 15],
    'min_samples_leaf':[2],
    'bootstrap' : [True], 
    'n_jobs':[-1],
    'oob_score':[True]
    }]

gs_rf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=gs_random_forest_params, cv=split)
gs_rf.fit(X=X_train, y=y_train)

rf_best_params = gs_rf.best_params_
rf_best_score = gs_rf.best_score_
rf_best = gs_rf.best_estimator_

print (rf_best_params)
print (rf_best_score)
print (rf_best)

#Using different scaling on Fare and not including FareGroup dummy variable

'''
final_data_minmax = all_data.copy()

final_data_minmax = final_data_minmax.join(pd.get_dummies(final_data_minmax[['Cabin', 'Embarked', 'Sex', 'Ticket', 'Pclass', 'AgeGroup', 'FamilyType']]))
#final_data_minmax.shape

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
final_data_minmax['FareNew'] = scaler.fit_transform(final_data_minmax['Fare'])

#Drop unwantedcolumns and also columns which were used for dummy creation as they are not needed any more
final_data_minmax.drop(['Age', 'Cabin', 'Embarked', 'Fare', 'Name', 'Parch', 'PassengerId', 'Pclass', 'Sex', 'SibSp', 'Survived', 'Ticket', 'FareGroup', 'AgeGroup', 'FamilySize', 'FamilyType'], axis=1, inplace=True)

final_data_minmax.columns

X_train_minmax = final_data_minmax[:len(train)]
X_test_minmax = final_data_minmax[len(train):]

print (X_train_minmax.shape)
print (X_test_minmax.shape)

from sklearn.model_selection import cross_val_score

cv_mean_scores_minmax = []

for cls_name, classifier in classifiers.items():
    scores = cross_val_score(classifier, X_train_minmax, y_train, cv=split)
    cv_mean_scores_minmax.append(scores.mean() * 100)

sns.barplot(list(classifiers.keys()), cv_mean_scores_minmax)
'''