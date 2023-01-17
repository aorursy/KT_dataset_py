# Import libraries
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import time
dataset = pd.read_csv('../input/train.csv')
dataset.head()
# Encode sex data so we can better viualize it
dataset['Sex'] = dataset['Sex'].map({'male':0, 'female':1}).astype(int)
print(dataset.isnull().any())
dataset['Embarked'].value_counts()
dataset['Embarked'] = dataset['Embarked'].fillna('S')
dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)
dataset['Embarked'].head()
dataset.corr()
# Extract nan in age
data_age = dataset.dropna(subset=['Age'])
data_no_age = dataset[dataset['Age'].isnull()]

# Age vs Pclass
ax = data_age.boxplot(column='Age', by='Pclass', grid=False)
ax.set_ylabel('Ages')
plt.suptitle("")

# Age vs SibSp
ax = data_age.boxplot(column='Age', by='SibSp', grid=False)
ax.set_ylabel('Ages')
plt.suptitle("")

# Extract data for prediction
X_fit_age = data_age[['Pclass', 'SibSp']].loc[:].values
Y_fit_age = data_age['Age'].loc[:].values.reshape(-1,1)

# Building and fitting the model
regressor_fill_age = LinearRegression()
regressor_fill_age.fit(X_fit_age, Y_fit_age)

# Gives prediction
X_pred_age = data_no_age[['Pclass', 'SibSp']].loc[:].values
X_pred_age[:,1][X_pred_age[:,1]>5] = 5
data_no_age['Age'] = regressor_fill_age.predict(X_pred_age)

# Fill in the missing data
data_filled = pd.concat([data_age, data_no_age])
data_filled['Age'] = data_filled['Age'].astype(int)
print(data_filled.isnull().any())
data_filled.describe()
fig = plt.figure(figsize=(10, 12))

ax1 = fig.add_subplot(421)
plt.hist(data_filled['Survived'], edgecolor='black')
plt.title('Survived')

ax2 = fig.add_subplot(422)
plt.hist(data_filled['Pclass'], edgecolor='black')
plt.title('Pclass')

ax2 = fig.add_subplot(423)
plt.hist(data_filled['Sex'], edgecolor='black')
plt.title('Sex')

ax2 = fig.add_subplot(424)
plt.hist(data_filled['Age'], edgecolor='black')
plt.title('Age')

ax2 = fig.add_subplot(425)
plt.hist(data_filled['SibSp'], edgecolor='black')
plt.title('SibSp')

ax2 = fig.add_subplot(426)
plt.hist(data_filled['Parch'], edgecolor='black')
plt.title('Parch')

ax2 = fig.add_subplot(427)
plt.hist(data_filled['Fare'], edgecolor='black')
plt.title('Fare')

ax2 = fig.add_subplot(428)
plt.hist(data_filled['Embarked'], edgecolor='black')
plt.title('Embarked')

plt.tight_layout()
# copy data set to process
data_train = data_filled.copy()
data_train.head()
data_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
data_train['FareGrp'] = pd.qcut(data_filled['Fare'], 3)
data_train[['FareGrp', 'Survived']].groupby(['FareGrp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
filter_fare = lambda x: 0 if x < 8.662 else 1 if (x <26.0 and x >= 8.662) else 2
data_train['Fare'] = data_filled['Fare'].apply(filter_fare)
data_train[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean().sort_values(by='Survived', ascending=False)
data_train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
data_train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Has relatives?
data_train['Relatives'] = data_filled['SibSp']+data_filled['Parch']
filter_relatives = lambda x:0 if x<1 else 1
data_train['Relatives'] = data_train['Relatives'].apply(filter_relatives)
data_train[['Relatives', 'Survived']].groupby(['Relatives'], as_index=False).mean().sort_values(by='Survived', ascending=False)
filter_age = lambda x:0 if x<15 else 1 if (x<55 and x >=15) else 2
data_train['AgeGrp'] = data_filled['Age'].apply(filter_age)
data_train[['AgeGrp', 'Survived']].groupby(['AgeGrp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
data_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Encode the Embarked feature
onehot = OneHotEncoder()
one_hot_matrix = onehot.fit_transform(data_train['Embarked'].loc[:].values.reshape(-1,1)).toarray()
data_train['Embarked S'] = one_hot_matrix[:,0]
data_train['Embarked C'] = one_hot_matrix[:,1]
data_train = data_train[['Survived', 'Pclass', 'Sex', 'Fare', 'Embarked S', 'Embarked C', 'AgeGrp', 'Relatives']]
data_train.head()
# Extract training data to array
X_set = data_train.iloc[:,1:].values
Y_set = data_train.loc[:,'Survived'].values

# Splitting training and validation set
seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_set, Y_set, test_size=0.2, random_state=seed)
training_data = [X_train, X_val, Y_train, Y_val]
def do_classifiy(training_data, classifier, parameters, cv=5):
    '''
    Arguments:
    training_data -- training data set, consist of both train set and validation set
    classifier -- the desire classifier
    parameters -- dictionary of parameters to be tested
    cv -- number fold in cross validation
    
    Returns
    clf_best -- classifier corresponds to the best hyperparameters
    grid_score -- accuracy on training search when doing grid search with k-fold cross validation
    acc_score -- accuracy on the validation set
    
    '''
    X_train, X_val, Y_train, Y_val = training_data
    gs = GridSearchCV(classifier, param_grid=parameters, cv=cv)
    start_time = time.time()
    gs.fit(X_train,Y_train)
    print('Best score: %f, using: %s' %(gs.best_score_, gs.best_params_))
    print('Total run time: %s seconds' %(time.time()-start_time))
    clf_best = gs.best_estimator_
    
    means = gs.cv_results_['mean_test_score']
    stds = gs.cv_results_['std_test_score']
    params = gs.cv_results_['params']
    for mean, std, param in zip(means, stds, params):
        print('Test score: %f, std: %f, using %s' %(mean, std, param))
    
    grid_score = gs.best_score_
    acc_score = clf_best.score(X_val, Y_val)
    print('Best train score: %f' %grid_score)
    print('Validation score: %f' %acc_score)
    return clf_best, grid_score, acc_score
clf_svm = SVC(kernel='rbf')
params_svm = {'kernel':['rbf', 'sigmoid', 'linear']}
clf_svm, score_svm, acc_svm = do_classifiy(training_data, clf_svm, params_svm)
clf_rf = RandomForestClassifier(n_estimators=10)
params_rf = {'n_estimators': [5, 20, 100]}
clf_rf, score_rf, acc_rf = do_classifiy(training_data, clf_rf, params_rf)
clf_knn = KNeighborsClassifier(n_neighbors=5)
params_knn = {'n_neighbors':[3,10,20]}
clf_knn, score_knn, acc_knn = do_classifiy(training_data, clf_knn, params_knn)
clf_log = LogisticRegression(C=1.0, tol=1e-4)
params_log = {'C':[0.01, 0.1, 1]}
clf_log, score_log, acc_log = do_classifiy(training_data, clf_log, params_log)
models = pd.DataFrame({'Model': ['Support Vector Machine', 'Random Forest',
                                 'K-Nearest Neighbours', 'Logistic Regression'],
    'Training score': [score_svm, score_rf, score_knn, score_log],
    'Validation score': [acc_svm, acc_rf, acc_knn, acc_log]})
columns_title = ['Model', 'Training score', 'Validation score']
models = models.reindex(columns=columns_title)
models.sort_values(by='Validation score', ascending=False)
# Import
testset = pd.read_csv('../input/test.csv')

# Emcode sex
testset['Sex'] = testset['Sex'].map({'male':0, 'female':1}).astype(int)

# Fill in missing and encode Embarked
testset['Embarked'] = testset['Embarked'].fillna('S')
testset['Embarked'] = testset['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)
onehot = OneHotEncoder()
one_hot_matrix = onehot.fit_transform(testset['Embarked'].loc[:].values.reshape(-1,1)).toarray()
testset['Embarked S'] = one_hot_matrix[:,0]
testset['Embarked C'] = one_hot_matrix[:,1]

# Fill in missing age
X_pred_age = testset[['Pclass', 'SibSp']].loc[:].values
X_pred_age[:,1][X_pred_age[:,1]>5] = 5
age_pred = regressor_fill_age.predict(X_pred_age)
for i in range(len(testset['Age'])):
    if pd.isnull(testset.loc[i,'Age']):
        testset.loc[i,'Age'] = int(age_pred[i])

# Age group
testset['AgeGrp'] = testset['Age'].apply(filter_age)
        
# Encode the fare
testset['Fare'] = testset['Fare'].apply(filter_fare)

# Encode Relatives
testset['Relatives'] = testset['SibSp']+testset['Parch']
testset['Relatives'] = testset['Relatives'].apply(filter_relatives)

# Extract data for prediction
for_pred = testset[['Pclass', 'Sex', 'Fare', 'Embarked S', 'Embarked C', 'AgeGrp', 'Relatives']]
X = for_pred.loc[:,:].values
# Make predict
Y_pred = clf_rf.predict(X)

# Write output to a file
ID = testset['PassengerId']
filename = 'result.csv'
myfile = open(filename,'w')
titleRow = 'PassengerID,Survived\n'
myfile.write(titleRow)
for i in range(len(Y_pred)):
    row = str(ID[i]) + ',' + str(Y_pred[i]) + '\n'
    myfile.write(row)
myfile.close()