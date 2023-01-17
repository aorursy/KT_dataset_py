import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, randint
%matplotlib inline

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LinearRegression, BayesianRidge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import BaggingRegressor, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

train_dat = pd.read_csv('../input/train.csv')
test_dat = pd.read_csv('../input/test.csv')
all_dat = [train_dat, test_dat]
display(train_dat.head(3))
display(train_dat.info())
display(train_dat.describe())
display(test_dat.head(3))
display(test_dat.info())
display(test_dat.describe())
import missingno as msno
display(msno.bar(train_dat))
display(msno.bar(test_dat))
train_dat.drop(['Cabin', 'Ticket'], axis=1, inplace=True)
test_dat.drop(['Cabin', 'Ticket'], axis=1, inplace=True)
train_dat.corr()
train_dat[['Pclass', 'Survived']].groupby('Pclass').mean()
train_dat[['Sex', 'Survived']].groupby('Sex').mean()
age_survived = train_dat.loc[train_dat['Survived'] == 1]['Age'].dropna()
age_dead = train_dat.loc[train_dat['Survived'] == 0]['Age'].dropna()

fig = sns.kdeplot(age_survived, label='Survived', clip=[0,80])
fig.set(xlim=(0, 80))
sns.kdeplot(age_dead, label='Did Not Survive')

ks_2samp(age_survived, age_dead)
sns.barplot(train_dat['SibSp'], train_dat['Survived'])
display(train_dat[['Parch', 'Survived']].groupby('Parch').mean())
sns.barplot(train_dat['Parch'], train_dat['Survived'])
display(train_dat[['Parch', 'Survived']].groupby('Parch').mean())
fare_survived = train_dat.loc[train_dat['Survived'] == 1]['Fare'].dropna()
fare_dead = train_dat.loc[train_dat['Survived'] == 0]['Fare'].dropna()

fig = sns.kdeplot(fare_survived, label='Survived', clip=[0,80])
fig.set(xlim=(0, 80))
sns.kdeplot(fare_dead, label='Did Not Survive')
ks_2samp(fare_survived, fare_dead)
fig = plt.figure(figsize=(12,8))
sns.boxplot(train_dat['Pclass'], train_dat['Fare'])
import re as re
def title_extract_function(string):
    title = re.search('([A-Za-z]+)\.', string)[0]
    return title

for df in all_dat:
    df['Title'] = df['Name'].apply(title_extract_function)

display(pd.crosstab(all_dat[0]['Title'], all_dat[0]['Sex']))
display(pd.crosstab(all_dat[1]['Title'], all_dat[1]['Sex']))

for df in all_dat:
    df['Title'] = df['Title'].replace(['Mlle.', "Ms."], 'Miss.')
    df['Title'] = df['Title'].replace(['Mme.'], 'Mrs.')
    df['Title'] = df['Title'].replace(['Capt.','Col.','Countess.','Don.', 'Dona.','Dr.','Jonkheer.','Lady.', 'Major.', 'Rev.',
                                       'Sir.'],'Other')

train_dat[['Title', 'Survived']].groupby('Title').mean()
for df in all_dat:
    df['Family Size'] = df['Parch'] + df['SibSp'] +1

train_dat[['Family Size', 'Survived']].groupby('Family Size').mean()
for df in all_dat:
    df['Alone'] = 0
    df.loc[df['Family Size'] == 1, 'Alone'] = 1
train_dat[['Alone', 'Survived']].groupby('Alone').mean()
random.seed(1234)
while train_dat['Embarked'].isna().sum() > 0:
    train_dat['Embarked'].fillna(random.randint(1,3), limit=1, inplace=True)
train_dat['Embarked'].replace({1:"S", 2:"C", 3:"Q"}, inplace=True)

train_dat[['Embarked', 'Survived']].groupby('Embarked').mean()
Fare_corr = pd.DataFrame(train_dat.corr()['Fare'].drop(['Fare'],axis=0))
Fare_corr.reindex(Fare_corr.Fare.abs().sort_values(inplace=False, ascending=False).index)
sns.barplot(x=train_dat['Pclass'], y=train_dat['Fare'])

average_fare_by_class = test_dat.groupby(by=['Pclass'], as_index=False)['Fare'].mean()

if (test_dat.loc[test_dat['Fare'].isnull(),'Pclass'] == 3).values[0] == True:
    test_dat.loc[test_dat['Fare'].isnull(),'Fare'] = average_fare_by_class[average_fare_by_class['Pclass'] == 1]['Fare'][0]
elif (test_dat.loc[test_dat['Fare'].isnull(),'Pclass'] == 2).values[0] == True:
    test_dat.loc[test_dat['Fare'].isnull(),'Fare'] = average_fare_by_class[average_fare_by_class['Pclass'] == 2]['Fare'][0]
else:
    test_dat.loc[test_dat['Fare'].isnull(),'Fare'] = average_fare_by_class[average_fare_by_class['Pclass'] == 3]['Fare'][0]
sns.violinplot(train_dat['Fare'])
max_fare = train_dat['Fare'].max()
plt.plot([60, 60], [-1, 1], linewidth=2)
plt.plot([110, 110], [-1, 1], linewidth=2)
plt.plot([180, 180], [-1, 1], linewidth=2)
plt.plot([max_fare, max_fare], [-1, 1], linewidth=2)
for df in all_dat:
    df['Fare Type'] = pd.cut(df['Fare'], [0, 60, 110,180, 1000], labels=['1', '2', '3', "4"])

train_dat[['Fare Type', 'Survived']].groupby('Fare Type').mean()
display(all_dat[0].head(2))
display(all_dat[1].head(2))
categorical_df = []
for df in all_dat:
    sex = pd.get_dummies(df['Sex'],drop_first=True)
    embark = pd.get_dummies(df['Embarked'],drop_first=True)
    title = pd.get_dummies(df['Title'], drop_first=True)
    fare = pd.get_dummies(df['Fare Type'], drop_first=True)
    dummies = pd.concat([sex, embark, title, fare], axis=1)
    df2 = df.drop(['Sex','Embarked','Name','Fare','Fare Type','Title'], axis=1, inplace=True)
    df3 = pd.concat([df, sex, embark, title, fare], axis=1)
    categorical_df.append(df3)
missing_age = categorical_df[0][pd.isnull(categorical_df[0]).any(axis=1)].drop(['PassengerId'], axis=1)
complete_age = categorical_df[0].dropna().drop(['PassengerId'], axis=1)
total_RMSE = pd.DataFrame(columns=["LM RMSE", "BRR RMSE", "NNR RMSE", "DTR RMSE", "BR RMSE"])
total_MAE = pd.DataFrame(columns=["LM MAE", "BRR MAE", "NNR MAE", "DTR MAE", "BR MAE"])

for i in range(0, 100):
    x = complete_age.drop('Age',axis=1)
    y = complete_age['Age']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
    scaler = StandardScaler()  
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)  
    x_test = scaler.transform(x_test)
    
    lm = LinearRegression()
    lm.fit(x_train, y_train)
    lm_pred = lm.predict(x_test)

    brr = BayesianRidge()
    brr.fit(x_train, y_train)
    brr_pred = brr.predict(x_test)

    nnr = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam',    alpha=0.001,batch_size='auto',
                   learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
                   random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                   nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                   epsilon=1e-08)
    nnr.fit(x_train, y_train)
    nnr_pred = nnr.predict(x_test)

    dtr = DecisionTreeRegressor()
    dtr.fit(x_train,y_train)
    dtr_pred = dtr.predict(x_test)

    br = BaggingRegressor()
    br.fit(x_train, y_train)
    br_pred = br.predict(x_test)

    RMSEs = [mean_squared_error(y_test, lm_pred), mean_squared_error(y_test, brr_pred), mean_squared_error(y_test, nnr_pred),
            mean_squared_error(y_test, dtr_pred), mean_squared_error(y_test, br_pred)]
    total_RMSE.loc[i] = RMSEs
    MAEs = [mean_absolute_error(y_test, lm_pred), mean_absolute_error(y_test, brr_pred), mean_absolute_error(y_test, nnr_pred),
            mean_absolute_error(y_test, dtr_pred), mean_absolute_error(y_test, br_pred)]
    total_MAE.loc[i] = MAEs

average_RMSE = total_RMSE.mean(axis=0).values
average_MAE = total_MAE.mean(axis=0).values

Age_ML_Summary = pd.DataFrame(average_RMSE, columns=["RMSE"], index=['Linear Reg', "Bayesian Ridge", 'MLP', 
                                                                     'Decision Tree', 'Bagging Reg'])
Age_ML_Summary['MAE'] = average_MAE
Age_ML_Summary.transpose()
def predict_age(df):
    if (df['PassengerId'].max()== 891):
        df = df.drop('Survived', axis=1)
    missing_age = df[pd.isnull(df).any(axis=1)]
    complete_age = df.dropna()
    
    x_train = complete_age.drop(['Age', 'PassengerId'],axis=1)
    y_train = complete_age['Age']
    x_test = missing_age.drop(['Age', 'PassengerId'], axis=1)
    
    scaler = StandardScaler()  
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)  
    x_test = scaler.transform(x_test)
    
    nnr = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.001,batch_size='auto',
                   learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
                   random_state=1234, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                   nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                   epsilon=1e-08)
    nnr.fit(x_train, y_train)
    predicted_age = nnr.predict(x_test)
    
    missing_age = df[pd.isnull(df).any(axis=1)].drop('Age',axis=1)
    age_df = missing_age[['PassengerId']]
    age_df = age_df.assign(Age = predicted_age)
    age_df.sort_values(by=['PassengerId'])
    return age_df

complete_data = []
for df in categorical_df:
    pred_age = predict_age(df)
    complete_df = df.combine_first(pred_age)
    complete_data.append(complete_df)
x = complete_age.drop('Survived',axis=1)
y = complete_age['Survived']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
scaler = StandardScaler()  
scaler.fit(x_train)
x_train = scaler.transform(x_train)  
x_test = scaler.transform(x_test)
#import timeit
#start = timeit.default_timer()

total_accuracy = pd.DataFrame(columns=["Logistic Regression", "SVC", "Decision Tree", 
                                      "Random Forest", "Adaboost", "MLP", "Kneighbours"])

for i in range(0, 5):
    x = complete_age.drop('Survived',axis=1)
    y = complete_age['Survived']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
    scaler = StandardScaler()  
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)  
    x_test = scaler.transform(x_test)
    
    params = {'C': np.arange(0.001, 100, 0.5)}
    logreg_grid = RandomizedSearchCV(LogisticRegression(), params)
    logreg_grid.fit(x_train, y_train)
    logreg_pred = logreg_grid.predict(x_test)
    
    params = {'C': np.arange(0.1, 5, 0.1), 'gamma': np.arange(0.00001, 1, 0.05), 'kernel': ['rbf', 'linear', 'sigmoid']}
    svc_grid = RandomizedSearchCV(SVC(), params)
    svc_grid.fit(x_train, y_train)
    svc_predictions = svc_grid.predict(x_test)
    
    params = {'criterion': ['gini', 'entropy'], 'max_depth': range(1,50)}
    decisiontree_grid = RandomizedSearchCV(DecisionTreeClassifier(), params)
    decisiontree_grid.fit(x_train, y_train)
    DT_predictions = decisiontree_grid.predict(x_test)
    
    params = {'criterion': ['gini', 'entropy'], 'n_estimators': range(5, 100), 'max_depth': range(1,50)}
    rf_grid = RandomizedSearchCV(RandomForestClassifier(), params)
    rf_grid.fit(x_train, y_train)
    rf_predictions = rf_grid.predict(x_test)
    
    params = {'learning_rate': np.arange(0.01, 1.1, 0.1), 'n_estimators': np.arange(50,200, 25)}
    adaboost_grid = RandomizedSearchCV(AdaBoostClassifier(),params)
    adaboost_grid.fit(x_train, y_train)
    adaboost_predictions = adaboost_grid.predict(x_test)
    
    params = {'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver': ['lbfgs', 'sgd', 'adam'],
         'alpha': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]}
    mlp_grid = RandomizedSearchCV(MLPClassifier(max_iter=1000),params)
    mlp_grid.fit(x_train, y_train)
    mlp_predictions = mlp_grid.predict(x_test)

    params = {'n_neighbors': np.arange(5,50, 2), 'weights': ['uniform', 'distance']}
    KN_grid = RandomizedSearchCV(KNeighborsClassifier(),params)
    KN_grid.fit(x_train, y_train)
    KN_predictions = KN_grid.predict(x_test)
    
    accuracies = [accuracy_score(y_test, logreg_pred),accuracy_score(y_test, svc_predictions), 
                  accuracy_score(y_test, DT_predictions), accuracy_score(y_test, rf_predictions),
                  accuracy_score(y_test, adaboost_predictions), accuracy_score(y_test, mlp_predictions),
                  accuracy_score(y_test, KN_predictions)]
    
    total_accuracy.loc[i] = accuracies

average_accuracy = pd.DataFrame(total_accuracy.mean(axis=0), columns=['Accuracy'])
average_accuracy.transpose()
x_train = complete_data[0].drop(['Survived', 'PassengerId'], axis=1)
y_train = complete_data[0]['Survived']
x_test = complete_data[1].drop(['PassengerId'], axis=1)

scaler = StandardScaler()  
scaler.fit(x_train)
x_train = scaler.transform(x_train)  
x_test = scaler.transform(x_test)
mlp = MLPClassifier(max_iter=2000, random_state=1234)
mlp.fit(x_train, y_train)
final_mlp = mlp.predict(x_test)
final_submission = complete_data[1].copy()
final_submission['Survived'] = final_mlp
submission_df = pd.DataFrame()
submission_df[['PassengerId', 'Survived']] = final_submission[['PassengerId', 'Survived']]
display(submission_df.head())
display(submission_df.groupby('Survived').count())

submission_df.to_csv("Titanic Submission.csv", index=False)