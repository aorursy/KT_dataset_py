import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
dataset = pd.read_csv('../input/train.csv')
backup_set = dataset[['PassengerId', 'Survived']]
dataset.info()
dataset = dataset.drop('Cabin', axis=1)
dataset.head()
tot_surv_died = dataset['Survived'].value_counts().values
plt.pie(labels=['Died', 'Survived'], x=tot_surv_died/891, autopct='%1.1f%%', colors=['red','green'])
plt.axis('equal')
plt.show()
print('Died = 0, Survived = 1')
print(dataset['Survived'].value_counts())
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey = True)
tot_pass_by_sex = dataset['Sex'].value_counts().values
ax1.bar(['Male', 'Female'], tot_pass_by_sex)
ax1.set_title('Number of Passengers')
surv_by_sex = dataset[dataset['Survived'] == 1]['Sex'].value_counts().values
ax2.bar(['Male', 'Female'], surv_by_sex[::-1])
ax2.set_title('Survived')
died_by_sex = dataset[dataset['Survived'] == 0]['Sex'].value_counts().values
ax3.bar(['Male', 'Female'], died_by_sex)
ax3.set_title('Died')
plt.show()
print("Number of Passengers")
print(dataset['Sex'].value_counts())
print("\nSurvived")
print(dataset[dataset['Survived'] == 1]['Sex'].value_counts())
print("\nDied")
print(dataset[dataset['Survived'] == 0]['Sex'].value_counts())
dataset[dataset['Age'].notnull()].info()
dataset.loc[dataset['Embarked'].isnull(), 'Embarked'] = dataset.loc[dataset['Embarked'].notnull(), 'Embarked'].mode().values
dataset[dataset['Age'].notnull()].info()
dataset[dataset['Age'].notnull()].head()
dataset = dataset.drop('Ticket', axis=1)
dataset['Title'] = dataset['Name'].str.extract('(\w+(?=\.))', expand=False)
dataset = dataset.drop('Name', axis=1)
dataset.head()
dataset['Title'].value_counts()
backup_set['Title'] = dataset['Title']
dataset['Title'] = dataset['Title'].str.replace('(Mlle|Mme|Ms)', 'Miss')
dataset['Title'] = dataset['Title'].str.replace('(Dr|Rev|Col|Major|Sir|Capt)', 'Occupation')
dataset['Title'] = dataset['Title'].str.replace('(Don|Countess|Lady|Jonkheer)', 'Noble')
dataset['Title'].value_counts()
dataset['FamilyCount'] = dataset['SibSp'] + dataset['Parch']
dataset['IsAlone'] = 0
dataset.loc[dataset['FamilyCount'] == 0, 'IsAlone'] = 1
dataset['IsMother'] = 0
dataset.loc[(dataset['Sex'] == 'female') & (dataset['Parch'] > 0) & (dataset['Title'] == 'Mrs'), 'IsMother'] = 1
dataset.head(10)
dataset.loc[dataset['Sex'] == 'male', 'Sex'] = 0
dataset.loc[dataset['Sex'] == 'female', 'Sex'] = 1
dataset.head()
dataset['Fare'].describe()
pd.cut(dataset['Fare'], bins=[0, 8, 15, 32, 100, 600]).value_counts()
dataset['FareBand'] = pd.cut(dataset['Fare'], bins=[0, 8, 15, 32, 100, 600], labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
dataset = dataset.drop('Fare', axis=1)
fare_band_surv_rate = dataset[dataset['Survived'] == 1]['FareBand'].value_counts().values / dataset['FareBand'].value_counts().values * 100
plt.bar(['Very Low', 'Low', 'Medium', 'High', 'Very High'], fare_band_surv_rate)
plt.title('Survival Rate (Ticket Fare)')
plt.show()
dataset[dataset['Survived'] == 1]['FareBand'].value_counts().values / dataset['FareBand'].value_counts().values * 100
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8,8))
ax1.hist([dataset.loc[:,'FamilyCount'], dataset.loc[dataset['Survived'] == 1, 'FamilyCount']],bins=np.arange(12)-0.5, color=['blue', 'green'], histtype='bar', label=['Family Members On-Board', 'Family Members Survived'])
ax1.set_xticks([0,1,2,3,4,5,6,7,10])
ax1.legend()
ax1.set_title('Survival Count (Family Count)')
ax2.hist([dataset.loc[:,'FamilyCount'], dataset.loc[dataset['Survived'] == 0, 'FamilyCount']],bins=np.arange(12)-0.5, color=['blue', 'red'], histtype='bar', label=['Family Members On-Board', 'Family Members Died'])
ax2.set_xticks([0,1,2,3,4,5,6,7,10])
ax2.legend()
ax2.set_title('Death Count (Family Count)')
plt.show()
dataset.loc[:,'FamilyCount'].value_counts()
dataset.loc[dataset['Survived'] == 1, 'FamilyCount'].value_counts()
dataset.loc[dataset['Survived'] == 0, 'FamilyCount'].value_counts()
mother_surv_died =  dataset.loc[dataset['IsMother'] == 1, 'Survived'].value_counts().values
plt.pie(labels=['Died', 'Survived'], x=mother_surv_died[::-1]/56, autopct='%1.1f%%', colors=['red','green'])
plt.axis('equal')
plt.show()
dataset.loc[dataset['IsMother'] == 1, 'Survived'].value_counts()
plt.figure(figsize=(6,12))
plt.bar(['Master', 'Miss', 'Mr', 'Mrs', 'Noble', 'Occupation'], dataset.loc[:, 'Title'].value_counts().sort_index().values, color='blue')
plt.bar(['Master', 'Miss', 'Mr', 'Mrs', 'Noble', 'Occupation'], dataset.loc[dataset['Survived'] == 1, 'Title'].value_counts().sort_index().values, color='green')
plt.legend()
plt.title('Survival Count (Title)')
plt.show()
dataset.loc[:, 'Title'].value_counts().sort_index()
dataset.loc[dataset['Survived'] == 1, 'Title'].value_counts().sort_index()
backup_set[(backup_set['Title'] == 'Lady') | (backup_set['Title'] == 'Countess')]
plt.bar(['(1) Upper', '(2) Middle', '(3) Lower'], dataset.loc[:, 'Pclass'].value_counts().sort_index().values, color='blue')
plt.bar(['(1) Upper', '(2) Middle', '(3) Lower'], dataset.loc[dataset['Survived'] == 1, 'Pclass'].value_counts().sort_index().values, color='green')
plt.legend()
plt.title('Survival Count (Pclass)')
plt.show()
dataset.loc[:, 'Pclass'].value_counts().sort_index()
dataset.loc[dataset['Survived'] == 1, 'Pclass'].value_counts().sort_index()
plt.bar(['C', 'Q', 'S'], dataset.loc[:, 'Embarked'].value_counts().sort_index().values, color='blue')
plt.bar(['C', 'Q', 'S'], dataset.loc[dataset['Survived'] == 1, 'Embarked'].value_counts().sort_index().values, color='green')
plt.legend()
plt.title('Survival Count (Embarked)')
plt.show()
dataset.loc[:, 'Embarked'].value_counts().sort_index()
dataset.loc[dataset['Survived'] == 1, 'Embarked'].value_counts().sort_index()
dataset = pd.concat([dataset, pd.get_dummies(dataset['Embarked'], drop_first=True), pd.get_dummies(dataset['Title'], drop_first=True)], axis=1).drop(['Embarked', 'Title'], axis=1)
dataset.head()
dataset['FareBand'] = dataset['FareBand'].cat.codes
dataset.head()
age_set = dataset.loc[dataset['Age'].notnull(), ['PassengerId', 'Survived', 'Pclass', 'Sex', 'SibSp', 'Parch',
       'FamilyCount', 'IsAlone', 'IsMother', 'FareBand', 'Q', 'S', 'Miss',
       'Mr', 'Mrs', 'Noble', 'Occupation', 'Age']] 
age_set.info()
X = age_set.loc[:, 'Pclass':'Occupation']
y = age_set['Age']

X_sm = sm.add_constant(X)
ols = sm.OLS(endog=y, exog=X_sm)
ols.fit().summary()
def backwardelimination(y, X, SL=0.05, add_const=True):
    if(add_const == True):
        X = sm.add_constant(X)
    num_vars = X.shape[1]
    temp = pd.DataFrame(np.zeros(X.shape).astype('int'), columns=X.columns.values)
    temp = temp.set_index(X.index.values)
    for i in range(num_vars):
        ols_regressor = sm.OLS(endog=y, exog=X).fit()
        max_var = max(ols_regressor.pvalues)
        adj_rsquared_before = ols_regressor.rsquared_adj
        if(max_var > SL):
            for j in range(num_vars - i):
                if(ols_regressor.pvalues[j] == max_var):
                    temp.iloc[:,j] = X.iloc[:,j]
                    X = X.drop(X.columns[j], axis=1)
                    temp_regressor = sm.OLS(endog=y, exog=X).fit()
                    adj_rsquared_after = temp_regressor.rsquared_adj
#                     if(adj_rsquared_after <= adj_rsquared_before):
#                         X_rollback = pd.concat([X, temp.iloc[:,j]], axis=1)
#                         print(temp.iloc[:,j:j+1].head())
#                         print(ols_regressor.summary())
#                         return X_rollback
        else:
            print(ols_regressor.summary())
            return X
        
X_opt = backwardelimination(y, X)
X_opt.head()
X_train, X_test, y_train, y_test = train_test_split(X_opt.iloc[:,1:], y, test_size=0.2, random_state=1)


linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
y_pred = linear_regressor.predict(X_test)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print(rmse)
lin_reg_age = LinearRegression()
lin_reg_age.fit(X_opt.iloc[:,1:], y)

X_test_cols = ['Pclass', 'IsAlone', 'IsMother', 'Q', 'Miss', 'Mr', 'Mrs', 'Noble', 'Occupation']
dataset.loc[dataset['Age'].isnull(), 'Age'] = lin_reg_age.predict(dataset.loc[dataset['Age'].isnull(), X_test_cols])
dataset.info()
dataset['Age'].describe()
dataset[dataset['IsMother'] == 1]['Age'].describe()
dataset[dataset['Mrs'] == 1]['Age'].describe()
age_set[age_set['Mrs'] == 1]['Age'].describe()
dataset['AgeBand'] = pd.cut(dataset['Age'], bins=[0, 1, 12, 18, 21, 29, 36, 60, 100])
dataset = dataset.drop('Age', axis=1)
age_band_surv_rate = dataset[dataset['Survived'] == 1]['AgeBand'].value_counts().sort_index().values / dataset['AgeBand'].value_counts().sort_index().values * 100
plt.figure(figsize=(8,4))
plt.bar(['(0, 1]', '(1, 12]', '(12, 18]', '(18, 21]', '(21, 29]', '(29, 36]', '(36, 60]', '(60, 100]'], age_band_surv_rate, color='green')
plt.title('Survival Rate (Age)')
plt.show()
dataset['AgeBand'] = dataset['AgeBand'].cat.codes
dataset.head()
X = dataset.iloc[:, 2:]
y = dataset.iloc[:, 1]

log_reg = LogisticRegression(random_state=1)
selector = RFECV(log_reg, cv=5)
selector.fit(X, y)
selector.support_
parameters = {
    'solver': ['newton-cg', 'lbfgs', 'liblinear'],
}
gs_cv_lr = GridSearchCV(log_reg, param_grid=parameters, scoring='accuracy', cv=5, n_jobs=4)
gs_cv_lr.fit(X, y)
gs_cv_lr.best_score_
g_nb_clas = GaussianNB()
np.mean(cross_val_score(g_nb_clas, X, y, scoring='accuracy', cv=5))
sgd_clas = SGDClassifier(random_state=1, max_iter=1000, tol=None)
selector = RFECV(sgd_clas, cv=5)
selector.fit(X, y)
sgd_cols = X.columns[selector.support_].values
parameters = {
    'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
    'penalty': ['l2', 'l1', 'elasticnet']
}
gs_cv_sgd = GridSearchCV(sgd_clas, param_grid=parameters, scoring='accuracy', cv=5, n_jobs=4)
gs_cv_sgd.fit(X[sgd_cols], y)
gs_cv_sgd.best_score_
kn_clas = KNeighborsClassifier()
parameters = {
    'n_neighbors': range(1,21),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [1, 2]
}
gs_cv_kn = GridSearchCV(kn_clas, param_grid=parameters, scoring='accuracy', cv=5, n_jobs=4)
gs_cv_kn.fit(X, y)
gs_cv_kn.best_score_
sv_clas = SVC(random_state=1)
parameters = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'probability': [True, False],
    'shrinking': [True, False],
    'decision_function_shape': ['ovo', 'ovr']
}
gs_cv_sv = GridSearchCV(sv_clas, param_grid=parameters, scoring='accuracy', cv=5, n_jobs=4)
gs_cv_sv.fit(X, y)
gs_cv_sv.best_score_
dt_clas = DecisionTreeClassifier(random_state=1)
selector = RFECV(dt_clas, cv=5)
selector.fit(X, y)
dt_cols = X.columns[selector.support_].values
parameters = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [1, 2, 3]
}
gs_cv_dt = GridSearchCV(dt_clas, param_grid=parameters, scoring='accuracy', cv=5, n_jobs=4)
gs_cv_dt.fit(X[dt_cols], y)
gs_cv_dt.best_score_
rf_clas = RandomForestClassifier(random_state=1)
selector = RFECV(rf_clas, cv=5)
selector.fit(X, y)
rf_cols = X.columns[selector.support_].values
parameters = {
    'n_estimators': [150, 180],
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [1, 2, 3]
}
gs_cv_rf = GridSearchCV(rf_clas, param_grid=parameters, scoring='accuracy', cv=5, n_jobs=4)
gs_cv_rf.fit(X[rf_cols], y)
gs_cv_rf.best_score_
mlp_clas = MLPClassifier(random_state=1, solver='lbfgs')
parameters = {
    'activation': ['identity', 'logistic', 'tanh', 'relu']
}
gs_cv_mlp = GridSearchCV(mlp_clas, param_grid=parameters, scoring='accuracy', cv=5, n_jobs=4)
gs_cv_mlp.fit(X, y)
gs_cv_mlp.best_score_
gs_cv_rf.best_params_
rf_clas = RandomForestClassifier(random_state=1, criterion='entropy', max_depth=10, min_samples_leaf=2, min_samples_split=6, n_estimators=150)
rf_clas.fit(X[rf_cols], y)
test_set = pd.read_csv('../input/test.csv').drop(['Cabin', 'Ticket'], axis=1)
test_set.info()
test_set['Fare'] = test_set['Fare'].fillna(test_set['Fare'].mean())
test_set['Title'] = test_set['Name'].str.extract('(\w+(?=\.))', expand=False)
test_set = test_set.drop('Name', axis=1)
test_set.head()
test_set['Title'] = test_set['Title'].str.replace('(Mlle|Mme|Ms)', 'Miss')
test_set['Title'] = test_set['Title'].str.replace('(Dr|Rev|Col|Major|Sir|Capt)', 'Occupation')
test_set['Title'] = test_set['Title'].str.replace('(Dona|Countess|Lady|Jonkheer)', 'Noble')
test_set['Title'].value_counts()
test_set['FamilyCount'] = test_set['SibSp'] + test_set['Parch']
test_set['IsAlone'] = 0
test_set.loc[test_set['FamilyCount'] == 0, 'IsAlone'] = 1
test_set['IsMother'] = 0
test_set.loc[(test_set['Sex'] == 'female') & (test_set['Parch'] > 0) & (test_set['Title'] == 'Mrs'), 'IsMother'] = 1
test_set.head(10)
test_set.loc[test_set['Sex'] == 'male', 'Sex'] = 0
test_set.loc[test_set['Sex'] == 'female', 'Sex'] = 1
test_set.head()
test_set['FareBand'] = pd.cut(test_set['Fare'], bins=[0, 8, 15, 32, 100, 600], labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
test_set = test_set.drop('Fare', axis=1)
test_set = pd.concat([test_set, pd.get_dummies(test_set['Embarked'], drop_first=True), pd.get_dummies(test_set['Title'], drop_first=True)], axis=1).drop(['Embarked', 'Title'], axis=1)
test_set.head()
test_set['FareBand'] = test_set['FareBand'].cat.codes
test_set.head()
X_test_cols = ['Pclass', 'IsAlone', 'IsMother', 'Q', 'Miss', 'Mr', 'Mrs', 'Noble', 'Occupation']
test_set.loc[test_set['Age'].isnull(), 'Age'] = lin_reg_age.predict(test_set.loc[test_set['Age'].isnull(), X_test_cols])
test_set.head()
test_set['AgeBand'] = pd.cut(test_set['Age'], bins=[0, 1, 12, 18, 21, 29, 36, 60, 100])
test_set = test_set.drop('Age', axis=1)
test_set.head()
test_set['AgeBand'] = test_set['AgeBand'].cat.codes
test_set.head()
test_set['Survived'] = rf_clas.predict(test_set[rf_cols])
test_set.head()
submission = test_set[['PassengerId', 'Survived']]
submission.to_csv('submission.csv', index=False)