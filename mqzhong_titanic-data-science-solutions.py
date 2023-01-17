import pandas as pd
train_df = pd.read_csv("../input/train.csv")
train_df.head()
#Divided the whole table into two sub tables: survived and non-survived
is_survived_filter = train_df["Survived"] == 1
non_survived_filter = train_df["Survived"] == 0
survived = train_df[is_survived_filter]
non_survived = train_df[non_survived_filter]
survived.head()
non_survived.head()
train_df.info()
train_df.describe()
train_df.describe(include=['O']) 
#drop the variable name, because it not relevant to the prediction of survival
#figure out relevant between survived and each variable (including numerical and caregorical)
#firstly, discovery the relationship between discrete variables and survived. Including Pclass, Sex, Sibsp, Parch, and Embark.
#relationship between pclass and survived
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# there are some correlation between the pclass and survived, so keep the variable pclass
#relationship between sex and survived
train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# there are some correlation between the sex and survived, so keep this variable into final model.
#relationship between sibsp and survived
train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#do not have any clear correlation, hence drop the Sibsp variable
#relationship between parch and survived
train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#do not have any clear correlation, hence drop the Parch variable 
#relationship between embarked and survived
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# there are some correlation between the embarked and survived, so keep the embarked variable
# Secondly, print out the relationship between the continuous variable and survived. Including Age and Fare.
# utilize table (survived and non-survived)
#the relationship between the age and survived
#survived passenger age
age_survived_df = survived.ix[:,['Age', 'Survived']]
age_survived_df
hist_survive_age = age_survived_df.hist(column = "Age")
#non-survived passenger age
age_non_survived_df = non_survived.ix[:,['Age', 'Survived']]
age_non_survived_df
hist_non_survived_age = age_non_survived_df.hist(column = "Age")
#According to compare two histogram of age, we can figure out large part of child (0 to 10) and old people (older than 75) are survived. Hence, the age variable is relevant to the survived.
#relationship between fare and survived
#survived passenger fare
fare_survived_df = survived.ix[:,['Fare', 'Survived']]
fare_survived_df
hist_survive_fare = fare_survived_df.hist(column = "Fare")
#non survived passenger fare
fare_non_survived_df = non_survived.ix[:,['Fare', 'Survived']]
fare_non_survived_df
hist_non_survive_fare = fare_non_survived_df.hist(column = "Fare")
#According to compare two histogram of fare, we can figure out passengers who's fare is larger than 250 are almost survived. Hence, the fare variable is relevant to the survived
#dealing with the missing value， utilizing mean of each column to impute the missing value

fill_train_df = train_df.fillna(train_df.mean())
fill_train_df.head()
#preparation of the data frame

fill_train_df['Sex'] = fill_train_df['Sex'].map({'female': 1, 'male': 0}).astype(int)

fill_train_df['Embarked'] = fill_train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
data_train_df = fill_train_df.drop(columns = ['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'])
X_train = data_train_df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
Y_train = data_train_df['Survived']
X_train = X_train.fillna(X_train.mean())
data_train_df.head()
test_df = pd.read_csv("../input/test.csv")
test_df.head()
test_df['Sex'] = test_df['Sex'].map({'female': 1, 'male': 0}).astype(int)
test_df['Embarked'] = test_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
data_test_df = test_df.drop(columns = ['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'])
final_test_df = data_test_df.fillna(train_df.mean())
combine = [data_train_df, final_test_df]
X_test = final_test_df
X_train.shape, Y_train.shape, X_test.shape
X_train.head()
# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
# Decision Tree
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
# Random Forest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
models = pd.DataFrame({
    'Model': ['Stochastic Gradient Decent', 
              'Random Forest',
              'Decision Tree'],
    'Score': [acc_sgd, acc_random_forest,
              acc_decision_tree]})
print(models.sort_values(by='Score', ascending=False))
#because random forest with the highest score, choose the random forest
rf = RandomForestClassifier(n_estimators=250, max_depth=5, criterion='gini')
rf.fit(X_train, Y_train)
Y_pred = rf.predict(X_test)
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv("submission.csv")