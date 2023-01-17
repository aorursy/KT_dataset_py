import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import warnings

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
warnings.filterwarnings('ignore')
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
tt_data = [train_df, test_df]

mean_age = train_df['Age'].mean()
median_fare = train_df['Fare'].median()

# filling in missing values
for df in tt_data:
    df['Age'].fillna(value=mean_age, inplace=True)
    df['Fare'].fillna(value=median_fare, inplace=True)
    df.dropna(subset = ['Embarked'], inplace=True)
train_df.head()
test_df.head()
train_df.info()
test_df.info()
train_df.describe()
test_df.describe()
train_df.describe(include=['O'])
test_df.describe(include=['O'])
f, ax = plt.subplots()
sns.distplot(train_df['Age'], kde=False, ax=ax);
f, ax = plt.subplots()
sns.distplot(train_df['Fare'], kde=False, ax=ax);
cdf = train_df["Pclass"].value_counts().to_frame().reset_index()
cdf.rename(columns={"index": "Pclass", "Pclass": "Count"}, inplace=True)
sns.barplot(x='Pclass', y='Count', data=cdf);
cdf = train_df["SibSp"].value_counts().to_frame().reset_index()
cdf.rename(columns={"index": "SibSp", "SibSp": "Count"}, inplace=True)
sns.barplot(x='SibSp', y='Count', data=cdf);
cdf = train_df["Parch"].value_counts().to_frame().reset_index()
cdf.rename(columns={"index": "Parch", "Parch": "Count"}, inplace=True)
sns.barplot(x='Parch', y='Count', data=cdf);
cdf = train_df["Survived"].value_counts().to_frame().reset_index()
cdf.rename(columns={"index": "Survived", "Survived": "Count"}, inplace=True)
sns.barplot(x='Survived', y='Count', data=cdf);
cdf = train_df["Sex"].value_counts().to_frame().reset_index()
cdf.rename(columns={"index": "Sex", "Sex": "Count"}, inplace=True)
sns.barplot(x='Sex', y='Count', data=cdf);
cdf = train_df["Embarked"].value_counts().to_frame().reset_index()
cdf.rename(columns={"index": "Embarked", "Embarked": "Count"}, inplace=True)
sns.barplot(x='Embarked', y='Count', data=cdf);
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(train_df.corr(), annot=True, cmap='RdBu', ax=ax);
for df in tt_data:    
    df['Has_Cabin'] = df["Cabin"].apply(lambda x: 0 if pd.isnull(x) else 1)
    df['Family_members'] = df['SibSp'] + df['Parch'] + 1
    df['Alone'] = 0
    df.loc[df['Family_members'] == 1, 'Alone'] = 1
    df['Age'] = pd.cut(df['Age'], include_lowest=True, bins=7, 
                       labels=[x for x in range(1, 8)]).cat.codes
    df['Fare'] = pd.cut(df['Fare'], include_lowest=True, bins=7, 
                       labels=[x for x in range(1, 8)]).cat.codes
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'Q':1, 'C':2, 'S':3})


train_X = train_df.drop(['Survived', 'PassengerId', 'Cabin', 'Ticket', 'Name'], axis=1)
train_y = train_df['Survived']
test_X = test_df.drop(['PassengerId', 'Cabin', 'Ticket', 'Name'], axis=1)

# train_X = pd.get_dummies(train_X)
# test_X = pd.get_dummies(test_X)
x_train, x_test, y_train, y_test = train_test_split(train_X, train_y, 
                                                    test_size=0.25, random_state=0)
scaler = StandardScaler()
scaler.fit(x_train)
x_train_columns = x_train.columns
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
logisticRegr = LogisticRegression(solver='liblinear', max_iter=100, C=8)
logisticRegr.fit(x_train, y_train.values.ravel())
score = logisticRegr.score(x_train, y_train)
print('Accuracy: on training data: ', score)
score = logisticRegr.score(x_test, y_test)
print('Accuracy: on test data: ', score)
# Confusion matrix
predictions = logisticRegr.predict(x_test)
cm = metrics.confusion_matrix(y_test, predictions)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
rf = RandomForestClassifier(n_estimators=100, max_features=5, max_leaf_nodes=22)
rf.fit(x_train,y_train.values.ravel())
y_pred=rf.predict(x_train)
score = metrics.accuracy_score(y_train, y_pred)
print('Accuracy: on training data: ', score)
y_pred=rf.predict(x_test)
score = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: on test data: ', score)

# Feature importance
feature_imp = pd.Series(rf.feature_importances_, index=x_train_columns).sort_values(ascending=False)
fig, ax = plt.subplots()
sns.barplot(x=feature_imp, y=feature_imp.index);
plt.xlabel('Feature Importance Score');
plt.ylabel('Features');
hidden_layers = (3,)*3
nn = MLPClassifier(solver='lbfgs', alpha=2**(-12), hidden_layer_sizes=hidden_layers, random_state=1, 
                   activation='logistic', batch_size='auto')
nn.fit(x_train, y_train.values.ravel())
score = nn.score(x_train, y_train)
print('Accuracy: on training data: ', score)
score = nn.score(x_test, y_test)
print('Accuracy: on test data: ', score)
xgb = XGBClassifier(max_depth=3, learning_rate=2**(-10), n_estimators=100, 
                reg_lambda=50, reg_alpha=0.7, booster='gbtree')
xgb.fit(x_train, y_train.values.ravel())
pred = xgb.predict(x_train)
score = metrics.accuracy_score(y_train.values.ravel(), pred)
print('Accuracy: on training data: ', score)
print('F1-score (training data):')
print(metrics.classification_report(y_train.values.ravel(), pred))
pred = xgb.predict(x_test)
score = metrics.accuracy_score(y_test.values.ravel(), pred)
print('Accuracy: on test data: ', score)
print('F1-score (test data):')
print(metrics.classification_report(y_test.values.ravel(), pred))
logisticRegr = LogisticRegression(solver='liblinear', max_iter=100, C=8)
rf = RandomForestClassifier(n_estimators=100, max_features=5, max_leaf_nodes=22)
nn = MLPClassifier(solver='lbfgs', alpha=2**(-12), hidden_layer_sizes=hidden_layers, random_state=1, 
                   activation='logistic', batch_size='auto')
xgb = XGBClassifier(max_depth=3, learning_rate=2**(-8), n_estimators=100, 
                reg_lambda=1, reg_alpha=0, booster='gbtree')

vc = VotingClassifier(estimators=
                         [('lr', logisticRegr), ('rf', rf), ('nn', nn), ('xgb', xgb)], 
                      voting='hard', weights=[1,2,1,3])

scaler = StandardScaler()
train_X_columns = train_X.columns
scaler.fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)
vc = vc.fit(train_X, train_y)
final_pred = vc.predict(test_X)


xgb = XGBClassifier(max_depth=3, learning_rate=2**(-10), n_estimators=100, 
                reg_lambda=50, reg_alpha=0.7, booster='gbtree')
xgb = xgb.fit(train_X, train_y)
final_pred = xgb.predict(test_X)
submission = pd.DataFrame({
        "PassengerId": test_df['PassengerId'],
        "Survived": final_pred
})

submission.to_csv('submission.csv', index=False)