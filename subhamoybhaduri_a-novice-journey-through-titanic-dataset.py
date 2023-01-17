import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train_data = pd.read_csv("../input/titanic/train.csv")
test_data  = pd.read_csv("../input/titanic/test.csv")
train_data.info()
train_data.head()
x = DataFrame(train_data['Survived'].value_counts()).reset_index()
x.columns = ['Survived', 'Count']
print(x)

srviv_pct = 100 * x.iloc[1, 1]/(x.iloc[0, 1] + x.iloc[1, 1])
print("Survival Percentage in Train Data:", srviv_pct)
sns.barplot(x='Survived', y='Count', data=x)
x = DataFrame(train_data['Pclass'].value_counts()).reset_index()
x.columns = ['Pclass', 'Count']
print(x)
print('----------------')

x = DataFrame(train_data['Sex'].value_counts()).reset_index()
x.columns = ['Sex', 'Count']
print(x)
print('----------------')

x = DataFrame(train_data['SibSp'].value_counts()).reset_index()
x.columns = ['SibSp', 'Count']
print(x)
print('----------------')

x = DataFrame(train_data['Parch'].value_counts()).reset_index()
x.columns = ['Parch', 'Count']
print(x)
print('----------------')

x = DataFrame(train_data['Embarked'].value_counts()).reset_index()
x.columns = ['Embarked', 'Count']
print(x)
train_data[train_data['Fare'] == 0]
train_data[train_data['Fare'] == 0].shape
train_data[train_data['Age'] == 0]
fig, (axis1, axis2) = plt.subplots(1, 2, sharey=True)

sns.kdeplot(data=train_data['Age'], ax=axis1)
sns.kdeplot(data=train_data['Fare'], ax=axis2)
sns.violinplot(data=train_data['Age']).set_title('Age')
sns.violinplot(data=train_data['Fare']).set_title("Fare")
sns.distplot(train_data['Age'], bins=50,
            kde_kws ={'color': 'darkgreen', 'alpha': 0.9, 'label': 'KDE Plot'},
            hist_kws={'color': 'red', 'alpha': 0.6, 'label': 'Histogram'})
sns.distplot(train_data['Fare'], bins=50,
            kde_kws ={'color': 'darkgreen', 'alpha': 0.6, 'label': 'KDE Plot'},
            hist_kws={'color': 'red', 'alpha': 0.6, 'label': 'Histogram'})
g = sns.FacetGrid(train_data, col="Survived", col_wrap=3,height=4)
g = (g.map(plt.hist, "Pclass"))
g = sns.FacetGrid(train_data, col="Survived", col_wrap=3,height=4)
g = (g.map(plt.hist, "SibSp"))
g = sns.FacetGrid(train_data, col="Survived", col_wrap=3,height=4)
g = (g.map(plt.hist, "Parch"))
grid = sns.FacetGrid(train_data, row='Embarked')
grid.map(plt.hist, 'Survived')
grid = sns.FacetGrid(train_data, row='Sex')
grid.map(plt.hist, 'Survived')
grid = sns.FacetGrid(train_data, row='Embarked', col='Sex')
grid.map(plt.hist, 'Survived')
grid = sns.FacetGrid(train_data, row='Embarked')
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', order=None, hue_order=None)
grid.add_legend()
grid = sns.FacetGrid(train_data, row='Sex', col='Survived')
grid.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train_data, row='Sex')
grid.map(sns.pointplot, 'Pclass', 'Survived', order=None, hue_order=None)
grid.add_legend()
from sklearn.impute import SimpleImputer
char_imputer = SimpleImputer(strategy="most_frequent")
x = DataFrame(char_imputer.fit_transform(train_data[["Embarked"]]))
x.columns= ['Imputed_Embarked']

train_data = pd.concat([train_data, x], axis=1)
train_data = train_data.drop(['Cabin', 'Embarked'], axis=1)
train_data[['Sex', 'Age']].groupby(['Sex']).mean().sort_values('Age', ascending=False)
train_data[['Pclass', 'Sex', 'Age']].groupby(['Pclass', 'Sex']).mean().sort_values('Age', ascending=False)
x = DataFrame(train_data[['Pclass', 'Sex', 'Age']].groupby(['Pclass', 'Sex']).mean().reset_index())

for i in range(train_data.shape[0]):
    if np.isnan(train_data.loc[i, 'Age']):
        for j in range(0, 6):
            if train_data.loc[i, 'Sex'] == x.loc[j, 'Sex'] and train_data.loc[i, 'Pclass'] == x.loc[j, 'Pclass']:
                train_data.loc[i, 'Age'] = x.loc[j, 'Age']
train_data.info()
train_data['Family'] = train_data['SibSp'] + train_data['Parch']

grid = sns.FacetGrid(train_data, col='Survived')
grid.map(plt.hist, 'Family', bins=20)
train_data['Age_Bin'] = pd.qcut(train_data['Age'], 10, labels=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
train_data['Age_Bin'] = train_data['Age_Bin'].astype(int)
train_data[['Sex', 'Age_Bin', 'Survived']].groupby(['Sex', 'Age_Bin']).mean().sort_values('Survived', ascending=False)
train_data[['Age_Bin', 'Survived']].groupby(['Age_Bin']).mean().sort_values('Survived', ascending=False)
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1}).astype(int)
train_data['Imputed_Embarked'] = train_data['Imputed_Embarked'].map({'S': 0, 'Q': 1, 'C': 2}).astype(int)
train_data['Pclass_Sex'] = train_data['Pclass'] * train_data['Pclass'] + train_data['Sex']
train_data[['Pclass_Sex', 'Survived']].groupby('Pclass_Sex').mean().sort_values('Survived', ascending=False)
from sklearn.preprocessing import StandardScaler
std_scaler_Fare = StandardScaler()
std_scaler_Age = StandardScaler()
train_data['Fare'] = std_scaler_Fare.fit_transform(train_data[['Fare']])
train_data['Age']  = std_scaler_Age.fit_transform(train_data[['Age']])
train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

train_data['Title'] = train_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train_data['Title'] = train_data['Title'].replace('Mlle', 'Miss')
train_data['Title'] = train_data['Title'].replace('Ms', 'Miss')
train_data['Title'] = train_data['Title'].replace('Mme', 'Mrs')

train_data['Title'] = train_data['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}).astype(int)

train_data['Title'] = train_data['Title'].fillna(0)
train_data.head()
train_data_model = train_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket'], axis=1)
train_data_label = train_data['Survived']

train_data_model.info()
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
svc = SVC(random_state=42)
svc.fit(train_data_model, train_data_label)
acc_svc = round(svc.score(train_data_model, train_data_label) * 100, 2)
print(acc_svc)
perceptron = Perceptron(random_state=42)
perceptron.fit(train_data_model, train_data_label)
acc_perceptron = round(perceptron.score(train_data_model, train_data_label) * 100, 2)
print(acc_perceptron)
linear_svc = LinearSVC(random_state=42, max_iter=10000)
linear_svc.fit(train_data_model, train_data_label)
acc_linear_svc = round(linear_svc.score(train_data_model, train_data_label) * 100, 2)
print(acc_linear_svc)
sgd = SGDClassifier(random_state=42)
sgd.fit(train_data_model, train_data_label)
acc_sgd = round(sgd.score(train_data_model, train_data_label) * 100, 2)
print(acc_sgd)
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(train_data_model, train_data_label)
acc_decision_tree = round(decision_tree.score(train_data_model, train_data_label) * 100, 2)
print(acc_decision_tree)
random_forest = RandomForestClassifier(random_state=42, n_estimators=100)
random_forest.fit(train_data_model, train_data_label)
acc_random_forest = round(random_forest.score(train_data_model, train_data_label) * 100, 2)
print(acc_random_forest)
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
cv_score = cross_val_score(random_forest, train_data_model, train_data_label, cv=10, scoring='accuracy')

print("Mean Score     :", cv_score.mean())
print("Std Dev Score  :", cv_score.std())
importances = pd.DataFrame({'Feature':train_data_model.columns,'Importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('Importance',ascending=False).set_index('Feature')
importances.head(15)
train_data_model = train_data_model.drop(['SibSp', 'Parch'], axis=1)

random_forest.fit(train_data_model, train_data_label)
acc_random_forest = round(random_forest.score(train_data_model, train_data_label) * 100, 2)
print(acc_random_forest)
cv_score = cross_val_score(random_forest, train_data_model, train_data_label, cv=10, scoring='accuracy')

print("Mean Score     :", cv_score.mean())
print("Std Dev Score  :", cv_score.std())

importances = pd.DataFrame({'Feature':train_data_model.columns,'Importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('Importance',ascending=False).set_index('Feature')
importances.head(15)
param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10, 25], "min_samples_split" : [10, 12, 16, 18], "n_estimators": [100, 700, 1500]}
rf = RandomForestClassifier(n_estimators=100, max_features='auto', oob_score=True, random_state=1, n_jobs=-1)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1)

grid_search.fit(train_data_model, train_data_label)

grid_search.best_params_
random_forest = RandomForestClassifier(criterion = "gini", 
                                       min_samples_leaf = 1, 
                                       min_samples_split = 16,   
                                       n_estimators=1500, 
                                       max_features='auto', 
                                       oob_score=True, 
                                       random_state=1, 
                                       n_jobs=-1)
random_forest.fit(train_data_model, train_data_label)
acc_random_forest = round(random_forest.score(train_data_model, train_data_label) * 100, 2)
print(acc_random_forest)
print("oob score:", round(random_forest.oob_score_, 4)*100, "%")
cv_score = cross_val_score(random_forest, train_data_model, train_data_label, cv=10, scoring='accuracy')

print("Mean Score     :", cv_score.mean())
print("Std Dev Score  :", cv_score.std())
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, precision_recall_curve, classification_report

train_label_predict = random_forest.predict(train_data_model)
print("Confusion Matrix: \n", confusion_matrix(train_data_label, train_label_predict))

print("Precision: ", precision_score(train_data_label, train_label_predict))
print("Recall   : ", recall_score(train_data_label, train_label_predict))
print("f1 Score : ", f1_score(train_data_label, train_label_predict))
print(classification_report(train_data_label, train_label_predict, target_names=['Not-Survived', 'Survived']))
train_data_label_scores = random_forest.predict_proba(train_data_model)
train_data_label_scores = train_data_label_scores[:,1]

precision, recall, threshold = precision_recall_curve(train_data_label, train_data_label_scores)
def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="Precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="Recall", linewidth=5)
    plt.xlabel("Threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])

plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show()
def plot_precision_vs_recall(precision, recall):
    plt.plot(recall, precision, "g--", linewidth=2.5)
    plt.ylabel("Recall", fontsize=19)
    plt.xlabel("Precision", fontsize=19)
    plt.axis([0, 1.5, 0, 1.5])

plt.figure(figsize=(14, 7))
plot_precision_vs_recall(precision, recall)
plt.show()
false_positive_rate, true_positive_rate, thresholds = roc_curve(train_data_label, train_data_label_scores)

def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()
print(roc_auc_score(train_data_label, train_data_label_scores))
test_data.info()
test_data[np.isnan(test_data['Fare'])]
min_fare = np.min(test_data['Fare'])

for i in range(test_data.shape[0]):
    if np.isnan(test_data.loc[i, 'Fare']):
        test_data.loc[i, 'Fare'] = min_fare
test_data[np.isnan(test_data['Fare'])]
test_data.info()
x = DataFrame(char_imputer.transform(test_data[["Embarked"]]))
x.columns= ['Imputed_Embarked']

test_data = pd.concat([test_data, x], axis=1)

test_data = test_data.drop(['Cabin', 'Embarked'], axis=1)

x = DataFrame(test_data[['Pclass', 'Sex', 'Age']].groupby(['Pclass', 'Sex']).mean().reset_index())

for i in range(test_data.shape[0]):
    if np.isnan(test_data.loc[i, 'Age']):
        for j in range(0, 6):
            if test_data.loc[i, 'Sex'] == x.loc[j, 'Sex'] and test_data.loc[i, 'Pclass'] == x.loc[j, 'Pclass']:
                test_data.loc[i, 'Age'] = x.loc[j, 'Age']

test_data['Family'] = test_data['SibSp'] + test_data['Parch']

test_data['Age_Bin'] = pd.qcut(test_data['Age'], 10, labels=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
test_data['Age_Bin'] = test_data['Age_Bin'].astype(int)

test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1}).astype(int)
test_data['Imputed_Embarked'] = test_data['Imputed_Embarked'].map({'S': 0, 'Q': 1, 'C': 2}).astype(int)

test_data['Pclass_Sex'] = test_data['Pclass'] * test_data['Pclass'] + test_data['Sex']

test_data['Fare'] = std_scaler_Fare.transform(test_data[['Fare']])
test_data['Age']  = std_scaler_Age.transform(test_data[['Age']])

test_data['Title'] = test_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

test_data['Title'] = test_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test_data['Title'] = test_data['Title'].replace('Mlle', 'Miss')
test_data['Title'] = test_data['Title'].replace('Ms', 'Miss')
test_data['Title'] = test_data['Title'].replace('Mme', 'Mrs')

test_data['Title'] = test_data['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}).astype(int)

test_data['Title'] = test_data['Title'].fillna(0)

## Prepare final data for model
test_data_model = test_data.drop(['PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch'], axis=1)
test_data_model.info()
test_data_predictions = random_forest.predict(test_data_model)
pred = DataFrame(test_data_predictions)
pred.columns = ['Survived']
predictions = pd.concat([test_data['PassengerId'], pred], axis=1)
predictions.to_csv("Submission_RF.csv", index=False)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
classifier = Sequential([Dense(128, activation='relu', input_shape=(train_data_model.shape[1], )),
                         Dropout(rate=0.1),
                         Dense(64, activation='relu'),
                         Dropout(rate=0.1),
                         Dense(1, activation='sigmoid')])

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['Accuracy'])
classifier.fit(train_data_model, train_data_label,
                    batch_size=20,
                    epochs=20)
test_data_predictions = classifier.predict(test_data_model)
pred = DataFrame(test_data_predictions)
pred.columns = ['Survived']
predictions = pd.concat([test_data['PassengerId'], pred], axis=1)
predictions.to_csv("Submission_ANN.csv", index=False)
