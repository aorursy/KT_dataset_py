!pip install sweetviz
import pandas as pd
import sweetviz
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import model_selection, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
train.head()
test.head()
train.info()
test.info()
report = sweetviz.compare([train, 'Train'], [test, 'Test'], 'Survived')
report.show_html('Report.html')
numerical_features = train[['Age', 'Fare']]
numerical_features
sns.set(style = 'whitegrid')
fig = plt.figure(figsize = (10,5))
for i in range(len(numerical_features.columns)):
    fig.add_subplot(1, 2, i+1)
    sns.boxplot(y = numerical_features.iloc[:,i])
plt.tight_layout()
plt.show()
fig = plt.figure(figsize = (10,5))
for i in range(len(numerical_features.columns)):
    fig.add_subplot(1, 2, i+1)
    sns.scatterplot(x = numerical_features.iloc[:,i], y = train.Survived)
plt.tight_layout()
plt.show()
train.drop(train[train.Fare > 300].index, inplace = True)
train.Age = train.groupby('Pclass').Age.apply(lambda x : x.fillna(x.median()))
test.Age = test.groupby('Pclass').Age.apply(lambda x : x.fillna(x.median()))
train.Age.isna().sum()
test.Age.isna().sum()
test.Fare.fillna(test.Fare.mean(), inplace = True)
test.Fare.isna().sum()
train.drop(columns = 'Cabin', inplace = True)
test.drop(columns = 'Cabin', inplace = True)
train.Embarked.fillna(train.Embarked.mode().iloc[0], inplace = True)
train.Embarked.isna().sum()
train.isna().sum()
test.isna().sum()
train.set_index('PassengerId', inplace = True)
test.set_index('PassengerId', inplace = True)
train.head()
test.head()
report = sweetviz.compare([train, 'Train'], [test, 'Test'], 'Survived')
report.show_html('Report_after_EDA.html')
train.drop(columns = ['Name', 'Ticket'], inplace = True)
test.drop(columns = ['Name', 'Ticket'], inplace = True)
train.head()
test.head()
cat_columns = ['Sex', 'Embarked']
train = pd.get_dummies(train, columns = cat_columns)
train.head().T
test = pd.get_dummies(test, columns = cat_columns)
test.head().T
x = np.array(train.drop(columns = 'Survived'))
x = preprocessing.scale(x)
y = np.array(train.Survived)
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.2)
model = RandomForestClassifier(n_estimators = 100)
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print('Model accuracy:', accuracy)
y_predict = model.predict(x_test)
mse = mean_squared_error(y_test, y_predict)
print('Mean Square error:', mse)
x_predict = np.array(test)
x_predict = preprocessing.scale(x_predict)
y_predict = model.predict(x_predict)
submission = pd.DataFrame(data = {'PassengerID':np.array(test.index.values), 'Survived':y_predict})
submission.to_csv('my_submission.csv', index = False)
print('Submission file saved')
