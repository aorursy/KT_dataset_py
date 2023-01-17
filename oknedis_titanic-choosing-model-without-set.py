import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
train_data.head()
#Checking missing values and data types
train_data.info()
#Looking the describe for explore outliers
train_data.describe(include="all")
train_data.shape, test_data.shape
#Cut the target feature
y = train_data['Survived']
x_train = train_data.copy()
x_train = x_train.drop(['Survived'], axis=1)
#Checking all name of features
x_train.columns.values
#Choosing essential features and creating functions for preprocessing data
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

imputer = SimpleImputer(strategy='most_frequent')

def data_preprocessing (data):
    data = data[features]
    data = pd.get_dummies(data, drop_first=True)
    return data

def my_imputer (fit_data, imput_data):
    imputer.fit(fit_data)
    data = pd.DataFrame(imputer.transform(imput_data))
    data.columns = fit_data.columns
    return data
#Preprocessing train_data using functions above
x_train_preprocessed = data_preprocessing(x_train)
x_train_preprocessed = my_imputer(x_train_preprocessed, x_train_preprocessed)
x_train_preprocessed.head() #Check preprocessed data
#Check preprocessed data
x_train_preprocessed.info()
#For choosing best parameters of our model using GridSearchCV
parameters = {'n_estimators': [100, 150, 200, 250],
                'max_features': np.arange(4, 9),
                'max_depth': np.arange(3, 10),
    
}

clf = GridSearchCV(
            estimator=RandomForestClassifier(),
            param_grid=parameters,
            scoring='accuracy',
            cv=5)
clf.fit(x_train_preprocessed, y)
cv_results = pd.DataFrame(clf.cv_results_)

cv_results.columns
param_columns = [
    column
    for column in cv_results.columns
    if column.startswith('param_')
]

score_columns = ['mean_test_score']

cv_results = (cv_results[param_columns + score_columns]
              .sort_values(by=score_columns, ascending=False))

cv_results.head(10)
#Checking the best parameters
clf.best_params_
test_data_preprocessed = data_preprocessing(test_data)
test_data_preprocessed = my_imputer(x_train_preprocessed, test_data_preprocessed)
test_data_preprocessed.head()
test_data_preprocessed.info()
model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=6)
model.fit(x_train_preprocessed, y)
predictions = model.predict(test_data_preprocessed)
predictions
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")