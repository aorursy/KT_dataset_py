import numpy as np

import pandas as pd
train_filepath = "../input/train.csv"

test_filepath = "../input/test.csv"



Data_train = pd.read_csv(train_filepath, index_col = 'PassengerId')

X_test = pd.read_csv(test_filepath, index_col = 'PassengerId')
Data_train_sur = Data_train[(Data_train['Survived'] == 1)]

Data_train_non_sur = Data_train[(Data_train['Survived'] == 0)]

Data_train_sur.head()
import seaborn as sns

import matplotlib.pyplot as plt



sns.kdeplot(data = Data_train_sur['Age'], label = 'Survived')

sns.kdeplot(data = Data_train_non_sur['Age'], label = 'Non-Survived')

plt.title("Distribution of Age")
sns.kdeplot(data = Data_train_sur['Fare'], label = 'Survived')

sns.kdeplot(data = Data_train_non_sur['Fare'], label = 'Non Survived')

plt.title("Distribution of Fare")
sns.countplot(x="Sex", hue="Survived", data=Data_train) 

plt.title("Distribution of Sex")
sns.countplot(x="Pclass", hue="Survived", data=Data_train) 

plt.title("Distribution of Pclass")
sns.countplot(x="Embarked", hue="Survived", data=Data_train) 

plt.title("Distribution of Embarked")
sns.countplot(x="SibSp", hue="Survived", data=Data_train) 

plt.title("Distribution of SibSp")
sns.countplot(x="Parch", hue="Survived", data=Data_train) 

plt.title("Distribution of Parch")
y_train = Data_train.Survived #target column

X_train = Data_train.drop(['Survived'], axis = 1) # Input columns



X_train.info()
X_train["child"] = X_train["Age"].apply(lambda x: 1 if x < 16 else 0)

X_train["family"] = X_train["SibSp"] + X_train["Parch"]

#X_train['NumEmbarked'] = X_train["Embarked"].apply(lambda x: 1 if x != 'NaN' else 0)
cols_with_missing_train = [col for col in X_train.columns

                     if X_train[col].isnull().any()]

object_cols = ['Embarked', 'Sex']

all_need_cols = ['Sex', 'Embarked', 'Pclass', 'family', 'Fare', 'child']

cols_with_missing_train
#X_train['NUmEmbarked'].value_counts()
X_train['Age'].fillna(X_train['Age'].mean(), inplace = True)

X_train['Embarked'].fillna('C', inplace = True)

X_train.head(10)

#X_train
cols_with_missing_test = [col for col in X_test.columns

                     if X_test[col].isnull().any()]

cols_with_missing_test
X_test['Age'].fillna(X_test['Age'].mean())

X_test['Fare'].fillna(X_test['Fare'].mean())

X_test["child"] = X_test["Age"].apply(lambda x: 1 if x < 16 else 0)

X_test["family"] = X_test["SibSp"] + X_test["Parch"]

#X_test['NumEmbarked'] = X_test["Embarked"].apply(lambda x: 1 if x != 'NaN' else 0)

X_test.info()
from sklearn.preprocessing import LabelEncoder



label_X_train = X_train.copy()

label_X_test = X_test.copy()



label_encoder = LabelEncoder()

for col in object_cols:

    label_X_train[col] = label_encoder.fit_transform(X_train[col])

    label_X_test[col] = label_encoder.transform(X_test[col])



label_X_train.head()

X_train_need = label_X_train[all_need_cols]



from sklearn.model_selection import train_test_split



X_train_real, X_valid, y_train_real, y_valid = train_test_split(X_train_need, y_train)

X_train_real.info()
from xgboost import XGBClassifier



my_model = XGBClassifier(learning_rate=0.02, max_depth=2, 

silent=True, objective='binary:logistic')



X_test_need = label_X_test[all_need_cols]



from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score



param_test = {

    'n_estimators': range(30, 50, 2),

    'max_depth': range(2, 7, 1)

}



grid_search = GridSearchCV(estimator = my_model, param_grid = param_test, 

scoring='accuracy', cv=5)

grid_search.fit(X_train_real, y_train_real)

grid_search.best_index_, grid_search.best_params_, grid_search.best_score_
y_test = grid_search.predict(X_test_need)

y_valid_predict = grid_search.predict(X_valid)



output = pd.DataFrame({'PassengerId' : X_test_need.index,

                     'Survived' : y_test})

output.to_csv('submission11.csv', index = False)

print(output)