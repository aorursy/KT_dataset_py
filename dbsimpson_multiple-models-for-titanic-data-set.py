import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
training = pd.read_csv('../input/titanic/train.csv')
training.head()
training.loc[training['Sex'] == 'male', 'Sex_Numeric'] = 0

training.loc[training['Sex'] == 'female', 'Sex_Numeric'] = 1



training.head()
training['Title'] = training.Name.apply(lambda x: x.split(',')[1].split('.')[0].replace(' ',''))
training.Title.unique()
training.loc[training['Title'] == 'Mr', 'Title_Numeric'] = 0

training.loc[training['Title'] == 'Mrs', 'Title_Numeric'] = 1

training.loc[training['Title'] == 'Miss', 'Title_Numeric'] = 2

training.loc[training['Title'] == 'Master', 'Title_Numeric'] = 3

training.loc[training['Title'] == 'Don', 'Title_Numeric'] = 4

training.loc[training['Title'] == 'Rev', 'Title_Numeric'] = 5

training.loc[training['Title'] == 'Dr', 'Title_Numeric'] = 6

training.loc[training['Title'] == 'Mme', 'Title_Numeric'] = 7

training.loc[training['Title'] == 'Ms', 'Title_Numeric'] = 8

training.loc[training['Title'] == 'Major', 'Title_Numeric'] = 9

training.loc[training['Title'] == 'Lady', 'Title_Numeric'] = 10

training.loc[training['Title'] == 'Sir', 'Title_Numeric'] = 11

training.loc[training['Title'] == 'Mlle', 'Title_Numeric'] = 12

training.loc[training['Title'] == 'Col', 'Title_Numeric'] = 13

training.loc[training['Title'] == 'Capt', 'Title_Numeric'] = 14

training.loc[training['Title'] == 'theCountess', 'Title_Numeric'] = 15

training.loc[training['Title'] == 'Jonkheer', 'Title_Numeric'] = 16

training.loc[training['Title'] == 'Dona', 'Title_Numeric'] = 17
training.head()
training.describe()
import missingno as msno

msno.matrix(training)
training.Age = training.Age.fillna(training.Age.median())
training.groupby('Embarked')['Ticket'].nunique()
training.Embarked = training.Embarked.fillna('S')
training.loc[training['Embarked'] == 'S', 'Embarked_Numeric'] = 0

training.loc[training['Embarked'] == 'Q', 'Embarked_Numeric'] = 1

training.loc[training['Embarked'] == 'C', 'Embarked_Numeric'] = 2
msno.matrix(training)
training = training.drop(columns = ['Cabin'])
numeric_training = training[['Age', 'SibSp', 'Parch', 'Fare']]

categorical_training = training[['Survived', 'Pclass', 'Sex_Numeric', 'Title_Numeric', 'Embarked_Numeric']]
import seaborn as sns

sns.heatmap(numeric_training.corr(), cmap="YlGnBu")
df_training = training[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_Numeric', 'Title_Numeric', 'Embarked_Numeric', 'Survived']]
X = df_training[['Pclass', 'Age', 'SibSp', 'Fare', 'Sex_Numeric', 'Title_Numeric', 'Embarked_Numeric']]

y = df_training[['Survived']]
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn import tree

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier
seed = 404

np.random.seed(seed)
gnb = GaussianNB()

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

cv_score = cross_val_score(gnb, X, y.values.ravel(), cv=kfold)

print('Gaussian Naive Bayes K-fold Scores:')

print(cv_score)

print()

print('Gaussian Naive Bayes Average Score:')

print(cv_score.mean())

print()
lr = LogisticRegression(max_iter = 2000)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

cv_score = cross_val_score(lr, X, y.values.ravel(), cv=kfold)

print('Logistic Regression K-fold Scores (training):')

print(cv_score)

print()

print('Logistic Regression Average Score:')

print(cv_score.mean())
dt = tree.DecisionTreeClassifier(random_state = 1)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

cv_score = cross_val_score(dt, X, y.values.ravel(), cv=kfold)

print('Decision Tree K-fold Scores:')

print(cv_score)

print()

print('Decision Tree Average Score:')

print(cv_score.mean())
knn = KNeighborsClassifier()

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

cv_score = cross_val_score(knn, X, y.values.ravel(), cv=kfold)

print('KNN K-fold Scores):')

print(cv_score)

print()

print('KNN Average Score:')

print(cv_score.mean())
rf = RandomForestClassifier(random_state = 1)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

cv_score = cross_val_score(rf, X, y.values.ravel(), cv=kfold)

print('Random Forest K-fold Scores:')

print(cv_score)

print()

print('Random Forest Average Score:')

print(cv_score.mean())
svc = SVC(probability = True)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

cv_score = cross_val_score(svc, X, y.values.ravel(), cv=kfold)

print('Support Vector Classification K-fold Scores:')

print(cv_score)

print()

print('Support Vector Classification Average Score:')

print(cv_score.mean())
xgb = XGBClassifier(random_state =1)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

cv_score = cross_val_score(xgb, X, y.values.ravel(), cv=kfold)

print('XGBoost Classifier K-fold Scores:')

print(cv_score)

print()

print('XGBoost Classifier Average Score:')

print(cv_score.mean())
def create_model():

    model = Sequential()

    

    model.add(Dense(7, input_dim=7, activation='relu'))

    model.add(Dense(14, activation='relu'))

    model.add(Dense(21, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))



    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



seed = 7

np.random.seed(seed)



model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)



kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

cv_score = cross_val_score(model, X, y, cv=kfold)

print('Neural Network K-fold Scores:')

print(cv_score)

print()

print('Neural Network Average Score:')

print(cv_score.mean())