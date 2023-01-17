import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
data = pd.read_csv('../input/train.csv')
data.head()
data.info()
sns.heatmap(data.isna())
data.Ticket.value_counts()
data.Sex.value_counts()
data.Embarked.value_counts()
data.SibSp.value_counts()
data.Parch.value_counts()
sns.distplot(data[~data.Fare.isna()]['Fare'])
sns.distplot(data[~data.Age.isna()]['Age'])
data.Survived.value_counts()
sns.distplot(data[~data.Age.isna()]['Age'])
data.head()
def prepare_data(initial_data, train_data=True):
    prepared_data = pd.DataFrame()
    prepared_data['sex'] = pd.get_dummies(initial_data.Sex, drop_first=True)
    prepared_data = pd.concat([prepared_data, initial_data[['SibSp', 'Parch', 'Pclass', 'Fare']]], axis=1)
    prepared_data = pd.concat([prepared_data, pd.get_dummies(initial_data.Embarked, drop_first=True)], axis=1)
    if train_data:
        prepared_data = pd.concat([prepared_data, initial_data['Survived']], axis=1)
    return prepared_data
train_data = prepare_data(data)
train_data.head()
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(train_data.drop('Survived', axis=1))
y = train_data['Survived']
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout


def build_classifier(optimizer,first_layer_units,second_lays_units, dropout_rate):
    classifier = Sequential()
    classifier.add(Dense(units = first_layer_units, kernel_initializer = 'uniform', activation = 'relu', input_dim = 7))
    classifier.add(Dropout(dropout_rate))
    classifier.add(Dense(units = second_lays_units, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, verbose = 0)

parameters = {'batch_size': [5, 15, 30],
              'epochs': [100],
              'first_layer_units': [7, 4],
              'second_lays_units': [4, 7],
              'dropout_rate': [0, 0.2, 0.3],
              'optimizer': ['adam', 'rmsprop']}

## This step may take a while
# grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 5)
# grid_search = grid_search.fit(X, y)
# print('Best parameters:', grid_search.best_params_)
# print('Best accuracy:', grid_search.best_score_)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
# Comparing classifiers

classifiers = [GradientBoostingClassifier(), GradientBoostingClassifier(n_estimators=300), RandomForestClassifier(),RandomForestClassifier(n_estimators=300), RandomForestClassifier(class_weight='balanced'), \
               ExtraTreesClassifier(), ExtraTreesClassifier(class_weight='balanced') ,ExtraTreesClassifier(n_estimators=50), LogisticRegression(), XGBClassifier(n_estimators=50), LinearSVC(), LinearSVC(class_weight='balanced'),  \
               KerasClassifier(build_fn=build_classifier, batch_size= 15, dropout_rate=0.15, epochs=100, first_layer_units= 7, optimizer='rmsprop', second_lays_units= 7, verbose=0)]

X_train, X_test, y_train, y_test = train_test_split(X, y)

accuracies = []

for classifier in classifiers:
    print(str(classifier))
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)
    accuracies.append({classifier: accuracy_score(y_test, pred)})
#     print(classification_report(y_test, pred))
    print(accuracy_score(y_test, pred))
# Generating submission file
classifier = GradientBoostingClassifier(n_estimators=300)
classifier.fit(X, y)
data_sub = pd.read_csv('../input/test.csv')
data_sub_prepared = prepare_data(data_sub, train_data=False)
# Filling NaN fare
data_sub_prepared.loc[data_sub_prepared[data_sub_prepared.Fare.isna()].index, 'Fare'] = data_sub_prepared[(data_sub_prepared.sex == 1) &( data_sub_prepared.SibSp ==0) &( data_sub_prepared.Parch ==0) &( data_sub_prepared.Pclass ==3)  &( data_sub_prepared.Q ==0)  &( data_sub_prepared.S ==1) ].Fare.mean()
data_sub_prepared = sc.transform(data_sub_prepared)
data_sub['Survived'] = classifier.predict(data_sub_prepared)
data_sub = data_sub[['PassengerId' , 'Survived']]
data_sub.to_csv('submission.csv', index=False)
data_sub.Survived.value_counts()