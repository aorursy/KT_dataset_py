# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Reading the train and the test data.
trainData = pd.read_csv('../input/train.csv')
testData = pd.read_csv('../input/test.csv')

# Displaying a sample of the train data to get more detailed info
trainData.head()
trainData.describe()
trainData.dtypes
trainData.apply(lambda x: x.isnull().any())
pd.DataFrame({'percent_missing': trainData.isnull().sum() * 100 / len(trainData)})
pd.DataFrame({'percent_unique': trainData.apply(lambda x: x.unique().size/x.size*100)})
# Names of the features extarcted from the data
selFeatures = list(trainData.columns.values)
# Removing the target variable from the column values
targetCol = 'Survived'
selFeatures.remove(targetCol)

# Removing features with unique values
for i in selFeatures:
    if trainData.shape[0] == len(pd.Series(trainData[i]).unique()) :
        selFeatures.remove(i)
        
# Removing features with high percentage of missing values
selFeatures.remove('Cabin')
import seaborn as sns
sns.set(style="ticks")
plotFeatures = [x for x in selFeatures]
plotFeatures.append("Survived")
sns.pairplot(trainData[plotFeatures], hue="Survived")
# Also removing cabin and ticket features for the initial run.
selFeatures.remove('Ticket')
        
print("Target Class: '"+ targetCol + "'")
print('Features to be investigated: ')
print(selFeatures)
def handle_categorical_na(df):
    ## replacing the null/na/nan values in 'Cabin' attribute with 'X'
#     df.Cabin = df.Cabin.fillna(value='X')
#     ## Stripping the string data in 'Cabin' and 'Ticket' features of numeric values and duplicated characters
#     df.Cabin = [''.join(set(filter(str.isalpha, s))) for s in df.Cabin]
#     df.Ticket = [''.join(set(filter(str.isalpha, s))) for s in df.Ticket]
#     ## replacing the '' values in 'Ticket' attribute with 'X'
#     df.Ticket.replace(to_replace='',value='X',inplace=True)
    ## Imputing the null/na/nan values in 'Age' attribute with its mean value 
    df.Age.fillna(value=df.Age.mean(),inplace=True)
    ## replacing the null/na/nan values in 'Embarked' attribute with 'X'
    df.Embarked.fillna(value='X',inplace=True)
    return df
from sklearn.model_selection import train_test_split
seed = 7
np.random.seed(seed)
X_train, X_test, Y_train, Y_test = train_test_split(trainData[selFeatures], trainData.Survived, test_size=0.2)

X_train = handle_categorical_na(X_train)
X_test = handle_categorical_na(X_test)

## using One Hot Encoding for handling categorical data
X_train = pd.get_dummies(X_train,columns=['Embarked','Sex'],prefix=['Embarked','Sex'])
X_test = pd.get_dummies(X_test,columns=['Embarked','Sex'],prefix=['Embarked','Sex'])
common_col = [x for x in X_test.columns if x in X_train.columns]
X_test = X_test[common_col]

missing_col = [x for x in X_train.columns if x not in X_test.columns]
## Inserting missing columns in test data
for val in missing_col:
    X_test.insert(X_test.shape[1], val, pd.Series(np.zeros(X_test.shape[0])))
def rf_hyperparameter_optimization():
    from sklearn.model_selection import RandomizedSearchCV
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    from sklearn.ensemble import RandomForestRegressor
    # Using the random grid to search for best hyperparameters
    # Creating the base model to tune
    rf = RandomForestRegressor()
    # Random search of parameters, using 5 fold cross validation, 
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(X_train, Y_train)
    return rf_random.best_params_

## based on the hyper parameter optimization, the below model is built.
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=1400, min_samples_split=5, min_samples_leaf=4, max_features= 'sqrt', max_depth= 80, bootstrap= True)
rf_model.fit(X_train,Y_train)
# Fetching predictions
Y_pred = rf_model.predict(X_test)
# Calculating the test accuracy
from sklearn import metrics
score_rf = metrics.accuracy_score(Y_test, Y_pred)
print("Test Accuracy:",score_rf)
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
in_shape = X_train.shape[1]

def create_model(optimizer='Adam', neurons=50):
    # Initialize the constructor
    model = Sequential()
    # Input - Layer
    model.add(Dense(neurons, input_dim=in_shape, activation=activation))
    # Hidden - Layers
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, activation = activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, activation = activation))
    # Output- Layer
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
def nn_hyperparameter_optimization():
    model = KerasClassifier(build_fn=create_model, verbose=0)
    # defining the grid search parameters
    neurons = [65, 75, 85]
    batch_size= [10, 20, 30, 40]
    epochs= [10, 20, 30, 40]
    optimizer = ['RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    dropout_rate = [0.1, 0.2, 0.3, 0.4, 0.5]
    param_grid = dict(neurons=neurons,
                      optimizer=optimizer,
                      batch_size=batch_size,
                      epochs=epochs,
                      activation=activation,
                      dropout_rate=dropout_rate)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)
    grid_result = grid.fit(X_train, Y_train)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return grid_result
## based on the hyper parameter optimization, the below model is built.
nn_model = Sequential()
# Input - Layer
nn_model.add(Dense(65, input_dim=in_shape, activation='relu'))
# Hidden - Layers
nn_model.add(Dropout(0.2))
nn_model.add(Dense(65, activation = 'relu'))
nn_model.add(Dropout(0.2))
nn_model.add(Dense(65, activation = 'relu'))
# Output- Layer
nn_model.add(Dense(1, activation='sigmoid'))
nn_model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])
nn_model.fit(X_train, Y_train,
          batch_size=20,
          epochs=20,
          verbose=1,
          validation_data=(X_test, Y_test))

score = nn_model.evaluate(X_test, Y_test, verbose=2)
score_nn = score[1]
print('Test loss:', score[0])
print('Test accuracy:', score[1])
xTest = testData[selFeatures]
xTest = handle_categorical_na(xTest)
## using One Hot Encoding for handling categorical data
xTest = pd.get_dummies(xTest,columns=['Embarked','Sex'],prefix=['Embarked','Sex'])
common_col = [x for x in xTest.columns if x in X_train.columns]
xTest = xTest[common_col]
missing_col = [x for x in X_train.columns if x not in xTest.columns]
## Inserting missing columns in test data
for val in missing_col:
    xTest.insert(xTest.shape[1], val, pd.Series(np.zeros(xTest.shape[0])))
col_names = xTest.columns
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
xTest = my_imputer.fit_transform(xTest)
xTest = pd.DataFrame(xTest)
xTest.columns = col_names

submission = pd.DataFrame()
## Comparing and submitting the best result
if score_nn>score_rf:
    predictions = nn_model.predict_classes(xTest)
    predictions = [x[0] for x in predictions]
else:
    predictions = rf_model.predict(xTest)
submission = pd.DataFrame({'PassengerId': testData.PassengerId, 'Survived': predictions})
submission.to_csv('submission.csv', index=False)
submission.head()
