%matplotlib inline
import pandas as pd



df = pd.read_csv('../input/googleplaystore.csv')

df.info()
df.head()
# Trim the dataset by removing paid apps

data = df.drop(df.index[df['Type'] != 'Free'])

data.drop('Type', 1, inplace=True)

data.drop('Price', 1, inplace=True)

data.info()
# Trim the dataset by removing free apps without ratings

data.drop(data.index[data['Rating'].isnull()], inplace=True)

data.info()
# Remove free apps without current application version or minimum android version

data.drop(data.index[data['Current Ver'].isnull()], inplace=True)

data.drop(data.index[data['Android Ver'].isnull()], inplace=True)

data.info()
from sklearn import preprocessing



# Integer-encode application names

le = preprocessing.LabelEncoder()

data['App'] = le.fit_transform(data['App'])
# One-hot encode application categories

data['Category'] = pd.Categorical(data['Category'])

data = pd.concat([data, pd.get_dummies(data['Category'], prefix='cat', drop_first=True)], axis=1)

data.drop('Category', 1, inplace=True)
# Convert reviews to integers

data['Reviews'] = data['Reviews'].astype(float)
import numpy as np

from sklearn.impute import SimpleImputer



# Convert all sizes to Megabytes

data['Size'] = data['Size'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)

data['Size'] = data['Size'].apply(lambda x: float(str(x).replace('k', '')) / 1024 if 'k' in str(x) else x)



# Use median imputation for 'Varies with device' values

data['Size'] = data['Size'].replace(regex='Varies with device', value=np.nan)

imp = SimpleImputer(missing_values=np.nan, strategy='median')

data['Size'] = imp.fit_transform(data[['Size']]).ravel()

data['Size'] = data['Size'].astype(float)
# Convert installs to floats

data['Installs'] = data['Installs'].apply(lambda x: str(x).strip('+').replace(',', '')).astype(float)
# Integer-encode content ratings

le = preprocessing.LabelEncoder()

data['Content Rating'] = le.fit_transform(data['Content Rating'])
# Integer-encode genres

le = preprocessing.LabelEncoder()

data['Genres'] = le.fit_transform(data['Genres'])
from datetime import date, datetime



# Convert last updated to the number of days away from January 1, 2019

data['Last Updated'] = data['Last Updated'].apply(lambda x: str(date(2019, 1, 1) - datetime.strptime(x, '%B %d, %Y').date()))

data['Last Updated'] = data['Last Updated'].apply(lambda x: float(x.split(' ', 1)[0]))
# Round current versions to only major and minor versions

data['Current Ver'] = data['Current Ver'].apply(lambda x: x.split(' ', 1)[0])

data['Current Ver'] = data['Current Ver'].apply(lambda x: '.'.join(x.split('.', 2)[:2]) if '.' in x else x)  

data['Current Ver'] = data['Current Ver'].apply(lambda x: ''.join([c for c in x if c in '1234567890.']))

data['Current Ver'] = data['Current Ver'].apply(lambda x: '0' if x == '' or x == '.' else x)

data['Current Ver'] = data['Current Ver'].astype(float)

data['Current Ver'] = data['Current Ver'].apply(lambda x: 0 if x > 10 else x)



# Use median imputation for missing values

imp = SimpleImputer(missing_values=0, strategy='median')

data['Current Ver'] = imp.fit_transform(data[['Current Ver']]).ravel()
# Round minimum android versions to only major and minor versions

data['Android Ver'] = data['Android Ver'].apply(lambda x: x.split(' ', 1)[0])

data['Android Ver'] = data['Android Ver'].apply(lambda x: '.'.join(x.split('.', 2)[:2]) if '.' in x else x)  

data['Android Ver'] = data['Android Ver'].apply(lambda x: ''.join([c for c in x if c in '1234567890.']))

data['Android Ver'] = data['Android Ver'].apply(lambda x: '0' if x == '' or x == '.' else x)



# Use median imputation for missing values

imp = SimpleImputer(missing_values=0, strategy='median')

data['Android Ver'] = imp.fit_transform(data[['Android Ver']]).ravel()
scaler = preprocessing.MinMaxScaler()

data[['Reviews', 'Size', 'Installs', 'Last Updated', 'Current Ver', 'Android Ver']] =  scaler.fit_transform(data[['Reviews', 'Size', 'Installs', 'Last Updated', 'Current Ver', 'Android Ver']])
from sklearn.model_selection import train_test_split



# Split data into training and test sets

X = data.loc[:, data.columns != 'Rating']

y = data['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score



# Analyse a linear regression model's performance using R2 scores and squared errors

def analyse_linear(model, y_pred, y_test):    

    # Compute R2 score of model on training and test sets

    print('Variance (R\N{SUPERSCRIPT TWO} score) for train set: %.3f' % model.score(X_train, y_train))

    print('Variance (R\N{SUPERSCRIPT TWO} score) for test set: %.3f' % r2_score(y_test, y_pred))

    

    # Depict a histogram of the squared errors of the data points in the test set

    squared_error = np.square(y_pred - y_test)

    print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))

    plt.figure()

    plt.hist(squared_error)

    plt.title('Distribution of squared errors for dataset')

    plt.xlabel('Squared error')

    plt.ylabel('No. of test data points')

    plt.show()

    

    plt.figure(figsize=(12,7))

    sns.regplot(y_pred, y_test, color='blue', marker='x')

    plt.title('Linear regression model on the test set')

    plt.xlabel('Predicted ratings')

    plt.ylabel('Actual ratings')

    plt.show()



def do_linear_regression(X_train, X_test, y_train, y_test):

    model = linear_model.LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    analyse_linear(model, y_pred, y_test)

    

do_linear_regression(X_train, X_test, y_train, y_test)
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPRegressor



def find_hidden_units(X_train, y_train):

    # Set the parameters by cross-validation

    tuned_parameters = []

    model_solvers = ['adam', 'sgd']

    for i in np.arange(1, 10):

        for j in np.arange(0, 2):

            tuned_parameters.append({'hidden_layer_sizes': [(i,)], 'solver':[model_solvers[j]]})



    print('Tuning hyper-parameters...')

    mlp = MLPRegressor(max_iter=2000, alpha=1e-4, tol=1e-4, random_state=42)

    clf = GridSearchCV(mlp, tuned_parameters, cv=5)

    clf.fit(X_train, y_train)



    print("Best parameters set found on development set:")

    print(clf.best_params_)

    print('Grid scores on development set:')

    means = clf.cv_results_['mean_test_score']

    stds = clf.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, clf.cv_results_['params']):

        print('%0.3f (+/-%0.03f) for %r' % (mean, std * 2, params))

        print()

              

find_hidden_units(X_train, y_train)
from sklearn.neural_network import MLPRegressor



# Train and visualise loss changes while training the netwrok

def train_network(X_train, y_train):

    model = MLPRegressor(max_iter=100, alpha=1e-4, solver='sgd', tol=1e-4, learning_rate='adaptive', 

                          hidden_layer_sizes=(1,), early_stopping=True, validation_fraction=0.2)



    model.fit(X_train, y_train)

    print('Training set score: %f' % model.score(X_train, y_train))

    print('Training set loss: %f' % model.loss_)

    

    plt.figure()

    plt.title('Changes in performance during model training')

    plt.plot(model.loss_curve_, linestyle='-', color='red', label='loss curve')

    plt.plot(model.validation_scores_, linestyle='--', color='blue', label='cross validation score curve')

    plt.ylabel('Average loss/cross validation score over time')

    plt.xlabel('No. of iterations')

    plt.legend(loc='lower right')

    plt.show()

    

    return model

    

trained_model = train_network(X_train, y_train)
from sklearn.neighbors import KNeighborsRegressor



model = KNeighborsRegressor(n_neighbors=15)

model.fit(X_train, y_train)

accuracy = model.score(X_test,y_test)

'Accuracy: ' + str(np.round(accuracy*100, 2)) + '%'