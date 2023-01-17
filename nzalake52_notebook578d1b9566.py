import pandas as pd

import numpy as np

from matplotlib import pyplot

import seaborn as sb



redWineQual = pd.read_csv('../input/redWineQuality.csv', delimiter=';')
print('The features of the redwine dataset are as follows:')

print(redWineQual.columns)



print('Glimpse of the data')

redWineQual.head()
print('Summary Statistics of the dataset:')

redWineQual.describe()
print('Visualizing data using histograms for all the features')



%matplotlib inline

sb.set()

redWineQual.hist(figsize=(10, 10))

print('As we can see from the above histograms all the features have different distributions, the quality of wine with values 5 and 6 are much more than others.')
pyplot.figure(figsize=(13, 13))

pyplot.title('Correlation of the features')

sb.heatmap(redWineQual.astype(float).corr(), annot=True)

print('Alcohol with highest correlation is followed by sulphates, citric acid, fixed acidity and total sulfur dioxide')
print('Function to detect outliers')

def iqr(data):

    qtr1, qtr3 = np.percentile(data, [25, 75])

    iqr = qtr3 - qtr1

    lower_b = qtr1 - (iqr * 1.5)

    upper_b = qtr3 + (iqr * 1.5)

    return np.where(np.logical_and(data > lower_b, data < upper_b), data, np.median(data))



print('Using the above function for all the features')



redWineQual['fixed acidity'] = iqr(redWineQual['fixed acidity'])

redWineQual['volatile acidity'] = iqr(redWineQual['volatile acidity'])

redWineQual['citric acid'] = iqr(redWineQual['citric acid'])

redWineQual['residual sugar'] = iqr(redWineQual['residual sugar'])

redWineQual['chlorides'] = iqr(redWineQual['chlorides'])

redWineQual['free sulfur dioxide'] = iqr(redWineQual['free sulfur dioxide'])

redWineQual['total sulfur dioxide'] = iqr(redWineQual['total sulfur dioxide'])

redWineQual['density'] = iqr(redWineQual['density'])

redWineQual['pH'] = iqr(redWineQual['pH'])

redWineQual['sulphates'] = iqr(redWineQual['sulphates'])

redWineQual['alcohol'] = iqr(redWineQual['alcohol'])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(redWineQual[['fixed acidity', 'citric acid',

                                                                'residual sugar', 'free sulfur dioxide',

                                                                'total sulfur dioxide', 'sulphates',

                                                                'alcohol']], redWineQual['quality'], random_state=0,

                                                    test_size=0.3)

print('Splitting the dataset in train and test datasets')
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()

LR.fit(X_train, y_train)

print('The train dataset score: ', LR.score(X_train, y_train))

print('The test dataset score: ',LR.score(X_test, y_test))

print('From the scores we can see that the accuracy of the model is low with accuracy on train data as 54.7% and test data as 60%')
from sklearn.metrics import confusion_matrix

CM = confusion_matrix(y_test, LR.predict(X_test))

import seaborn as sb

sb.heatmap(pd.DataFrame(CM), annot=True)

print('The confusion matrix for the multiclass distribution is as follows: ')
from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(n_estimators=7, n_jobs=3, max_features=5)

RFC.fit(X_train, y_train)

print('Train score', RFC.score(X_train, y_train))

print('Test scrore', RFC.score(X_test, y_test))

print('Building a random forest classifier for the data\n', 'As we can see the model overfits')

from sklearn.metrics import confusion_matrix

CM = confusion_matrix(y_test, RFC.predict(X_test))

import seaborn as sn

sn.heatmap(pd.DataFrame(CM), annot=True)

print('Confusion matrix for the predicted data')