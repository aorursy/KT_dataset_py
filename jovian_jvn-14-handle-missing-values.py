import pandas as pd

train = pd.read_csv('http://bit.ly/kaggletrain')

test = pd.read_csv('http://bit.ly/kaggletest', nrows=175)
train = train[['Survived', 'Age', 'Fare', 'Pclass']]

test = test[['Age', 'Fare', 'Pclass']]
# count the number of NaNs in each column

train.isna().sum()
test.isna().sum()
label = train.pop('Survived')
# new in 0.22: this estimator (experimental) has native support for NaNs

from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import HistGradientBoostingClassifier
clf = HistGradientBoostingClassifier()
# no errors, despite NaNs in train and test!

clf.fit(train, label)

clf.predict(test)