import pandas as pd

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
data = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

data.head()
X = data.drop(columns = ['Unnamed: 32','diagnosis', 'id'])

labels = data[['diagnosis']]
y = labels.apply(lambda x: x=='M')

y.head(20)
X_train,X_eval,y_train, y_eval = train_test_split(X,y)

X_eval, X_test, y_eval, y_test = train_test_split(X_eval,y_eval, test_size = .5)
model = XGBClassifier(n_estimators = 1000, max_depth = 3)
model.fit(X_train, y_train, eval_set =[(X_eval, y_eval)], early_stopping_rounds = 100, verbose = 10)
test_predictions = model.predict(X_test)
score = accuracy_score(y_test, test_predictions)

print( 'Accuracy: ', score)