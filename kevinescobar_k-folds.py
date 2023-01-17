import pandas as pd

import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

data = pd.read_csv('../input/heart.csv')

X = (data.iloc[:,:-1]).as_matrix()

y = (data.iloc[:,2]).as_matrix()
y = y.reshape((-1,1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import Imputer

from sklearn.neural_network import MLPClassifier as mlp_c

#my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())

clf = mlp_c(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100,100), random_state=1)

clf.fit(X_train, y_train) 

y_hat = clf.predict(X_test)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf,y_test, y_hat, scoring='fowlkes_mallows_score')

print(scores)
resultado=scores.mean()

print(resultado)