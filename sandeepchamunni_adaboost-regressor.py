from sklearn import datasets

from sklearn import model_selection

from sklearn import ensemble

from sklearn.metrics import r2_score

from sklearn import tree
(X,y) = datasets.load_boston(return_X_y = True)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.33, random_state=42)
stump = tree.DecisionTreeRegressor(max_depth=1)
AdaRegressor = ensemble.AdaBoostRegressor(base_estimator = stump, loss="linear", n_estimators=10, random_state=20)

AdaRegressor.fit(X_train, y_train)
y_predict = AdaRegressor.predict(X_test)

r2_score(y_test, y_predict)