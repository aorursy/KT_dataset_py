from sklearn import datasets

from sklearn import model_selection

from sklearn import tree

from sklearn.metrics import r2_score
(X,y) = datasets.load_boston(return_X_y = True)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.33, random_state=42)
RegressionTree = tree.DecisionTreeRegressor(criterion="mse",max_depth=4,min_samples_split=50,min_samples_leaf=10)

RegressionTree.fit(X_train,y_train)

tree.plot_tree(RegressionTree) 
y_predict = RegressionTree.predict(X_test)

r2_score(y_test, y_predict)