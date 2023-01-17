import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
trainY = train.median_house_value 
train = train.drop(columns=["median_house_value"])
train.shape
train["persons/room"] = train["population"]/train["total_rooms"]
test["persons/room"] = test["population"]/test["total_rooms"]
train["persons/room"]
train["persons/bedroom"] = train["population"]/train["total_bedrooms"]
test["persons/bedroom"] = test["population"]/test["total_bedrooms"]
train["persons/bedroom"]
train.hist(bins=200, figsize=(25,20))
train = train.drop(columns=["Id"])
plt.xlabel('Persons per Room')
train["persons/room"].hist(bins=200, range=(0,5))
test["persons/room"].hist(bins=200, range=(0,5))
plt.xlabel('Persons per Bedroom')
train["persons/bedroom"].hist(bins=200, range=(0,10))
test["persons/bedroom"].hist(bins=200, range=(0,10))
import seaborn
plt.figure(figsize=(10,10))
plt.title("Matriz de correlação")
seaborn.heatmap(train.corr(), annot=True, linewidths=0.2)
train = train.drop(columns=["persons/room"])
test = test.drop(columns=["persons/bedroom"])
from sklearn import tree
regression_tree = tree.DecisionTreeRegressor(criterion='mse', splitter='best', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,  min_impurity_decrease=0.0, presort=True)
model = regression_tree.fit(train, trainY)
regression_tree.predict(train)
# from IPython.display import Image  
# from sklearn import tree
# import pydotplus

# dot_data = tree.export_graphviz(regression_tree, out_file=None, 
#                                 feature_names=train.columns)

# graph = pydotplus.graph_from_dot_data(dot_data)  

# # Mostrar grafo e salvar um pdf
# graph.write_pdf("Tree.pdf")
# Image(graph.create_png())
from sklearn.model_selection import learning_curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
plt.figure(figsize=(12,8))
plot_learning_curve(regression_tree,"Regression Tree Learning Curve", train, trainY)
regression_tree = tree.DecisionTreeRegressor(criterion='mse', splitter='best', min_samples_split=2, min_samples_leaf=1, max_depth=7, min_weight_fraction_leaf=0.0,  min_impurity_decrease=0.0, presort=True)
model = regression_tree.fit(train, trainY)
regression_tree.predict(train)
plt.figure(figsize=(12,8))
plot_learning_curve(regression_tree,"Regression Tree Learning Curve", train, trainY)
regression_tree.score(train, trainY)
from sklearn.linear_model import LinearRegression
LR = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
LR.fit(train, trainY)
LR.score(train, trainY)
plt.figure(figsize=(12,8))
plot_learning_curve(LR,"Linear Regression Learning Curve", train, trainY)
testId = test["Id"]
test = test.drop(columns=["Id"])
predictions = LR.predict(test)
#predictions.to_csv('predictions.csv')
predictions
predict = pd.DataFrame(index=testId)
predict["median_house_value"] = predictions
predict.to_csv('predictions.csv')
from sklearn.ensemble import AdaBoostRegressor
boost = AdaBoostRegressor(base_estimator=LR, n_estimators=100, learning_rate=1.0, loss='linear', random_state=None)
boost.fit(train, trainY)
boost.score(train, trainY)
plt.figure(figsize=(12,8))
plot_learning_curve(boost,"AdaBoost Learning Curve", train, trainY)
predictions = boost.predict(test)
predict = pd.DataFrame(index=testId)
predict["median_house_value"] = predictions
predict.to_csv('predictions.csv')
from sklearn import neural_network

neural_net = neural_network.MLPRegressor(hidden_layer_sizes=(100,),
                                       activation='relu', solver='adam',
                                       learning_rate='adaptive', max_iter=800,
                                       learning_rate_init=0.01, warm_start = True, alpha=0.01)
neural_net.fit(train, trainY)

print ("Neural Net score: ", str(neural_net.score(train, trainY)*100), "%")
neural_net
plt.figure(figsize=(12,8))
plot_learning_curve(neural_net,"Neural Network Learning Curve", train, trainY) #62
from catboost import CatBoostRegressor

regressor = CatBoostRegressor(loss_function='RMSE')
regressor.fit(train, trainY)
regressor.score(train, trainY)
from sklearn.ensemble import RandomForestRegressor
num=100
forest = RandomForestRegressor(n_estimators=num, criterion='mse', min_samples_split=5, 
                      min_samples_leaf=5, min_weight_fraction_leaf=0.0, max_features="sqrt", 
                     random_state=None, verbose=0, warm_start=True)
forest.fit(train, trainY)
print (forest.score(train, trainY))
predictions = forest.predict(test)
plt.figure(figsize=(12,8))
plot_learning_curve(forest,"Random Forest Learning Curve", train, trainY)
from sklearn.ensemble import GradientBoostingRegressor
boosting = GradientBoostingRegressor()
# boosting = GradientBoostingClassifier(loss='deviance', learning_rate=0.02, n_estimators=100, subsample=1.0, 
#                                       criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, 
#                                       min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0,
#                                       min_impurity_split=None, init=forest, random_state=None, max_features=None,
#                                       verbose=0, max_leaf_nodes=None, warm_start=True, presort='auto',
#                                       validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)



boosting.fit(train, trainY)
print (boosting.score(train, trainY))

predictions = boosting.predict(test)
plt.figure(figsize=(12,8))
plot_learning_curve(boosting,"Gradient Boosting Learning Curve", train, trainY)
predict = pd.DataFrame(index=testId)
predict["median_house_value"] = predictions
predict.to_csv('predictions.csv')
from sklearn.ensemble import AdaBoostRegressor
adaboost1 = AdaBoostRegressor(base_estimator=neural_net, n_estimators=10, learning_rate=0.01, random_state=None)
adaboost2 = AdaBoostRegressor(base_estimator=forest, n_estimators=10, learning_rate=0.01, random_state=None)


adaboost1.fit(train, trainY)
adaboost2.fit(train, trainY)

print ("AdaBoost on Neural Network score: ", str(adaboost1.score(train, trainY)*100), "%")
print ("AdaBoost on Random Forest score: ", str(adaboost2.score(train, trainY)*100), "%")
predictions = adaboost2.predict(test)
predict = pd.DataFrame(index=testId)
predict["median_house_value"] = predictions
predict.to_csv('predictions.csv')
predictions = neural_net.predict(test)
print(predictions)
predict = pd.DataFrame(index=testId)
predict["median_house_value"] = predictions
predict.to_csv('predictions.csv')