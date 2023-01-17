# Imports

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.neural_network import MLPClassifier

from sklearn import ensemble

from sklearn import svm

import time

import random

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler

import math

import matplotlib.ticker as plticker

import matplotlib.patches as mpatches

import matplotlib.lines as mlines

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn import metrics 

from sklearn.ensemble import AdaBoostClassifier

import matplotlib.cm as cm

from matplotlib.colors import Normalize

# Read in data

big_df = pd.read_csv("../input/titanic-cleaned-data/train_clean.csv")

big_df.info()

# util functions



def eval_for_conclusion(model_id, clf, test_x, test_y):

    y_pred = clf.predict(test_x)

    print(classification_report(test_y, y_pred))

    print(confusion_matrix(test_y, y_pred))

    accuracy = metrics.accuracy_score(test_y, y_pred)

    precision = metrics.precision_score(test_y, y_pred)

    recall = metrics.recall_score(test_y, y_pred)

    print("Final {0} model accuracy:".format(model_id), accuracy)

    print("Final {0} model precision:".format(model_id), precision) 

    print("Final {0} model recall:".format(model_id), recall) 

    return {"model":model_id, "recall":recall, "accuracy":accuracy, "precision":precision}



def drop_columns(df, columns_to_drop):

    for col in columns_to_drop:

        del df[col]   

def split_test_train(train_size, all_data):

    msk = np.random.rand(len(all_data)) < train_size

    train_df = all_data[msk]

    test_df = all_data[~msk]

    train_y = train_df["Survived"]

    train_x = train_df.drop("Survived", axis=1)

    test_y = test_df["Survived"]

    test_x  = test_df.drop("Survived", axis=1)

    return (train_x, train_y, test_x, test_y)



def cross_validate(all_data, model):

    depth = []

    all_y = all_data["Survived"]

    all_x  = all_data.drop("Survived", axis=1)

    # Perform k-fold cross validation 

    scores = cross_val_score(estimator=model, X=all_x, y=all_y, cv=5, n_jobs=4)

    depth.append((i,scores.mean()))

    return depth

    

def train_and_test(all_data, model):

    test_scores = []

    train_scores = []

    times = []

    for i in range(1,10):

        (train_x, train_y, test_x, test_y) = split_test_train(0.1 * i, big_df)

        #print("len test: ", len(test_x), ", len train: ", len(train_x))

        start = time.time()

        #TODO iterations

        model.fit(train_x, train_y)

        end = time.time()

        times.append(end - start)

        pred_test_y = model.predict(test_x) # TODO add wallclock time

        test_score = round(model.score(test_x, test_y) * 100, 2)

        pred_train_y = model.predict(train_x)

        train_score = round(model.score(train_x, train_y) * 100, 2)

        test_scores.append(test_score)

        train_scores.append(train_score)

    return (test_scores, train_scores, times)



def plot_data(x_vars, x_label, all_y_vars, y_var_labels, y_label, title, y_bounds=None):

    plt.rcParams["figure.figsize"] = (4,3)

    colors = ['red','orange','black','green','blue','violet']

    i = 0

    for y_var in all_y_vars:

#         if i == 2: # don't plot when i = 1 for cv

#             x_vars = x_vars[1:]

        plt.plot(x_vars, y_var, 'o-', color=colors[i % 6], label=y_var_labels[i])

        i += 1

    plt.xlabel(x_label)

    plt.ylabel(y_label)

    plt.title(title)

    if y_bounds != None:

        plt.ylim(y_bounds)

    leg = plt.legend();

    plt.show()



def evaluate_model(all_data, model, model_id):

    (test_scores, train_scores, times) = train_and_test(all_data, model)

    print("{0} train timings (seconds): {1}".format(model_id, times))

    print("{0} test set scores: {1} ".format(model_id, test_scores))

    print("{0} train set scores: {1}".format(model_id, train_scores))

    plot_data([x * 10 for x in range(1,10)], "Percentage of data in training set", [test_scores, train_scores],\

              ["test_scores", "train_scores"], "Accuracy", "{0} Accuracy Over Train/Test Split".format(model_id), (50,105))

    plot_data([x * 10 for x in range(1,10)], "Percentage of data in training set", [times],

             ["times"], "Train time in Seconds", "{0} Time Spent Training Over Train/Test Split".format(model_id))

    return (test_scores, train_scores, times)



def plot_grid_search(grid_results, plotting_func, title, x_label, y_label, grid_size, model_handles):

    means = grid_results.cv_results_['mean_test_score']

    stds = grid_results.cv_results_['std_test_score']

    params = grid_results.cv_results_['params']

    plt.rcParams["figure.figsize"] = grid_size

    plt.xlabel(x_label)

    plt.ylabel(y_label)

    plt.title(title)

    plt.subplots

    ax = plt.subplot()

    

    for mean, std, params in zip(means, stds, params):

        #print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

        plotting_func(mean, params, plt, ax)

    if model_handles: plt.legend(handles=model_handles)

    plt.show()





#def grid_search(model, params, x_train, y_train, x_test, y_test):

    



#TODO come up with graphing function that takes in two arrays of test and train and plots them
columns_to_drop = ["Cabin", "Name", "Ticket", "Parch", "Embarked", "Title", "PassengerId"]  # TODO include reasoning for dropping these

drop_columns(big_df, columns_to_drop)

is_male = {"male": 1, "female": 0}

big_df["Sex"].replace(is_male, inplace=True)

# Decision Tree

dt_model = DecisionTreeClassifier()

(dt_test_scores, dt_train_scores, dt_times) = evaluate_model(big_df, dt_model, "Titanic Decision Tree")

# choose best test split and k fold value

optimal_test_split = dt_test_scores.index(max(dt_test_scores)) * 0.1

print("max index for means was: ", optimal_test_split * 10)

(dt_grid_train_x, dt_grid_train_y, dt_grid_test_x, dt_grid_test_y) = split_test_train(optimal_test_split, big_df)

dt_param_grid = {"criterion":["gini","entropy"], "max_depth":[3,4,5,6,7,8,9,10], "min_samples_split":[3,5,7]}  #"splitter":["best", "random"], 

dt_grid_results = GridSearchCV(dt_model, dt_param_grid, cv=5).fit(dt_grid_train_x, dt_grid_train_y)

dt_means = dt_grid_results.cv_results_['mean_test_score']

dt_stds = dt_grid_results.cv_results_['std_test_score']

dt_params = dt_grid_results.cv_results_['params']



max_depths = []

accuracies = []

plt.xlabel("Max Depth of Decision Tree")

plt.ylabel("Accuracy")

plt.title("Titanic Decision Tree Grid Search, Accuracy as a Function of Max_depth")

for mean, std, params in zip(dt_means, dt_stds, dt_params):

    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    accuracies.append(mean)

    max_depths.append(params["max_depth"])

plt.plot(max_depths, accuracies, 'o', color="red")

plt.show()

print("best params ", dt_grid_results.best_params_)



def plotting_func_dt(mean, params, plt, ax):

    x_var = "max_depth"

    #print(params["hidden_layer_sizes"])

    color_map = {"gini":"b", "entropy":"red"}

    ax.plot(params[x_var], mean, "o", color=color_map[params["criterion"]])



blue_patch = mpatches.Patch(color='blue', label='gini')

red_patch = mpatches.Patch(color='red', label='entropy')

handles = [blue_patch, red_patch]

plot_grid_search(dt_grid_results, plotting_func_dt, "Titanic Decision Tree Training Accuracy as a Function of Max_depth", "max_depth", "accuracy",(6,4), handles)



eval_for_conclusion("Titanic Decision Tree", dt_grid_results, dt_grid_test_x, dt_grid_test_y)
knn_classifier = KNeighborsClassifier()

evaluate_model(big_df, knn_classifier, "Titanic knn baseline")

knn_param_grid = {"n_neighbors":[i for i in range(2,21)]+[k*10 for k in range(3,11)]}

(knn_grid_train_x, knn_grid_train_y, knn_grid_test_x, knn_grid_test_y) = split_test_train(0.8, big_df)

knn_grid_results = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=5).fit(knn_grid_train_x, knn_grid_train_y)

# All results



knn_means = knn_grid_results.cv_results_['mean_test_score']

knn_stds = knn_grid_results.cv_results_['std_test_score']

knn_params = knn_grid_results.cv_results_['params']



for mean, std, params in zip(knn_means, knn_stds, knn_params):

    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

print('Best parameters found:\n', knn_grid_results.best_params_, "with score of: ", max(knn_grid_results.cv_results_['mean_test_score']))



def plotting_func_knn(mean, params, plt, ax):

    x_var = "n_neighbors"

    #print(params["hidden_layer_sizes"])

    ax.plot(params[x_var], mean, "o", color="r")

    

plot_grid_search(knn_grid_results, plotting_func_knn, "Titanic KNN Training Accuracy as a Function of Number of Neigbors", "n_neighbors", "accuracy",(6,4), [])



eval_for_conclusion("Titanic KNN", knn_grid_results, knn_grid_test_x, knn_grid_test_y)





# (knn_test_scores, knn_train_scores) = train_and_test(big_df, knn_classifier)

# knn_cv_scores = cross_validate(big_df, knn_classifier)



# print("knn test set scores: ", knn_test_scores)

# print("knn train set scores: ", knn_train_scores) 

# print("knn cross validation set scores: ", knn_cv_scores) 

neural_net_classifier = MLPClassifier(max_iter=10000) #, alpha=0.01, hidden_layer_sizes=(6, 3), random_state=1)

# tried with6,3 and works great. Other dimensions are horrible

evaluate_model(big_df, neural_net_classifier, "Titanic NeuralNet baseline")
# neural_net_classifier = MLPClassifier(max_iter=10000, alpha=0.01, hidden_layer_sizes=(6, 3), random_state=1)

# # tried with6,3 and works great. Other dimensions are horrible

# (nn_test_scores, nn_train_scores, nn_cv_scores) = evaluate_model(big_df, neural_net_classifier, "NeuralNet baseline model")

# optimal_test_split = nn_test_scores.index(max(nn_test_scores)) * 0.1

# optimal_test_split = max(optimal_test_split, 0.7)



# print("max index for means was: ", optimal_test_split * 10)





mlp = MLPClassifier(max_iter=10000)

#     'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],

#     'activation': ['tanh', 'relu'],

#     'solver': ['sgd', 'adam'],

#     'learning_rate': ['constant','adaptive'],

parameter_space = {

    'activation': ['tanh', 'relu'],

    'alpha': [0.0001, 0.0005, 0.01, 0.05, 0.1],

    "hidden_layer_sizes": [(3,), (5,), (7,)]

    

}



nn_grid_clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)

(nn_train_x, nn_train_y, nn_test_x, nn_test_y) = split_test_train(0.8, big_df)

nn_grid_clf.fit(nn_train_x, nn_train_y)

# Best paramete set

print('Best parameters found:\n', nn_grid_clf.best_params_)



# All results

nn_means = nn_grid_clf.cv_results_['mean_test_score']

nn_stds = nn_grid_clf.cv_results_['std_test_score']

nn_params = nn_grid_clf.cv_results_['params']



for mean, std, params in zip(nn_means, nn_stds, nn_params):

    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

print("best params ", nn_grid_clf.best_params_)

nn_grid_score = nn_grid_clf.score(nn_test_x,nn_test_y)

print("nn grid search model score: ", nn_grid_score)





# mlp = MLPClassifier(max_iter=10000, hidden_layer_sizes=(6, 3), alpha=clf.best_params_["alpha"], activation=clf.best_params_["activation"])

# mlp.fit(nn_train_x, nn_train_y)

# nn_test_score = round(mlp.score(nn_test_x, nn_test_y) * 100, 2)

# print("nn grid search model score: ", nn_test_score)

# All results

# nn_means = nn_grid_clf.cv_results_['mean_test_score']

# nn_stds = nn_grid_clf.cv_results_['std_test_score']

# nn_params = nn_grid_clf.cv_results_['params']



# for mean, std, params in zip(nn_means, nn_stds, nn_params):

#     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

# print("best params ", nn_grid_clf.best_params_)





def plotting_func_nn(mean, params, plt, ax):

    x_var = "alpha"

    tick_spacing = 0.1

    layer_colors = {"(3,)":"orange", "(5,)":"red", "(7,)":"black"}

    activation_labels = {"tanh": "o", "relu":"s"}

    #print(params["hidden_layer_sizes"])

    layer_color_idx = str(params["hidden_layer_sizes"])

    activation_idx = params["activation"]

    ax.plot(math.log(params[x_var],10), mean, activation_labels[activation_idx], color=layer_colors[layer_color_idx])

    x_loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals

    ax.xaxis.set_major_locator(x_loc)

    y_loc = plticker.MultipleLocator(base=0.005) # this locator puts ticks at regular intervals

    ax.yaxis.set_major_locator(y_loc)

red_patch = mpatches.Patch(color='orange', label='(3,) layer')

orange_patch = mpatches.Patch(color='red', label='(5,) layer')

black_patch = mpatches.Patch(color='black', label='(7,) layer')

tanh = mlines.Line2D([], [],marker='o',

                         label='tanh')

relu = mlines.Line2D([], [],marker='s',

                         label='relu')

handles = [tanh, relu, red_patch, orange_patch, black_patch]



plot_grid_search(nn_grid_clf, plotting_func_nn, "Titanic NN accuracy as a function of log(alpha Param)", "log(alpha)", "accuracy",(5,7), handles)





boost_classifier = AdaBoostClassifier()

evaluate_model(big_df, boost_classifier, "Titanic Boosting_classifier")
boost_parameter_space = {

    'n_estimators': [i*10 for i in range(5,11)],

    'learning_rate': [ float(i) / 100 for i in range (1, 150, 10)]

}

boost_grid_clf = GridSearchCV(boost_classifier, boost_parameter_space, n_jobs=-1, cv=3)

(boost_train_x, boost_train_y, boost_test_x, boost_test_y) = split_test_train(0.1 * 8, big_df)

scaler = StandardScaler()

scaler.fit(boost_train_x)

boost_train_x = scaler.transform(boost_train_x)

boost_test_x = scaler.transform(boost_test_x)

boost_grid_clf.fit(boost_train_x, boost_train_y)



print("best Boost params ", boost_grid_clf.best_params_)

boost_grid_score = boost_grid_clf.score(boost_test_x,boost_test_y)

print("Bost grid search model test set score: ", boost_grid_score)

print('Best Boost parameters found through cv:\n', boost_grid_clf.best_params_, "with score of: ", max(boost_grid_clf.cv_results_['mean_test_score']))

def plotting_func_boost(mean, params, plt, ax):

    x_var = "learning_rate"

    cmap = cm.hot

    norm = Normalize(vmin=-110, vmax=-20)

    ax.plot(params[x_var], mean,"o", color=cmap(norm(-1*params["n_estimators"]))) #, markeredgecolor = "black")

    plt.ylim(.76, .835)

    plt.xlim(0, 1.5)



cmap = cm.hot

norm = Normalize(vmin=-110, vmax=-20)

yellow_patch = mpatches.Patch(color=cmap(norm(-50)), label='n_estimators=50')

red_patch = mpatches.Patch(color=cmap(norm(-80)), label='n_estimators=80')

black_patch = mpatches.Patch(color=cmap(norm(-110)), label='n_estimators=110')

handles = [yellow_patch, red_patch, black_patch] #[linear, rbf]#, red_patch, orange_patch, black_patch]





plot_grid_search(boost_grid_clf, plotting_func_boost, "Titanic Boosting accuracy as a function of learning rate", "learning rate", "accuracy",(5,8), handles)



eval_for_conclusion("Titanic Boosting", boost_grid_clf, boost_test_x, boost_test_y)

svm_classifier = svm.SVC()

evaluate_model(big_df, svm_classifier, "Titanic svm_classifier_baseline")



# (svm_test_scores, svm_train_scores) = train_and_test(big_df, svm_classifier)

# svm_cv_scores = cross_validate(big_df, svm_classifier)



# print("svm test set scores: ", svm_test_scores)

# print("svm train set scores: ", svm_train_scores) 

# print("svm cross validation set scores: ", svm_cv_scores) 



svm_parameter_space = {

    'kernel': ['poly', 'rbf'],

    'C': [ float(i) / 100 for i in range (1, 130, 5)]+[0.0001, 0.000001, 3,4,5,6,7],    

}

svm_grid_clf = GridSearchCV(svm_classifier, svm_parameter_space, n_jobs=-1, cv=3)

(svm_train_x, svm_train_y, svm_test_x, svm_test_y) = split_test_train(0.1 * 8, big_df)

scaler = StandardScaler()

scaler.fit(svm_train_x)

svm_train_x = scaler.transform(svm_train_x)

svm_test_x = scaler.transform(svm_test_x)

svm_grid_clf.fit(svm_train_x, svm_train_y)



print("best params ", svm_grid_clf.best_params_)

svm_grid_score = svm_grid_clf.score(svm_test_x,svm_test_y)

print("Titanic SVM grid search model test set score: ", svm_grid_score)

print('Titanic Best SVM parameters found through cv:\n', svm_grid_clf.best_params_, "with score of: ", max(svm_grid_clf.cv_results_['mean_test_score']))







# svm_classifier.fit(nn_train_x, nn_train_y)

# svm_svc = svm.SVC(kernel='rbf', gamma=0.013, C=1)

# evaluate_model(big_df, rbf_svc, "svm_classifier__kernel='rbf'__gamma=0.013__C=1")
def plotting_func_svm(mean, params, plt, ax):

    x_var = "C"

    tick_spacing = 0.1

    #layer_colors = {"(3,)":"orange", "(5,)":"red", "(7,)":"black"}

    kernel_labels = {"poly": "o", "rbf":"o"}

    kernel_colors = {"poly": "red", "rbf":"black"}

    #print(params["hidden_layer_sizes"])

    #layer_color_idx = str(params["hidden_layer_sizes"])

    kernel_idx = params["kernel"]

    ax.plot(params[x_var], mean, kernel_labels[kernel_idx], color=kernel_colors[kernel_idx])

    x_loc = plticker.MultipleLocator(base=0.5) # this locator puts ticks at regular intervals

    ax.xaxis.set_major_locator(x_loc)

    y_loc = plticker.MultipleLocator(base=0.005) # this locator puts ticks at regular intervals

    ax.yaxis.set_major_locator(y_loc)

    plt.ylim(.77, .83)



linear = mlines.Line2D([], [],marker='o',

                         label='poly', color="r")

rbf = mlines.Line2D([], [],marker='o',

                         label='rbf', color="black")

handles = [linear, rbf]#, red_patch, orange_patch, black_patch]



plot_grid_search(svm_grid_clf, plotting_func_svm, "Titanic SVM accuracy as a function of C", "C", "accuracy",(5,7), handles)

eval_df = pd.DataFrame(columns=["model", "recall", "accuracy", "precision"])

eval_df = eval_df.append(eval_for_conclusion("Titanic DT", dt_grid_results, dt_grid_test_x, dt_grid_test_y), ignore_index=True)

eval_df = eval_df.append(eval_for_conclusion("Titanic Boosting", boost_grid_clf, boost_test_x, boost_test_y), ignore_index=True)

eval_df = eval_df.append(eval_for_conclusion("Titanic SVM", svm_grid_clf, svm_test_x, svm_test_y), ignore_index=True)

eval_df = eval_df.append(eval_for_conclusion("Titanic KNN", knn_grid_results, knn_grid_test_x, knn_grid_test_y), ignore_index=True)

eval_df = eval_df.append(eval_for_conclusion("Titanic Neural Network", nn_grid_clf, nn_test_x, nn_test_y), ignore_index=True)

eval_df.head()
