# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import time

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.neural_network import MLPClassifier

from sklearn import ensemble

from sklearn import svm

import random

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV

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







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



data_frames = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        file_path = os.path.join(dirname, filename)

        data_frames.append(pd.read_csv(file_path))

big_frame = pd.concat(data_frames, ignore_index=True)

big_frame.info()



# Any results you write to the current directory are saved as output.
# for col in big_frame:

#     print ("col is ", col, " unique values:  ",big_frame[col].unique())

#     print ("#####")

#eval_df = pd.DataFrame(data={"model":[], "recall":[], "accuracy":[], "precision":[], "training_time": []})

eval_df = pd.DataFrame(columns=["model", "recall", "accuracy", "precision"])

big_frame = big_frame.drop(columns=["tourney_id","tourney_name", "draw_size", "tourney_date", "match_num", "winner_id", "winner_seed", "tourney_level", "winner_ioc",\

                                    "winner_rank_points", "winner_entry","winner_name", "loser_id", "loser_seed", "loser_rank_points", "loser_entry", "loser_ioc",\

                                    "score", "round", "loser_name", "minutes","surface", "winner_hand", "winner_ht", "winner_age", "winner_rank", "loser_hand", "loser_ht",\

                                    "loser_age", "loser_rank" ])

big_frame = big_frame.dropna()

big_frame.info()

#big_frame["surface"].value_counts()

# filtered_df = big_frame[(big_frame.surface == "Hard")] # | (big_frame.surface == "Clay" )] #"Clay", "Grass", "Carpet"])]

filtered_df = big_frame[(big_frame.best_of == 3)]

big_frame.columns

"""filtered_df["best_of"].value_counts()

w

w_ace        44451 non-null float64

w_df         44451 non-null float64

w_svpt       44451 non-null float64

w_1stIn      44451 non-null float64

w_1stWon     44451 non-null float64

w_2ndWon     44451 non-null float64

w_SvGms      44451 non-null float64

w_bpSaved    44451 non-null float64

w_bpFaced    44451 non-null float64

l_ace        44451 non-null float64

l_df         44451 non-null float64

l_svpt       44451 non-null float64

l_1stIn      44451 non-null float64

l_1stWon     44451 non-null float64

l_2ndWon     44451 non-null float64

l_SvGms      44451 non-null float64

l_bpSaved    44451 non-null float64

l_bpFaced    44451 non-null float64

"""

lost_df = filtered_df[['l_ace', 'l_df', 'l_svpt','l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced']]

won_df = filtered_df[['w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon','w_SvGms', 'w_bpSaved', 'w_bpFaced']]

lost_df = lost_df.rename(columns={'l_ace':'ace', 'l_df':'df', 'l_svpt':'svpt','l_1stIn':'1stIn', 'l_1stWon':'1stWon', 'l_2ndWon':'2ndWon', \

                                  'l_SvGms':'SvGms', 'l_bpSaved':'bpSaved', 'l_bpFaced':'bpFaced'})

won_df = won_df.rename(columns={'w_ace':'ace', 'w_df':'df', 'w_svpt':'svpt','w_1stIn':'1stIn', 'w_1stWon':'1stWon', 'w_2ndWon':'2ndWon',\

                                'w_SvGms':'SvGms', 'w_bpSaved':'bpSaved', 'w_bpFaced':'bpFaced'})



all_players = pd.concat(data_frames, ignore_index=True)

won_df["won"] = won_df['svpt']**0

lost_df["won"] = lost_df['svpt']*0



big_df = pd.concat([won_df, lost_df], ignore_index=True)

big_df = big_df.sample(frac=0.1, random_state=1)

big_df = big_df.drop(columns=['svpt'])

big_df.info

# lost_df["ace_pct"] = filtered_df.apply(lambda x: x.l_ace / x.l_svpt)

# won_df["ace_pct"] = filtered_df.apply(lambda x: x.w_ace / x.w_svpt)
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



def split_test_train(train_size, all_data):

    msk = np.random.rand(len(all_data)) < train_size

    train_df = all_data[msk]

    test_df = all_data[~msk]

    train_y = train_df["won"]

    train_x = train_df.drop("won", axis=1)

    test_y = test_df["won"]

    test_x  = test_df.drop("won", axis=1)

    return (train_x, train_y, test_x, test_y)



def cross_validate(all_data, model):

    depth = []

    all_y = all_data["won"]

    all_x  = all_data.drop("won", axis=1)

    for i in range(2,10):

        # Perform n-fold cross validation 

        scores = cross_val_score(estimator=model, X=all_x, y=all_y, cv=i, n_jobs=4)

        # print("i scores for cv: ", scores)

        depth.append((i,scores.mean()))

    # print(depth)

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

    colors = ['red','orange','black','green','blue','violet']

    plt.rcParams["figure.figsize"] = (4,3)



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

    cv_scores = cross_validate(all_data, model)

    print("{0} train timings (seconds): {1}".format(model_id, times))

    print("{0} test set scores: {1} ".format(model_id, test_scores))

    print("{0} train set scores: {1}".format(model_id, train_scores))

    print("{0} cross validation set scores: {1}".format(model_id, cv_scores))

    plot_data([x * 10 for x in range(1,10)], "Percentage of data in training set", [test_scores, train_scores],\

              ["test_scores", "train_scores"], "Accuracy", "{0} Accuracy Over Train/Test Split".format(model_id), (50,103))

    plot_data([x[0] for x in cv_scores], "Number of folds", [[x[1] for x in cv_scores]],

             ["cross_validation_accuracy"], "Accuracy", "{0} Accuracy Over Different Cross Validation Values of K".format(model_id), (0.3,1))

    plot_data([x * 10 for x in range(1,10)], "Percentage of data in training set", [times],

             ["times"], "Train time in Seconds", "{0} Time Spent Training Over Train/Test Split".format(model_id))

    return (test_scores, train_scores, times, cv_scores)



def plot_grid_search(grid_results, plotting_func, title, x_label, y_label, grid_size, model_handles):

    plt.rcParams["figure.figsize"] = grid_size

    means = grid_results.cv_results_['mean_test_score']

    stds = grid_results.cv_results_['std_test_score']

    params = grid_results.cv_results_['params']

    plt.xlabel(x_label)

    plt.ylabel(y_label)

    plt.title(title)

    plt.subplots

    ax = plt.subplot()

    for mean, std, params in zip(means, stds, params):

        #print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

        plotting_func(mean, params, plt, ax)

    if handles: plt.legend(handles=model_handles)

    plt.show()





#def grid_search(model, params, x_train, y_train, x_test, y_test):

    



#TODO come up with graphing function that takes in two arrays of test and train and plots them
neural_net_classifier = MLPClassifier(max_iter=10000, random_state=1)

# tried with6,3 and works great. Other dimensions are horrible

evaluate_model(big_df, neural_net_classifier, "Tennis NeuralNet Baseline")
from sklearn.preprocessing import StandardScaler

#     'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],

#     'activation': ['tanh', 'relu'],

#     'solver': ['sgd', 'adam'],

#     'learning_rate': ['constant','adaptive'],

parameter_space = {

    'activation': ['tanh', 'relu'],

    'alpha': [10 ** i for i in range (-1, -8, -1)],

    "hidden_layer_sizes": [(5,5), (10,10), (15,15)]

    

}

nn_clf = GridSearchCV(neural_net_classifier, parameter_space, n_jobs=-1, cv=3)

(nn_train_x, nn_train_y, nn_test_x, nn_test_y) = split_test_train(0.1 * 8, big_df)

scaler = StandardScaler()

scaler.fit(nn_train_x)

nn_train_x = scaler.transform(nn_train_x)

nn_test_x = scaler.transform(nn_test_x)





nn_clf.fit(nn_train_x, nn_train_y)

eval_for_conclusion("Tennis Neural Network", nn_clf, nn_test_x, nn_test_y)

# Best parameters set

print('Best parameters found:\n', nn_clf.best_params_, "with score of: ", max(nn_clf.cv_results_['mean_test_score']))







#pl

def plotting_func_nn(mean, params, plt, ax):

    x_var = "alpha"

    tick_spacing = 0.1

    layer_colors = {"(5, 5)":"orange", "(10, 10)":"red", "(15, 15)":"black"}

    activation_labels = {"tanh": "o", "relu":"s"}

    #print(params["hidden_layer_sizes"])

    layer_color_idx = str(params["hidden_layer_sizes"])

    activation_idx = params["activation"]

    ax.plot(math.log(params[x_var],10), mean, activation_labels[activation_idx], color=layer_colors[layer_color_idx])

    x_loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals

    ax.xaxis.set_major_locator(x_loc)

    y_loc = plticker.MultipleLocator(base=0.005) # this locator puts ticks at regular intervals

    ax.yaxis.set_major_locator(y_loc)

red_patch = mpatches.Patch(color='orange', label='(5, 5) layer')

orange_patch = mpatches.Patch(color='red', label='(10, 10) layer')

black_patch = mpatches.Patch(color='black', label='(15, 15) layer')





tanh = mlines.Line2D([], [],marker='o',

                         label='tanh')

relu = mlines.Line2D([], [],marker='s',

                         label='relu')

handles = [red_patch,orange_patch,black_patch,tanh, relu]



plot_grid_search(nn_clf, plotting_func_nn, "Tennis NN accuracy as a function of log(alpha Param)", "log(alpha)", "accuracy",(5,7), handles)

# # Decision Tree

# dt_model = DecisionTreeClassifier()

# (dt_test_scores, dt_train_scores, dt_times, dt_cv_scores) = evaluate_model(big_df, dt_model, "Decision Tree")

# # choose best test split and k fold value

# optimal_test_split = dt_test_scores.index(max(dt_test_scores)) * 0.1

# print("max index for means wsa: ", optimal_test_split * 10)

# (dt_grid_train_x, dt_grid_train_y, dt_grid_test_x, dt_grid_test_y) = split_test_train(optimal_test_split, big_df)

# dt_param_grid = {"criterion":["gini","entropy"], "max_depth":[5, 10, 15, 30], "splitter":["best", "random"]}

# dt_grid_results = GridSearchCV(dt_model, dt_param_grid, cv=5).fit(dt_grid_train_x, dt_grid_train_y)

# dt_best_params = dt_grid_results.best_params_

# grid_dt_model = DecisionTreeClassifier(

#     criterion=dt_best_params["criterion"], max_depth=dt_best_params["max_depth"])



# grid_dt_model.fit(dt_grid_train_x, dt_grid_train_y)



# pred_test_y = grid_dt_model.predict(dt_grid_test_x) #TODO add wallclock time

# test_score = round(grid_dt_model.score(dt_grid_test_x, dt_grid_test_y) * 100, 2)

# print("test score for decision tree model: ", test_score, dt_best_params)

# #get other interesting information from the model



dt_model = DecisionTreeClassifier()

(dt_test_scores, dt_train_scores, dt_times, dt_cv_scores) = evaluate_model(big_df, dt_model, "Tennis Decision Tree")

# choose best test split and k fold value

optimal_test_split = dt_test_scores.index(max(dt_test_scores)) * 0.1

print("max index for means was: ", optimal_test_split * 10)

(dt_grid_train_x, dt_grid_train_y, dt_grid_test_x, dt_grid_test_y) = split_test_train(optimal_test_split, big_df)

#or

dt_param_grid = {"criterion":["gini","entropy"], "max_depth":[x for x in range(2,31)], "min_samples_split":[3,5,7]}  #"splitter":["best", "random"], 

# max depth below 6 didnt work for this dataset since the data is more complex

dt_grid_results = GridSearchCV(dt_model, dt_param_grid, cv=5).fit(dt_grid_train_x, dt_grid_train_y)

# dt_means = dt_grid_results.cv_results_['mean_test_score']

# dt_stds = dt_grid_results.cv_results_['std_test_score']

# dt_params = dt_grid_results.cv_results_['params']



# max_depths = []

# accuracies = []

# plt.xlabel("Max Depth of Decision Tree")

# plt.ylabel("Accuracy")

# plt.title("Tennis Decision Tree Grid Search, Accuracy as a Function of Max_depth")

# for mean, std, params in zip(dt_means, dt_stds, dt_params):

#     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

#     accuracies.append(mean)

#     max_depths.append(params["max_depth"])

# plt.plot(max_depths, accuracies, 'o', color="red")

# plt.show()

# print("best params ", dt_grid_results.best_params_)

# print("\nClassification results for decision tree model on test set:")

# dt_y_pred = dt_grid_results.predict(dt_grid_test_x)

# print(classification_report(dt_grid_test_y, dt_y_pred))

# print(confusion_matrix(dt_grid_test_y, dt_grid_results.predict(dt_grid_test_x)))

# print("DT model accuracy:", metrics.accuracy_score(dt_grid_test_y, dt_grid_results.predict(dt_grid_test_x))) 







def plotting_func_dt(mean, params, plt, ax):

    x_var = "max_depth"

    #print(params["hidden_layer_sizes"])

    color_map = {"gini":"b", "entropy":"red"}

    ax.plot(params[x_var], mean, "o", color=color_map[params["criterion"]])



blue_patch = mpatches.Patch(color='blue', label='gini')

red_patch = mpatches.Patch(color='red', label='entropy')

handles = [blue_patch, red_patch]

plot_grid_search(dt_grid_results, plotting_func_dt, "Tennis Decision Tree Training Accuracy as a Function of Max_depth", "max_depth", "accuracy",(6,4), handles)



eval_for_conclusion("Tennis Decision Tree", dt_grid_results, dt_grid_test_x, dt_grid_test_y)
print(confusion_matrix(dt_grid_test_y, dt_grid_results.predict(dt_grid_test_x)))

print("DT model accuracy:", metrics.accuracy_score(dt_grid_test_y, dt_grid_results.predict(dt_grid_test_x))) 

print("DT model precision:", metrics.precision_score(dt_grid_test_y, dt_grid_results.predict(dt_grid_test_x))) 

print("DT model recall:", metrics.recall_score(dt_grid_test_y, dt_grid_results.predict(dt_grid_test_x))) 





knn_classifier = KNeighborsClassifier()

evaluate_model(big_df, knn_classifier, "Tennis knn baseline")

knn_param_grid = {"n_neighbors":[x for x in range (2,21)] + [y*10 for y in range(3,11)]}

(knn_grid_train_x, knn_grid_train_y, knn_grid_test_x, knn_grid_test_y) = split_test_train(0.8, big_df)

knn_grid_results = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=5).fit(knn_grid_train_x, knn_grid_train_y)



# All results

plt.xlabel("Number of Neighbors (k)")

plt.ylabel("Accuracy")

plt.title("Tennis K nearest neighbor Grid Search, Accuracy as a Function of K")

knn_means = knn_grid_results.cv_results_['mean_test_score']

knn_stds = knn_grid_results.cv_results_['std_test_score']

knn_params = knn_grid_results.cv_results_['params']

k_vals = []

knn_accuracies = []

for mean, std, params in zip(knn_means, knn_stds, knn_params):

    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    knn_accuracies.append(mean)

    k_vals.append(params["n_neighbors"])

print('Best parameters found:\n', knn_grid_results.best_params_, "with score of: ", max(knn_grid_results.cv_results_['mean_test_score']))



    

eval_for_conclusion("Tennis KNN", knn_grid_results, knn_grid_test_x, knn_grid_test_y)



    

plt.plot(k_vals, knn_accuracies, 'o', color="red")

plt.show()    
svm_classifier = svm.SVC()

evaluate_model(big_df, svm_classifier, "Tennis svm_classifier")
svm_parameter_space = {

    'kernel': ['linear', 'rbf'],

    'C': [ float(i) / 100 for i in range (1, 300, 5)]+[5,7,9],    

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

print("SVM grid search model test set score: ", svm_grid_score)

print('Best SVM parameters found through cv:\n', svm_grid_clf.best_params_, "with score of: ", max(svm_grid_clf.cv_results_['mean_test_score']))

def plotting_func_svm(mean, params, plt, ax):

    x_var = "C"

    tick_spacing = 0.1

    if mean < 0.7: return #don't need to show too many outliers to save space

    #layer_colors = {"(3,)":"orange", "(5,)":"red", "(7,)":"black"}

    kernel_labels = {"linear": "o", "rbf":"o"}

    kernel_colors = {"linear": "red", "rbf":"black"}

    #print(params["hidden_layer_sizes"])

    #layer_color_idx = str(params["hidden_layer_sizes"])

    kernel_idx = params["kernel"]

    ax.plot(params[x_var], mean, kernel_labels[kernel_idx], color=kernel_colors[kernel_idx]) #, color=layer_colors[layer_color_idx])

#     x_loc = plticker.MultipleLocator(base=0.5) # this locator puts ticks at regular intervals

#     ax.xaxis.set_major_locator(x_loc)



    y_loc = plticker.MultipleLocator(base=0.005) # this locator puts ticks at regular intervals

    ax.yaxis.set_major_locator(y_loc)

    #plt.ylim(.76, .80)



linear = mlines.Line2D([], [],marker='o',

                         label='linear', color="r")

rbf = mlines.Line2D([], [],marker='o',

                         label='rbf', color="black")

handles = [linear, rbf]#, red_patch, orange_patch, black_patch]



plot_grid_search(svm_grid_clf, plotting_func_svm, "Tennis SVM accuracy as a function of C", "C", "accuracy",(5,8), handles)

eval_for_conclusion("Tennis SVM", svm_grid_clf, svm_test_x, svm_test_y)



boost_classifier = AdaBoostClassifier()

evaluate_model(big_df, boost_classifier, "Tennis Boosting_classifier")
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

    ax.plot(params[x_var], mean,"o", color=cmap(norm(-1*params["n_estimators"])))

    #plt.ylim(.70, .80)

    #plt.xlim(0, 1.8)



cmap = cm.hot

norm = Normalize(vmin=-110, vmax=-20)

yellow_patch = mpatches.Patch(color=cmap(norm(-50)), label='n_estimators=50')

red_patch = mpatches.Patch(color=cmap(norm(-80)), label='n_estimators=80')

black_patch = mpatches.Patch(color=cmap(norm(-110)), label='n_estimators=110')

handles = [yellow_patch, red_patch, black_patch] #[linear, rbf]#, red_patch, orange_patch, black_patch]





plot_grid_search(boost_grid_clf, plotting_func_boost, "Tennis Boosting accuracy as a function of learning rate", "learning rate", "accuracy",(5,8), handles)



eval_for_conclusion("Tennis Boosting", boost_grid_clf, boost_test_x, boost_test_y)

import matplotlib.cm as cm

from matplotlib.colors import Normalize



cmap = cm.autumn

norm = Normalize(vmin=-20, vmax=10)

print(cmap(norm(5)))



eval_df = pd.DataFrame(columns=["model", "recall", "accuracy", "precision"])

eval_df = eval_df.append(eval_for_conclusion("Tennis DT", dt_grid_results, dt_grid_test_x, dt_grid_test_y), ignore_index=True)

eval_df = eval_df.append(eval_for_conclusion("Tennis Boosting", boost_grid_clf, boost_test_x, boost_test_y), ignore_index=True)

eval_df = eval_df.append(eval_for_conclusion("Tennis SVM", svm_grid_clf, svm_test_x, svm_test_y), ignore_index=True)

eval_df = eval_df.append(eval_for_conclusion("Tennis KNN", knn_grid_results, knn_grid_test_x, knn_grid_test_y), ignore_index=True)

eval_df = eval_df.append(eval_for_conclusion("Tennis Neural Network", nn_clf, nn_test_x, nn_test_y), ignore_index=True)



eval_df.info()

eval_df.head()
