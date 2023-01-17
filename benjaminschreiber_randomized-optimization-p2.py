!pip install mlrose
import mlrose

import numpy as np



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score

import time





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

print("Finished installing modules.")





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# EDIT HERE!!

input_source = '/kaggle/input'

data_frames = []

for dirname, _, filenames in os.walk(input_source):

    for filename in filenames:

        file_path = os.path.join(dirname, filename)

        data_frames.append(pd.read_csv(file_path))

big_frame = pd.concat(data_frames, ignore_index=True)

big_frame.info()



# Any results you write to the current directory are saved as output.
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
(nn_train_x, nn_train_y, nn_test_x, nn_test_y) = split_test_train(0.1 * 8, big_df)

scaler = StandardScaler()

scaler.fit(nn_train_x)

nn_train_x = scaler.transform(nn_train_x)

nn_test_x = scaler.transform(nn_test_x)



def randomized_train_nn(rando_algo, train_x, train_y, test_x, test_y, accuracies, times, max_iters=100, max_attempts=3, 

                        early_stopping=False, mutation_prob=0.1, hidden_nodes=[5,5], pop_size=200, to_print=False):

    nn_model = mlrose.NeuralNetwork(hidden_nodes = hidden_nodes, activation = 'tanh', \

                                     algorithm = rando_algo, max_iters = max_iters, \

                                     bias = True, is_classifier = True, learning_rate = 0.1,

                                     clip_max = 5, max_attempts = max_attempts, random_state = 3,curve=True, early_stopping=early_stopping, mutation_prob=mutation_prob,pop_size=pop_size)

    start = time.time()

    nn_model.fit(train_x, train_y)

    end = time.time()

    time_elapsed = end - start

    # Predict labels for train set and assess accuracy

    y_train_pred = nn_model.predict(train_x)

    #y_train_accuracy = accuracy_score(train_y, y_train_pred)

    #print("{0} y train accuracy: {1}".format(rando_algo,y_train_accuracy))

    # Predict labels for test set and assess accuracy

    y_test_pred = nn_model.predict(test_x)

    y_test_accuracy = accuracy_score(test_y, y_test_pred)

    

    accuracies.append(y_test_accuracy)

    times.append(time_elapsed)



    if to_print:

        print(classification_report(test_y, y_test_pred))

        print(confusion_matrix(test_y, y_test_pred))

        print("{0} y test accuracy: {1}".format(rando_algo,y_test_accuracy))

        print("{0} time elapsed: {1}".format(rando_algo,time_elapsed))

    return nn_model



def iterative_train(times, accuracies, iters_range, problem_text, early_stopping=False, mutation_prob=0.1):

    nn_model = None

    for i in iters_range:

        nn_model = randomized_train_nn(problem_text, nn_train_x, nn_train_y, nn_test_x, nn_test_y, accuracies, times, i, 

                                       early_stopping=early_stopping, mutation_prob=mutation_prob)



    plt.plot(iters_range,times,'o-')

    plt.title("Neural Net Training Time over {0}".format(problem_text))

    plt.xlabel("Number of iterations")

    plt.ylabel("Time")

    plt.show()



    plt.plot(iters_range,accuracies,'o-')

    plt.title("Neural Net Test Accuracy over {0}".format(problem_text))

    plt.xlabel("Number of iterations")

    plt.ylabel("Highest Score")

    plt.show()

    return nn_model
hc_times = []

hc_acc = []

iters_range = range(1, 2200, 200)

iterative_train(hc_times, hc_acc, iters_range, "random_hill_climb")

    

hc_times = []

hc_acc = []

#iters_range = range(1, 2200, 200)

#iterative_train(hc_times, hc_acc, iters_range, "random_hill_climb")

hc_nn = randomized_train_nn("random_hill_climb", nn_train_x, nn_train_y, nn_test_x, nn_test_y, ga_accuracies, ga_times,\

                            max_iters=2000, early_stopping=False, mutation_prob=.15, hidden_nodes=[2], pop_size=2000, max_attempts=10, to_print=True)

plt.plot(range(1,len(hc_nn.fitness_curve)+1), hc_nn.fitness_curve)

plt.title("Neural Net Fitness over genetic_alg")

plt.xlabel("Number of iterations")

plt.ylabel("Fitness")

plt.show()

sa_times = []

sa_accuracies = []

iters_range = range(1, 2200, 200)

iterative_train(sa_times, sa_accuracies, iters_range, "simulated_annealing")

sa_accuracies
later_iters_range = range(2000, 3500, 500)

iterative_train(sa_times, sa_accuracies, later_iters_range, "simulated_annealing")

x_axis = list(range(1, 2200, 200)) + [2500,3000]

# nn_model = randomized_train_nn("simulated_annealing", nn_train_x, nn_train_y, nn_test_x, nn_test_y,sa_times, sa_accuracies)

plt.plot(x_axis,sa_accuracies,"o-")

plt.title("Neural Net Test Accuracy over simulated_annealing")

plt.xlabel("Number of iterations")

plt.ylabel("Highest Score")

plt.show()
ga_times = []

ga_accuracies = []

iters_range = range(1,101)

#ga_nn = iterative_train(ga_times, ga_accuracies, iters_range, "genetic_alg",early_stopping=True, mutation_prob=.3)



ga_nn = randomized_train_nn("genetic_alg", nn_train_x, nn_train_y, nn_test_x, nn_test_y, ga_accuracies, ga_times,\

                            max_iters=100, early_stopping=True, mutation_prob=.1, hidden_nodes=[5,5], pop_size=2000, max_attempts=3, to_print=True)

plt.plot(range(1,len(ga_nn.fitness_curve)+1), ga_nn.fitness_curve)

plt.title("Neural Net Fitness over genetic_alg")

plt.xlabel("Number of iterations")

plt.ylabel("Fitness")

plt.show()

ga_nn
def run_and_time_problem(run_problem, problem_name, to_print=True):

    start = time.time()

    best_state, best_fitness = run_problem()

    end = time.time()

    time_diff = end - start

    if to_print:

        print('The {} best state found is: {}'.format(problem_name, best_state))

        print('The {} fitness at the best state is: {}'.format(problem_name, best_fitness))

        print('The {} time elapsed to compute is: {}'.format(problem_name, time_diff))

        print("")

    return(best_state, best_fitness, time_diff)



def train_and_time(algo, times, scores):

    start = time.clock()

    state, top_score = algo()

    end = time.clock()

    times.append(end - start)

    scores.append(top_score)
hill_times = []

hill_scores = []

anneal_times = []

anneal_scores = []

genetic_times = []

genetic_scores = []

mimic_times = []

mimic_scores = []



def run_tsp(num_points):

    dist_list = []



    for x in range(num_points):

        for y in range(num_points):

            random.seed(x+y)

            dist_list.append((x, y, random.uniform(0.0, 1.0)))



    fitness_dists = mlrose.TravellingSales(distances=dist_list)

    tsp_problem = mlrose.TSPOpt(

        length=num_points, fitness_fn=fitness_dists, maximize=True)



    print ("Traveling Salesman Problem with ", num_points, " points")



    train_and_time(lambda : mlrose.random_hill_climb(problem=tsp_problem, max_attempts=1),hill_times, hill_scores)

    train_and_time(lambda : mlrose.simulated_annealing(problem=tsp_problem, max_attempts=1), anneal_times, anneal_scores)

    train_and_time(lambda : mlrose.genetic_alg(pop_size = 30, problem=tsp_problem, max_attempts=1), genetic_times, genetic_scores)

    train_and_time(lambda : mlrose.mimic(pop_size = 30, problem=tsp_problem, max_attempts=1), mimic_times, mimic_scores)

    

point_range = range(10,50,10)

for p in point_range:

    run_tsp(p)

plt.rcParams["figure.figsize"] = (5,5)



plt.plot(point_range,hill_times,'o-',point_range,anneal_times,'o-',point_range,genetic_times,'o-',point_range,mimic_times,'o-')

plt.title("TSP Problem")

plt.xlabel("Num cities")

plt.ylabel("Time")

plt.legend(["hill climb", "simulated annealing","genetic algorithm","mimic"])

plt.show()



plt.plot(point_range,hill_scores,'o-',point_range,anneal_scores,'o-', point_range,genetic_scores,'o-',point_range,mimic_scores,'o-')

plt.title("TSP Problem")

plt.xlabel("Num cities")

plt.ylabel("Highest Score")

plt.legend(["hill climb", "simulated annealing","genetic algorithm","mimic"])

plt.show()
####### Continuous PEAKS



cp_hill_times = []

cp_hill_scores = []

cp_anneal_times = []

cp_anneal_scores = []

cp_genetic_times = []

cp_genetic_scores = []

cp_mimic_times = []

cp_mimic_scores = []



def run_continuous_peaks(num_points):

    continuous_peaks = mlrose.DiscreteOpt(length = num_points,fitness_fn = mlrose.ContinuousPeaks(t_pct=0.15))

    print ("Continuous Peaks Problem with ", num_points, " points")

    train_and_time(lambda : mlrose.random_hill_climb(problem=continuous_peaks, max_attempts=6),cp_hill_times, cp_hill_scores)

    train_and_time(lambda : mlrose.simulated_annealing(problem=continuous_peaks, max_attempts=6), cp_anneal_times, cp_anneal_scores)

    train_and_time(lambda : mlrose.genetic_alg(pop_size = 30, problem=continuous_peaks, max_attempts=6), cp_genetic_times, cp_genetic_scores)

    train_and_time(lambda : mlrose.mimic(pop_size = 30, problem=continuous_peaks, max_attempts=6), cp_mimic_times, cp_mimic_scores)

    

point_range = range(10,100,10)

for p in point_range:

    run_continuous_peaks(p)

plt.rcParams["figure.figsize"] = (5,5)

plt.plot(point_range,cp_hill_times,'o-',point_range,cp_anneal_times,'o-',point_range,cp_genetic_times,'o-')

plt.title("Continuous Peaks Problem")

plt.xlabel("Array Length")

plt.ylabel("Time")

plt.legend(["hill climb", "simulated annealing","genetic algorithm","mimic"])

plt.show()



plt.plot(point_range,cp_hill_scores,'o-',point_range,cp_anneal_scores,'o-', point_range,cp_genetic_scores,'o-',point_range,cp_mimic_scores,'o-')

plt.title("Continuous Peaks Problem")

plt.xlabel("Array Length")

plt.ylabel("Highest Score")

plt.legend(["hill climb", "simulated annealing","genetic algorithm","mimic"])

plt.show()
cp_anneal_times
####### KNAPSACK

hill_times = []

hill_scores = []

anneal_times = []

anneal_scores = []

genetic_times = []

genetic_scores = []

mimic_times = []

mimic_scores = []



def run_knapsack(num_points):

    weights = list(np.random.randint(low = 1, high = 100, size = num_points))

    values = list(np.random.randint(low = 1, high = 100, size = num_points))

    knapsack = mlrose.DiscreteOpt(length = num_points,fitness_fn = mlrose.Knapsack(weights,values,0.2))

    print ("Knapsack Problem with ", num_points, " points")

    train_and_time(lambda : mlrose.random_hill_climb(problem=knapsack, max_attempts=10),hill_times, hill_scores)

    train_and_time(lambda : mlrose.simulated_annealing(problem=knapsack, max_attempts=10), anneal_times, anneal_scores)

    train_and_time(lambda : mlrose.genetic_alg(problem=knapsack, max_attempts=1), genetic_times, genetic_scores)

    train_and_time(lambda : mlrose.mimic(problem=knapsack, max_attempts=10, keep_pct=.20, fast_mimic=True), mimic_times, mimic_scores)

    

point_range = range(10,100,20)

for p in point_range:

    run_knapsack(p)

plt.rcParams["figure.figsize"] = (5,5)



plt.plot(point_range,hill_times,'o-',point_range,anneal_times,'o-',point_range,genetic_times,'o-',point_range,mimic_times,'o-')

plt.title("Knapsack Problem Times")

plt.xlabel("Array Length")

plt.ylabel("Time")

plt.legend(["hill climb", "simulated annealing","genetic algorithm","mimic"])

plt.show()



plt.plot(point_range,hill_scores,'o-',point_range,anneal_scores,'o-', point_range,genetic_scores,'o-',point_range,mimic_scores,'o-')

plt.title("Knapsack Problem Scores")

plt.xlabel("Array Length")

plt.ylabel("Highest Score")

plt.legend(["hill climb", "simulated annealing","genetic algorithm","mimic"])

plt.show()


c_hill_times = []

c_hill_scores = []

c_anneal_times = []

c_anneal_scores = []

c_genetic_times = []

c_genetic_scores = []

c_mimic_times = []

c_mimic_scores = []



def run_flip_flop(num_points):

    flip_flop = mlrose.DiscreteOpt(length = num_points, fitness_fn = mlrose.FlipFlop(), max_val=2)

    print ("Flip flop Problem with ", num_points, " points")

    train_and_time(lambda : mlrose.random_hill_climb(problem=flip_flop),c_hill_times, c_hill_scores)

    train_and_time(lambda : mlrose.simulated_annealing(problem=flip_flop), c_anneal_times, c_anneal_scores)

    train_and_time(lambda : mlrose.genetic_alg(problem=flip_flop), c_genetic_times, c_genetic_scores)

    train_and_time(lambda : mlrose.mimic(problem=flip_flop, max_iters=100), c_mimic_times, c_mimic_scores)

    

point_range = range(10,100,10)

for p in point_range:

    run_flip_flop(p)

plt.rcParams["figure.figsize"] = (5,5)

plt.plot(point_range,c_hill_times,'o-',point_range,c_anneal_times,'o-',point_range,c_genetic_times,'o-', point_range,c_mimic_times,'o-')

plt.title("Flip flop Problem")

plt.xlabel("Array Length")

plt.ylabel("Time")

plt.legend(["hill climb", "simulated annealing","genetic algorithm","mimic"])

plt.show()



plt.plot(point_range,c_hill_scores,'o-',point_range,c_anneal_scores,'o-', point_range,c_genetic_scores,'o-', point_range,c_mimic_times,'o-')

plt.title("Flip Flop Problem")

plt.xlabel("Array Length")

plt.ylabel("Highest Score")

plt.legend(["hill climb", "simulated annealing","genetic algorithm","mimic"])

plt.show()
