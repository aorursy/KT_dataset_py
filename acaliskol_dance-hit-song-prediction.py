def warn(*args, **kwargs):

    pass

import warnings

warnings.warn = warn



import math

import numpy as np

import pandas as pd

from sklearn import linear_model

from sklearn.preprocessing import normalize, RobustScaler, StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.metrics import accuracy_score, precision_recall_curve, confusion_matrix

from sklearn.feature_selection import f_regression

from sklearn.feature_selection import SelectKBest

from sklearn.naive_bayes import GaussianNB

from sklearn import tree

from sklearn.model_selection import train_test_split, RepeatedKFold

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, PredefinedSplit

from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

from sklearn.model_selection import StratifiedShuffleSplit

# import optunity

# import optunity.metrics

from sklearn.linear_model import ElasticNetCV

from scipy.stats import uniform as sp_rand

from patsy.user_util import balanced

from scipy import stats

from sklearn.preprocessing import MinMaxScaler

from scipy.stats import randint

from scipy.stats import wilcoxon

#from hypopt import GridSearch

from xgboost import XGBClassifier



# matplotlib and seaborn for plotting

import matplotlib.pyplot as plt

import seaborn as sns

import itertools



pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
# %% Data yukle

filename = "../input/hityeni3key.csv"

data = pd.read_csv(filename, sep=";")
# %% Data separate

training = data[data['year'] < 2012]

test = data[data['year'] >= 2012]



training = training[training['song_popularity'] != 50]

test = test[test['song_popularity'] != 50]
# %% K-Fold

def k_fold_cv(clf, X, y, param_grid):

    rkf = RepeatedKFold(n_splits=10, n_repeats=50, random_state=42)



    data = {}

    data['f1'] = []

    data['accuracy'] = []

    data['precision'] = []

    data['recall'] = []



    for train_index, test_index in rkf.split(X):

        X_train = X[train_index]

        X_test = X[test_index]

        y_train = y[train_index]

        y_test = y[test_index]



        grid_clf = grid_search(X_train, y_train, clf, param_grid)

        y_pred = grid_clf.predict(X_test)



        data['accuracy'].append(accuracy_score(y_test, y_pred))

        data['f1'].append(f1_score(y_test, y_pred))

        data['precision'].append(precision_score(y_test, y_pred))

        data['recall'].append(recall_score(y_test, y_pred))



    return [data['accuracy'], data['f1'], data['precision'], data['recall']]



# %% Split Training and Test Sets

def start_test(training, test, drop):

    y_train = np.where(training['song_popularity'] <= 50, 0, 1)

    y_test = np.where(test['song_popularity'] <= 50, 0, 1)



    print(np.unique(y_train, return_counts=True))

    print(np.unique(y_test, return_counts=True))



    # %% Training Data

    training.drop(["year", "title", "artist", "peak_position", "song_popularity", "spotify_id", "youtube_id"], axis=1,

                  inplace=True)

    training.drop(drop, axis=1, inplace=True)

    # training.drop(["artist_popularity","duration_ms","tempo","time_signature","mode","song_key","acousticness","danceability","energy","instrumentalness","liveness","loudness","speechiness","valence"], axis=1, inplace=True)

    training.set_index('id', inplace=True)

    X_train = training.values



    # %% Test Data

    popularity = test["song_popularity"].values

    test.drop(["year", "title", "artist", "peak_position", "song_popularity", "spotify_id", "youtube_id"], axis=1,

              inplace=True)

    test.drop(drop, axis=1, inplace=True)

    # test.drop(["artist_popularity","duration_ms","tempo","time_signature","mode","song_key","acousticness","danceability","energy","instrumentalness","liveness","loudness","speechiness","valence"], axis=1, inplace=True)

    test.set_index('id', inplace=True)

    X_test = test.values



    # %% Featurelari sec ve ihtiyac olmayanlari at

    # data.drop(["year","title","artist","peak_position","song_popularity","spotify_id","youtube_id"], axis=1, inplace=True)

    # #data.drop(["circular_chord_prog"], axis=1, inplace=True)

    # data.set_index('id', inplace=True)



    # %% Normalization



    # data = (data - data.min()) / (data.max() - data.min())

    X = np.concatenate([X_train, X_test], axis=0)

    robust_scaler = RobustScaler()

    X = robust_scaler.fit_transform(X)



    X_train = X[0:y_train.shape[0], :]

    X_test = X[y_train.shape[0]:, :]



    # %% Define Classifiers



    names = [

        "SVC (RBF)",

        #"SVC (Linear)",

        #"SVC (Poly)",

        #"Logistic Reg.",

        # "Naive Bayes",

        #"XGBoost",

        # "Decision Tree",

        #"Random Forest"

    ]



    classifiers = [

        SVC(class_weight="balanced", kernel="RBF", random_state=42, probability=True),

        #LinearSVC(class_weight="balanced", max_iter=500000, penalty='l2', random_state=42),

        # SVC(class_weight="balanced", kernel="poly", gamma="scale", random_state=42),

        #linear_model.LogisticRegression(class_weight='balanced', penalty='l2', dual=False, tol=0.1, fit_intercept=True,

        #                                intercept_scaling=1, solver='liblinear', multi_class='ovr', warm_start=False, random_state=42),

        # GaussianNB(),

        #XGBClassifier(class_weight="balanced", learning_rate=0.1, random_state=42, n_jobs=-1),

        # tree.DecisionTreeClassifier(class_weight="balanced", random_state=42),

        #RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)

    ]



    # %% Param Tuning

    param_grid = [

        {"kernel": ['rbf'], 'C': [1000], 'gamma': [1e-4]},

        #{'C': [1], 'tol': [1e-13]},  # Linear

        # {"kernel": ['poly'], 'C': [10], 'degree': [3]},  # Poly

        # {'C': [0.1], 'tol': [1]},  # Logistic

        # {},  # Naive

        #{'n_estimators':[1100],'max_depth':[5],'min_child_weight':[5],'subsample': [1],'colsample_bytree': [1],"colsample_bylevel":[1],'gamma':[0],'missing':[0.7],'base_score':[0.7]},

        #{"max_depth": [20],"max_features": [1,3,9],"min_samples_leaf": [1],"min_samples_split": [10],"criterion": ["gini"],"max_leaf_nodes":[2]},

        #{'n_estimators': [200],'max_depth': [1],"max_features": [8],"min_samples_split": [1.0],'min_samples_leaf': [1],'bootstrap': [False],"criterion": ["entropy"]}

    ]

    

    # K-Flod CV

    [accuracy, f1, precision, recall] = k_fold_cv(classifiers[0], X_train, y_train, param_grid[0])

    return np.column_stack((f1, accuracy))



    # %% Validation and Test

    i = 0

    for clf_name, clf in zip(names, classifiers):

        print(clf_name + " Results: ")



        # K-Fold CV

        [accuracy, f1, precision, recall] = k_fold_cv(clf, X_train, y_train, param_grid[i])

        

        print("\tAccuracy: %1.3f" % np.mean(accuracy))

        print("\tF1: %1.3f" % np.mean(f1))

        print("\tPrecision: %1.3f" % np.mean(precision))

        print("\tRecall: %1.3f" % np.mean(recall))



        i = i + 1

        continue



        # Test

        grid_clf = grid_search(X_train, y_train, clf, param_grid[i])

        # best_params = randomized_search(X_train_val, y_train_val, clf, param_grid[i], param_tuning_fold, 10)



        y_pred = grid_clf.predict(X_test)

        y_prob = grid_clf.predict_proba(X_test)[:, 1]



        print("\tAccuracy: %1.3f" % accuracy_score(y_test, y_pred))

        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))

        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))

        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))



        # Concat info

        arr_info = np.concatenate(

            [y_pred.reshape(-1, 1), y_test.reshape(-1, 1), y_prob.reshape(-1, 1), popularity.reshape(-1, 1)], axis=1)



        # Order By Confidence

        ordered_confidence_arr = order_by_x(arr_info, 2)



        for k in [5, 10, 15, 20]:

            print("Confidence Top %d" % k)

            get_top_k_accuracy(ordered_confidence_arr, k)



        # Order by Popularity

        ordered_popularity_arr = order_by_x(arr_info, 3)



        for k in [5, 10, 15, 20]:

            print("Spotify Popularity Top %d" % k)

            get_top_k_accuracy(ordered_popularity_arr, k)



        # Concordance Index

        c_index = concordance_index(ordered_confidence_arr, 5)

        print("Concordance Index: %f" % c_index)



        i = i + 1





# %% Functions



# Validation with GridSearchCV

def grid_search(X_train_val, y_train_val, clf, param_grid):

    grid_search_clf = GridSearchCV(clf, param_grid=param_grid, n_jobs=-1, scoring='f1', cv=2)

    grid_search_clf.fit(X_train_val, y_train_val)

    # print(grid_search_clf.best_estimator_.get_params())

    # print(grid_search_clf.best_score_)



    return grid_search_clf





# Validation with RandomizedSearchCV

def randomized_search(X_train_val, y_train_val, clf, param_grid, param_tuning_fold, n_iter):

    random_search_clf = RandomizedSearchCV(estimator=clf, param_distributions=param_grid, n_iter=n_iter, scoring='f1',

                                           cv=PredefinedSplit(test_fold=param_tuning_fold), verbose=2,

                                           random_state=42, n_jobs=-1)

    random_search_clf.fit(X_train_val, y_train_val)

    print(random_search_clf.best_estimator_.get_params())

    print(random_search_clf.best_score_)



    return random_search_clf.best_estimator_.get_params()



def performance_measure(y_actual, y_hat):

    TP = 0

    FP = 0

    TN = 0

    FN = 0



    for i in range(len(y_hat)):

        if y_actual[i] == y_hat[i] == 1:

            TP += 1

        if y_hat[i] == 1 and y_actual[i] == 0:

            FP += 1

        if y_hat[i] == y_actual[i] == 0:

            TN += 1

        if y_hat[i] == 0 and y_actual[i] == 1:

            FN += 1



    return (TP, FP, TN, FN)





def order_by_x(arr, x):

    a = arr[:, x].argsort()

    sorted_rows = arr[a][::-1]

    return sorted_rows





def get_top_k_accuracy(arr, k):

    arr = arr[:k, :]

    y_pred = arr[:, 0]

    y_test = arr[:, 1]



    tp, fp, tn, fn = performance_measure(y_test, y_pred)

    tp_acc = float(tp) / len(arr)



    print("\tTP Accuracy: %1.3f\n" % tp_acc)





def concordance_index(arr, k=None):

    if (k != None):

        arr = arr[:k, :]



    sum = 0

    popularity_sum = 0



    i = 1

    for row in arr:

        sum += (row[3] / i)

        popularity_sum += row[3]

        i = i + 1



    return 1.0 / (sum / popularity_sum)



drop = [

    ["common_chord_prog",

     "circular_chord_prog"],

    "dissonance",

    "consonance",

    "resolution",

    "melody_leap",

    "mean_pitch"]



# print("Hepsi:")

# start_test(training.copy(), test.copy(), [])



# print("Featuresiz sonuclar:")

# start_test(training.copy(), test.copy(), np.hstack(drop))



# print("en iyi kombinasyon 1:")

# start_test(training.copy(), test.copy(), ["resolution","common_chord_prog","circular_chord_prog","dissonance"])



# print("en iyi kombinasyon 2:")

# start_test(training.copy(), test.copy(), ["resolution","common_chord_prog","circular_chord_prog","consonance"])



# for curr in itertools.combinations(drop, 1):

#     drop_list = list(set(np.hstack(drop)) - set(np.hstack(curr)))

#     print(str(curr) + " icin sonuclar:")

#     #print(drop_list)

#     start_test(training.copy(), test.copy(), drop_list)



# for curr in itertools.combinations(drop, 2):

#     drop_list = list(set(np.hstack(drop)) - set(np.hstack(curr)))

#     print(str(curr) + " icin sonuclar:")

#     #print(drop_list)

#     start_test(training.copy(), test.copy(), drop_list)



# for curr in itertools.combinations(drop, 3):

#     drop_list = list(set(np.hstack(drop)) - set(np.hstack(curr)))

#     print(str(curr) + " icin sonuclar:")

#     #print(drop_list)

#     start_test(training.copy(), test.copy(), drop_list)

    

# for curr in itertools.combinations(drop, 4):

#     drop_list = list(set(np.hstack(drop)) - set(np.hstack(curr)))

#     print(str(curr) + " icin sonuclar:")

#     #print(drop_list)

#     start_test(training.copy(), test.copy(), drop_list)

    

# for curr in itertools.combinations(drop, 5):

#     drop_list = list(set(np.hstack(drop)) - set(np.hstack(curr)))

#     print(str(curr) + " icin sonuclar:")

#     #print(drop_list)

#     start_test(training.copy(), test.copy(), drop_list)



# T-Test

# 1. Test only basic features

results_basic = start_test(training.copy(), test.copy(), np.hstack(drop))



# 2. Test: Dissonance, Melody Leap, Mean Pitch

results_hypothesis = start_test(training.copy(), test.copy(), ["resolution","consonance","common_chord_prog","circular_chord_prog"])



results_total = np.column_stack((results_basic, results_hypothesis))



results_data = pd.DataFrame(results_total, columns=['basic_model_f1','basic_model_accuracy','best_model_f1', 'best_model_accuracy'])

print(results_data.head())

print(results_data.describe())



# Check outliers

results_data[['basic_model_f1','best_model_f1']].plot(kind='box')

# This saves the plot as a png file

plt.savefig('boxplot_outliers.png')

plt.clf()



# Check Normal Distribution

results_data['diff_f1'] = results_data['best_model_f1'] - results_data['basic_model_f1']



results_data['diff_f1'].plot(kind='hist', title= 'F1 Difference Histogram')



# Again, this saves the plot as a png file

plt.savefig('F1 difference histogram.png')

plt.clf()



# Check Normal Distribution with Q-Q Plot

stats.probplot(results_data['diff_f1'], plot= plt)

plt.title('F1 Difference Q-Q Plot')

plt.savefig('F1 difference qq plot.png')

plt.clf()



# Shapiro-Wilk Test to check normal distribution statistically

print(stats.shapiro(results_data['diff_f1']))



# Paired t-Test

print(stats.ttest_rel(results_data['basic_model_f1'], results_data['best_model_f1']))



# Wilcoxon

print(wilcoxon(results_data['diff_f1'], alternative='greater'))