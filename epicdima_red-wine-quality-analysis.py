import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
%matplotlib inline

import warnings
warnings.simplefilter("ignore")
# read from a regular csv file
data = pd.read_csv("../input/winequality-red.csv")
'''
This is translation for me :)
Input variables (based on physicochemical tests):
1 - fixed acidity        - фиксированная кислотность
2 - volatile acidity     - летучая кислотность
3 - citric acid          - лимонная кислота
4 - residual sugar       - остаточный сахар
5 - chlorides            - хлориды
6 - free sulfur dioxide  - свободный диоксид серы
7 - total sulfur dioxide - общий диоксид серы
8 - density              - плотность
9 - pH                   - водородный показатель(кислотность среды pH)
10 - sulphates           - сульфаты
11 - alcohol             - алкоголь
Output variable:
12 - quality (score between 0 and 10) - качество (0 - 10)
'''
data.head(10)
data.tail()
data.info()
data.describe()
print("Number of unique values in each column:\n")
for i in data.columns:
    print(i, len(data[i].unique()))
data['bin_quality'] = pd.cut(data['quality'], bins=[0, 6.5, 10], labels=["bad", "good"])
data.head(10)
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))

data_length = len(data)
quality_percentage = [100 * i / data_length for i in data["quality"].value_counts()]
bin_quality_percentage = [100 * i / data_length for i in data["bin_quality"].value_counts()]

sns.countplot("quality", data=data, ax=ax[0, 0])
sns.countplot("bin_quality", data=data, ax=ax[0, 1]);

sns.barplot(x=data["quality"].unique(), y=quality_percentage, ax=ax[1, 0])
ax[1, 0].set_xlabel("quality")

sns.barplot(x=data["bin_quality"].unique(), y=bin_quality_percentage, ax=ax[1, 1])
ax[1, 1].set_xlabel("bin_quality")

for i in range(2):
    ax[1, i].set_ylabel("The percentage of the total number")
    ax[1, i].set_yticks(range(0, 101, 10))
    ax[1, i].set_yticklabels([str(i) + "%" for i in range(0, 101, 10)])
    for j in range(2):
        ax[i, j].yaxis.grid()
        ax[i, j].set_axisbelow(True)
plt.figure(figsize=[9, 9])
sns.heatmap(data.corr(), xticklabels=data.columns[:-1], yticklabels=data.columns[:-1], 

            square=True, cmap="Spectral_r", center=0);
#  The function takes on the input column name and restrictions on the y axis. 
#  Next, the function builds a histogram of the distribution of the values 
#  of this column, a histogram of the dependence of the two types 
# of quality to the column passed as a parameter.

def drawing_two_barplots(column, ylims):
    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(2, 2)
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    
    sns.distplot(data[data.columns[column]], kde=False, ax=ax0)
    sns.barplot("quality", data.columns[column], data=data, ax=ax1)
    sns.barplot("bin_quality", data.columns[column], data=data, ax=ax2)
    ax1.set_ylim(ylims[0], ylims[1])
    ax2.set_ylim(ylims[0], ylims[1])
    ax1.set_yticks(np.linspace(ylims[0], ylims[1], 11))
    ax2.set_yticks(np.linspace(ylims[0], ylims[1], 11))
    ax1.yaxis.grid()
    ax2.yaxis.grid()
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)
drawing_two_barplots(0, [0, 10])
drawing_two_barplots(1, [0, 1.2])
drawing_two_barplots(2, [0, 0.5])
drawing_two_barplots(3, [0, 3.6])
drawing_two_barplots(4, [0, 0.18])
drawing_two_barplots(5, [0, 20])
drawing_two_barplots(6, [0, 60])
drawing_two_barplots(7, [0.994, 0.999])
drawing_two_barplots(8, [3.1, 3.5])
drawing_two_barplots(9, [0, 0.9])
drawing_two_barplots(10, [0, 13])
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, normalize

from sklearn.naive_bayes           import GaussianNB
from sklearn.linear_model          import LogisticRegression
from sklearn.neighbors             import KNeighborsClassifier
from sklearn.svm                   import SVC
from sklearn.tree                  import DecisionTreeClassifier
from sklearn.neural_network        import MLPClassifier
from sklearn.ensemble              import ExtraTreesClassifier
from sklearn.ensemble              import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from lightgbm import LGBMClassifier

FEATURES = slice(0,-2, 1)
model_names = ['LogisticRegression',
               'KNeighborsClassifier',
               'SVC',
               'MLPClassifier',
               'ExtraTreesClassifier',
               'RandomForestClassifier',
               'LinearDiscriminantAnalysis',
               'LGBMClassifier']

classifiers = [LogisticRegression, # Логистическая регрессия
               KNeighborsClassifier, # K-ближайших соседей
               SVC, # Метод опорных векторов
               MLPClassifier, # Трёхслойный перцептрон
               ExtraTreesClassifier, # Экстра (randomized) деревья 
               RandomForestClassifier, # Случайный лес
               LinearDiscriminantAnalysis, # Линейный дискриминантный анализ
               LGBMClassifier] # Градиентный бустинг
#  This function takes an instance of the model, data and labels as input, 
#  and there is an optional parameter that indicates the number of splits (rounds) to validate. 
#  The function returns the average value of cross-validation, as well as the standard deviation.

def cross_val_mean_std(clsf, data, labels, cv=5):
    cross_val = cross_val_score(clsf, data, labels, cv=cv)
    cross_val_mean = cross_val.mean() * 100
    cross_val_std = cross_val.std() * 100
    return round(cross_val_mean, 3), round(cross_val_std, 3)
#  This function takes the type of training model, training and test data, 
#  and parameters for that model, if any, as input. 
#  The function returns the already trained model, which we can use 
#  if necessary, as well as a dictionary with the 
#  results of cross-validation of training and test data.

def train_and_validate_model(model, train, train_labels, test, test_labels, parameters=None):
    
    if parameters is not None:
        model = model(**parameters)
    else:
        model = model()
        
    model.fit(train, train_labels)
    train_valid = cross_val_mean_std(model, train, train_labels)
    test_valid = cross_val_mean_std(model, test, test_labels)
        
    res_of_valid = {"train_mean": train_valid[0], "train_std": train_valid[1],
                    "test_mean":  test_valid[0],  "test_std":  test_valid[1]}
    
    return res_of_valid, model
#  This function takes a dictionary derived from the work of a past function 
#  that contains a cross-validation result for one or more models and 
#  creates a Pandas table (which returns), optionally adding postfix to the column names.

def create_table_with_scores(res_of_valid, postfix=""):
    if not hasattr(res_of_valid["test_std"], "len"):
        index = [0]
    else:
        index = list(res_of_valid["test_std"])

    table = pd.DataFrame({"Test mean score" + postfix:  res_of_valid["test_mean"],
                          "Test std score" + postfix:   res_of_valid["test_std"],
                          "Train mean score" + postfix: res_of_valid["train_mean"],
                          "Train std score" + postfix:  res_of_valid["train_std"]}, 
                          index=index)
    return table
#  This function takes a list of Pandas tables that are created by the function above, 
#  then it takes a list of names in text format (the length of the lists must match), 
#  there is an optional argument - the number of the column to sort, if necessary.
#  Returns one large table that consists of a list of tables that the function has accepted, 
#  as well as a new column with model names from the second argument. 
#  If the third parameter was specified, the function returns the table with sorting.

def table_of_results(model_results, model_names=None, col_sort_by=None):
    res = model_results[0]
    for i in model_results[1:]:
        res = res.append(i)
    if model_names is not None:
        names = []
        for i, j in enumerate(model_names):
            names += [j] * len(model_results[i])
        res["Model name"] = names
    if col_sort_by is not None:
        sort_by = res.columns[col_sort_by]
        res = res.sort_values(by=sort_by, ascending=False)
    res = res.reset_index(drop=True)
    return res
#  This function takes in a large table from the previous function as well 
#  as column numbers to draw a scatter chart. 
#  This function is used to draw cross-validation results from training and test data
#  and compare different models or the same models with different parameters.

def graph_for_the_results_table(table, col_x, col_y, col_style):
    x = table.columns[col_x]
    y = table.columns[col_y]
    style = table.columns[col_style]
    plt.figure(figsize=[8, 8])
    min_lim = min(min(table[x]), min(table[y]))
    max_lim = max(max(table[x]), max(table[y]))
    ax = sns.scatterplot(x, y, style, style=style, data=table, s=100)
    ax.set_xlim(min_lim - 0.01 * max_lim, max_lim + 0.01 * max_lim)
    ax.set_ylim(min_lim - 0.01 * max_lim, max_lim + 0.01 * max_lim)
    ax.grid()
    ax.set_axisbelow(True)
train, test, train_labels, test_labels = train_test_split(data[data.columns[FEATURES]], 
                                                          data[data.columns[-2:]], 
                                                          test_size=0.25, random_state=3)

b_train_labels = np.array(train_labels)[:, 1]
b_test_labels = np.array(test_labels)[:, 1]

train_labels = np.array(train_labels)[:, 0].astype(int)
test_labels = np.array(test_labels)[:, 0].astype(int)

sc = StandardScaler()
train = sc.fit_transform(train)
test = sc.fit_transform(test)
classifiers_scores = []
b_classifiers_scores = []

classifiers_importance = []

for i, clsf in enumerate(classifiers):
    t = [0, 0]
    
    res_of_valid, t[0] = train_and_validate_model(clsf, train, train_labels, test, test_labels)
    b_res_of_valid, t[1] = train_and_validate_model(clsf, train, b_train_labels, test, b_test_labels)
    
    classifiers_importance.append(t)
    
    classifiers_scores.append(create_table_with_scores(res_of_valid, " ('quality')"))
    b_classifiers_scores.append(create_table_with_scores(b_res_of_valid, " ('bin_quality')"))
    
classifiers_scores = table_of_results(classifiers_scores, model_names, 0)
b_classifiers_scores = table_of_results(b_classifiers_scores, model_names, 0)
classifiers_scores
graph_for_the_results_table(classifiers_scores, 0, 2, 4)
graph_for_the_results_table(classifiers_scores, 1, 3, 4)
b_classifiers_scores
graph_for_the_results_table(b_classifiers_scores, 0, 2, 4)
graph_for_the_results_table(b_classifiers_scores, 1, 3, 4)
importances = []
b_importances = []

for clsf, b_clsf in classifiers_importance:
    if hasattr(clsf, "feature_importances_"):
        importances.append(clsf.feature_importances_)
        b_importances.append(b_clsf.feature_importances_)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 6))
fig.suptitle("The average importance of features")
sns.barplot(list(range(1, 12)), np.mean(importances, axis=0), ax=ax1)
sns.barplot(list(range(1, 12)), np.mean(b_importances, axis=0), ax=ax2)
ax1.set_title("quality")
ax2.set_title("bin_quality")
ax1.get_yaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
ax1.set_xticklabels(data.columns, rotation=90)
ax2.set_xticklabels(data.columns, rotation=90);
#  This function takes model and parameters to find optimal, training and test data, 
#  postfix for column names, number of iterations and partitions to cross-validate 
#  for RandomizedSearchCV.
#  Returns only the table with the results.

def tuning_models(model, params, train, train_labels, 
                                 test, test_labels, postfix="", iterations=50, cv=5):
    
    model_1 = model()
    random_search = RandomizedSearchCV(model_1, params, iterations, scoring='accuracy', cv=cv)
    random_search.fit(train, train_labels)
    
    parameter_set = []
    mean_test_scores = list(random_search.cv_results_['mean_test_score'])
    for i in sorted(mean_test_scores, reverse=True):
        if i > np.mean(mean_test_scores):
            parameter_set.append(random_search.cv_results_["params"][mean_test_scores.index(i)])
        
    params_set_updated = []
    for i in parameter_set:
        if i not in params_set_updated:
            params_set_updated.append(i)
    
    results = []
    for i in params_set_updated:
        res_of_valid, _ = train_and_validate_model(model, train, train_labels, test, test_labels, parameters=i)
        results.append(create_table_with_scores(res_of_valid, postfix))
    
    results_table = table_of_results(results)
    return results_table
params = {"kernel": ["rbf", "poly", "linear", "sigmoid"],
          "C": np.arange(0.1, 1.5, 0.1), 
          "gamma": list(np.arange(0.1, 1.5, 0.1)) + ["auto"],
          "probability": [True, False],
          "shrinking": [True, False]}
svc_res = tuning_models(SVC, params, train, train_labels, 
                        test, test_labels, " ('quality')", 100)

b_svc_res = tuning_models(SVC, params, train, b_train_labels, 
                          test, b_test_labels, " ('bin_quality')", 100)
params = {"n_estimators": np.arange(1, 500, 2),
          "max_depth": list(np.arange(2, 100, 2)) + [None],
          "min_samples_leaf": np.arange(1, 20, 1),
          "min_samples_split": np.arange(2, 20, 2),
          "max_features": ["auto", "log2", None]}
extra_res = tuning_models(ExtraTreesClassifier, params, train, train_labels, 
                          test, test_labels, " ('quality')", 100)

b_extra_res = tuning_models(ExtraTreesClassifier, params, train, b_train_labels, 
                            test, b_test_labels, " ('bin_quality')", 100)

forest_res = tuning_models(RandomForestClassifier, params, train, train_labels, 
                           test, test_labels, " ('quality')", 100)

b_forest_res = tuning_models(RandomForestClassifier, params, train, b_train_labels, 
                             test, b_test_labels, " ('bin_quality')", 100)
params = {"boosting_type": ["gbdt"],
          "num_leaves": np.arange(2, 100, 2),
          "max_depth": list(np.arange(2, 100, 2)) + [-1],
          "learning_rate": [0.001, 0.003, 0.006, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.17, 0.2, 0.3, 0.4],
          "n_estimators": np.arange(2, 300, 5),
          "reg_alpha": np.arange(0, 1, 0.1),
          "reg_lambda": np.arange(0, 1, 0.1)}
lgb_res = tuning_models(LGBMClassifier, params, train, train_labels, 
                        test, test_labels, " ('quality')", 100)

b_lgb_res = tuning_models(LGBMClassifier, params, train, b_train_labels, 
                          test, b_test_labels, " ('bin_quality')", 100);
all_results = table_of_results([svc_res, extra_res, forest_res, lgb_res], 
                               ["SVC", "ExtraTrees", "RandomForest", "LightGBM"], 0)
all_results.head(10)
graph_for_the_results_table(all_results, 0, 2, 4)
graph_for_the_results_table(all_results, 1, 3, 4)
b_all_results = table_of_results([b_svc_res, b_extra_res, b_forest_res, b_lgb_res], 
                                 ["SVC", "ExtraTrees", "RandomForest", "LightGBM"], 0)
b_all_results.head(10)
graph_for_the_results_table(b_all_results, 0, 2, 4)
graph_for_the_results_table(b_all_results, 1, 3, 4)