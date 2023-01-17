import random



import matplotlib

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import plotly.express as px

import plotly.figure_factory as ff

import plotly.graph_objects as goT

import seaborn as sns

import xgboost as xgb

from catboost import CatBoostClassifier, CatBoostRegressor

from sklearn import metrics

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,

                              GradientBoostingClassifier,

                              RandomForestClassifier, VotingClassifier)

from sklearn.linear_model import (ElasticNet, Lasso, LinearRegression,

                                  LogisticRegression, Ridge)

from sklearn.metrics import (classification_report, log_loss,

                             mean_squared_error, mean_squared_log_error)

from sklearn.model_selection import (GridSearchCV, StratifiedKFold,

                                     cross_val_score, learning_curve,

                                     train_test_split)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
# Config 

TRAIN_PATH = "../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv"
train_df = pd.read_csv(TRAIN_PATH)
train_df.head()
train_df.describe()
train_df.info()
# Let us look at the different values of quality

train_df.quality.unique()
def correlation_matrix(dataframe, col_list_to_drop):

    """This plots a correlation matrix in a dataframe

    @param dataframe: The dataframe

    @param col_list_to_drop: The columns which we should skip.

    NOTE : We should only have numerical columns in the correlations plot.

    """

    f, ax = plt.subplots(figsize=[20, 15])

    sns.heatmap(dataframe.drop(col_list_to_drop, axis=1).corr(),

                annot=True,

                fmt=".2f",

                ax=ax,

                cbar_kws={'label': 'Correlation Coefficient'},

                cmap='viridis')

    ax.set_title("Correlation Matrix", fontsize=18)

    plt.show()
correlation_matrix(train_df, [])
# Modify quality attribute



def compute_quality_label(x):

    if x <= 5:

        return 'good'

    else:

        return 'better'



train_df['num quality'] = train_df['quality']

train_df['quality'] = train_df['quality'].apply(lambda x : compute_quality_label(x))
train_df.head()
def continuous_var_distribution_vs_single_cat_var(dataframe, numerical_col_1,

                                                  cat_col_name, cat_col_list,

                                                  plot_title):

    """This method plots multiple box plots for a single

    continuous variable distribution vs multiple categorical

    variable distribution.

    @param dataframe: The dataframe

    @param numerical_col_1: The numerical col name

    @param cat_col_name: The categorical col name

    @param cat_col_list: The different values of the categorical variable

    that it takes

    @param plot_title: The plot title name

    """

    hex_colors_names = []

    for name, hex in matplotlib.colors.cnames.items():

        hex_colors_names.append(name)



    dataframe_list = []

    for col_name in cat_col_list:

        dataframe_list.append(dataframe[dataframe[cat_col_name] == col_name][numerical_col_1])



    fig = go.Figure()

    for i in range(len(dataframe_list)):

        df = dataframe_list[i]

        fig.add_trace(go.Box(y=df,

                             jitter=0.3,

                             pointpos=-1.8,

                             boxpoints='all',  # Display all points in plot

                             marker_color=hex_colors_names[i+30],

                             name=cat_col_list[i]))

    fig.update_layout(title=plot_title)

    fig.show()
def continous_var_vs_single_cat_vars_waves(dataframe, numerical_col_1,

                                           cat_col_name, cat_col_list,

                                           binsize, show_hist=False):

    """This method plots distplot and also waves without

    the histogram in the background when there is a single

    continuous variable vs a single categorical vairable

    @param dataframe: The dataframe

    @param numerical_col_1: The numerical col name

    @param cat_col_name: The categorical col name

    @param cat_col_list: The different values the categorical

    variable takes

    @param binsize: The bin size for the single categorical variable

    @param show_hist: Whether to show the histogram or not

    """

    hex_colors_names = []

    for name, hex in matplotlib.colors.cnames.items():

        hex_colors_names.append(name)

    random.shuffle(hex_colors_names)

    hex_colors_names = hex_colors_names[:len(cat_col_list)]



    dataframe_list = []

    for col_name in cat_col_list:

        dataframe_list.append(dataframe[dataframe[cat_col_name] == col_name][numerical_col_1])



    binsize_list = []

    for i in range(len(cat_col_list)):

        binsize_list.append(binsize)



    fig = ff.create_distplot(dataframe_list,

                             cat_col_list,

                             show_hist=show_hist,

                             colors=hex_colors_names,

                             bin_size=binsize_list)

    fig.show()
continous_var_vs_single_cat_vars_waves(train_df, 

                                      'fixed acidity',

                                      'quality',

                                      ['good', 'better'],

                                      [10, 10])
## A new dataframe for storing the newly formed columns

temp_df = pd.DataFrame()

temp_df['quality'] = train_df['quality']
# Modify fixed acidity attribute



def compute_fixed_acidity_label(x):

    if x <= 6:

        return 0

    elif x <= 10:

        return 1

    else:

        return 2



temp_df['num fixed acidity'] = train_df['fixed acidity'].apply(lambda x : compute_fixed_acidity_label(x))
# Plot after modification and assigning the different groups

continous_var_vs_single_cat_vars_waves(temp_df, 

                                      'num fixed acidity',

                                      'quality',

                                      ['good', 'better'],

                                      [10, 10])
continous_var_vs_single_cat_vars_waves(train_df, 

                                      'volatile acidity',

                                      'quality',

                                      ['good', 'better'],

                                      [10, 10])
def compute_volatile_acidity_label(x):

    if x <= 0.53:

        return 0

    else:

        return 1



temp_df['num volatile acidity'] = train_df['volatile acidity'].apply(lambda x : compute_volatile_acidity_label(x))
continous_var_vs_single_cat_vars_waves(temp_df, 

                                      'num volatile acidity',

                                      'quality',

                                      ['good', 'better'],

                                      [10, 10])
def numerical_vs_numerical_or_categorical(dataframe, numerical_col_1,

                                          numerical_col_2, title_of_plot,

                                          x_axis_title, y_axis_title,

                                          categorical_col=None, numerical_col_3=None):

    """This method plots the a scatter plot between a numerical value,

    and a categorical value. It also supports when there are 1 or 2 numerical

    values along with a categorical value.

    @param dataframe: The dataframe

    @param numerical_col_1: The first numerical value

    @param numerical_col_2: The second numerical value

    @param title_of_plot: Title of the plot

    @param x_axis_title: X axis title

    @param y_axis_title: Y axis title

    @param categorical_col: Categorical column name (optional)

    @param numerical_col_3: The third numerical value (optional)

    """

    fig = px.scatter(dataframe, x=numerical_col_1, y=numerical_col_2, color=categorical_col, size=numerical_col_3)

    fig.update_layout(title=title_of_plot, xaxis_title=x_axis_title, yaxis_title=y_axis_title)

    fig.show()
## Let us first find out the bound sulphur in the wine. 

## We do this be subtracting the free sulphur dioxide from the total sulhpur dioxide

## We will drop this column later



train_df['bound sulfur dioxide'] = train_df['total sulfur dioxide'] - train_df['free sulfur dioxide']
numerical_vs_numerical_or_categorical(train_df, 

                                     'free sulfur dioxide', 'bound sulfur dioxide', 'Relationship between free and bound sulfur dioxide',

                                     'free sulfur dioxide', 'bound sulfur dioxide')
numerical_vs_numerical_or_categorical(train_df, 

                                     'free sulfur dioxide', 'sulphates', 'Relationship between free sulfur dioxide and sulphates',

                                     'free sulfur dioxide', 'sulphates')
numerical_vs_numerical_or_categorical(train_df, 

                                     'bound sulfur dioxide', 'sulphates', 'Relationship between bound sulphur dioxide and sulphates',

                                     'bound sulfur dioxide', 'sulphates')
train_df = train_df.drop(['bound sulfur dioxide'], axis=1)
train_df.columns
for col_name in train_df.columns:

    if col_name == 'quality' or col_name == 'fixed acidity' or col_name == 'volatile acidity' or col_name == 'num quality':

        continue

    continous_var_vs_single_cat_vars_waves(train_df, 

                                           col_name,

                                           'quality',

                                           ['good', 'better'],

                                           [5, 5])
def compute_citric_acid(x):

    if x <= 0.3:

        return 0

    else:

        return 1



def compute_chlorides(x):

    if x <= 0.1:

        return 0

    else:

        return 1

    

def compute_free_sulfur_dioxide(x):

    if x <= 20:

        return 0

    else:

        return 1

    

def compute_total_sulfur_dioxide(x):

    if x <= 57:

        return 0

    else:

        return 1

    

def compute_density(x):

    if x <= 0.9957:

        return 0

    elif x <= 0.9988:

        return 1

    else:

        return 2

    

def compute_sulphates(x):

    if x <= 0.62:

        return 0

    elif x <= 1.07:

        return 1

    else:

        return 2

    

def compute_alcohol(x):

    if x <= 10.23:

        return 0

    else:

        return 1





temp_df['num citric acid'] = train_df['citric acid'].apply(lambda x : compute_citric_acid(x))

temp_df['num chlorides'] = train_df['chlorides'].apply(lambda x : compute_chlorides(x))

temp_df['num free sulfur dioxide'] = train_df['free sulfur dioxide'].apply(lambda x : compute_free_sulfur_dioxide(x))

temp_df['num total sulfur dioxide'] = train_df['total sulfur dioxide'].apply(lambda x : compute_total_sulfur_dioxide(x))

temp_df['num density'] = train_df['density'].apply(lambda x : compute_density(x))

temp_df['num sulphates'] = train_df['sulphates'].apply(lambda x : compute_sulphates(x))

temp_df['num alcohol'] = train_df['alcohol'].apply(lambda x : compute_alcohol(x))

train_df = train_df.drop(['residual sugar', 'pH'], axis=1)
temp_df.columns
for col_name in temp_df.columns:

    if col_name == 'quality':

        continue

    continous_var_vs_single_cat_vars_waves(temp_df, 

                                           col_name,

                                           'quality',

                                           ['good', 'better'],

                                           [5, 5])
train_df = train_df.drop(['quality'], axis=1)

temp_df = temp_df.drop(['quality'], axis=1)
num_quality = train_df['num quality']

train_df = train_df.drop(['num quality'], axis=1)



scaled_features = StandardScaler().fit_transform(train_df.values)

train_df = pd.DataFrame(scaled_features, index=train_df.index, columns=train_df.columns)



train_df['num quality'] = num_quality.values
temp_df['quality'] = train_df['num quality']

X = temp_df.drop(['quality'], axis=1)

y = temp_df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
random_state = 2

classifiers = [SVC(random_state=random_state),

               DecisionTreeClassifier(random_state=random_state),

               AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state), random_state=random_state,

                                  learning_rate=0.1),

               RandomForestClassifier(random_state=random_state),

               ExtraTreesClassifier(random_state=random_state),

               GradientBoostingClassifier(random_state=random_state),

               MLPClassifier(random_state=random_state),

               KNeighborsClassifier(),

               LogisticRegression(random_state=random_state),

               LinearDiscriminantAnalysis()]
def ensembling_cross_val_classification_first_step(X_train, Y_train):

    kfold = StratifiedKFold(n_splits=10)



    cv_results = []

    for classifier in classifiers:

        cv_results.append(cross_val_score(classifier,

                                          X_train, y=Y_train,

                                          scoring="accuracy",

                                          cv=kfold,

                                          n_jobs=4))



    cv_means = []

    cv_std = []

    for cv_result in cv_results:

        cv_means.append(cv_result.mean())

        cv_std.append(cv_result.std())



    cv_res = pd.DataFrame(

        {

            "CrossValMeans": cv_means,

            "CrossValerrors": cv_std,

            "Algorithm": ["SVC", "DecisionTree", "AdaBoost",

                          "RandomForest", "ExtraTrees", "GradientBoosting",

                          "MultipleLayerPerceptron", "KNeighboors", "LogisticRegression",

                          "LinearDiscriminantAnalysis"]

        }

    )



    g = sns.barplot("CrossValMeans",

                    "Algorithm",

                    data=cv_res,

                    palette="Set3",

                    orient="h",

                    **{'xerr': cv_std})

    g.set_xlabel("Mean Accuracy")

    g.set_title("Cross validation scores")
ensembling_cross_val_classification_first_step(X_train, y_train)
classifiers_for_ensembling = [

    ExtraTreesClassifier(),

    RandomForestClassifier()

    #GradientBoostingClassifier(),

    #SVC(probability=True)

]



parameters_for_ensembling_models = [

    {

        # For ExtraTreeClassifier

        "max_depth": [None],

        "max_features": [1, 3, 10],

        "min_samples_split": [2, 3, 10],

        "min_samples_leaf": [1, 3, 10],

        "bootstrap": [False],

        "n_estimators": [100, 300],

        "criterion": ["gini"]

    },

    {

        # Random forest classifier

        "max_depth": [None],

        "max_features": [1, 3, 10],

        "min_samples_split": [2, 3, 10],

        "min_samples_leaf": [1, 3, 10],

        "bootstrap": [False],

        "n_estimators": [100, 300],

        "criterion": ["gini"]

    }

#     {

#         # Gradient boosting

#         'loss': ["deviance"],

#         'n_estimators': [100, 200, 300],

#         'learning_rate': [0.1, 0.05, 0.01],

#         'max_depth': [4, 8],

#         'min_samples_leaf': [100, 150],

#         'max_features': [0.3, 0.1]

#     },

#     {

#         # SVM Classifier

#         'kernel': ['rbf'],

#         'gamma': [0.001, 0.01, 0.1, 1],

#         'C': [1, 10, 50, 100, 200, 300, 1000]

#     }

]
def plot_learning_curve(estimator, title,

                        X, y, ylim=None,

                        cv=None, n_jobs=-1,

                        train_sizes=np.linspace(.1, 1.0, 5)):

    """Generate a simple plot of the test and training learning curve"""

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")



    train_sizes, train_scores, test_scores = learning_curve(estimator,

                                                            X,

                                                            y,

                                                            cv=cv,

                                                            n_jobs=n_jobs,

                                                            train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)



    plt.grid()

    plt.fill_between(train_sizes,

                     train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std,

                     alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes,

                     test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std,

                     alpha=0.1,

                     color="g")

    plt.plot(train_sizes,

             train_scores_mean,

             'o-',

             color="r",

             label="Training score")

    plt.plot(train_sizes,

             test_scores_mean,

             'o-',

             color="g",

             label="Cross-validation score")

    plt.legend(loc="best")

    return plt





def grid_search_find_best_models(X_train, Y_train,

                                 kfold):

    best_models = []



    for i in range(len(classifiers_for_ensembling)):

        model = classifiers_for_ensembling[i]

        params = parameters_for_ensembling_models[i]

        grid_search_model = GridSearchCV(model,

                                         param_grid=params,

                                         cv=kfold,

                                         scoring="accuracy",

                                         n_jobs=4,

                                         verbose=1)

        print(model)

        grid_search_model.fit(X_train, Y_train)

        best_models.append(grid_search_model.best_estimator_)

        plot_learning_curve(grid_search_model.best_estimator_,

                            "Learning curve for best model",

                            X_train,

                            Y_train,

                            cv=kfold)



    return best_models
best_models = grid_search_find_best_models(X_train, y_train, StratifiedKFold(n_splits=10))
def plot_feature_importance_of_tree_based_models(names_classifiers, X_train):

    # names_classifiers = [("AdaBoosting", ada_best), ("ExtraTrees", ExtC_best), ("RandomForest", RFC_best),

    #                     ("GradientBoosting", GBC_best)]



    nrows = ncols = 4

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex="all", figsize=(15, 15))



    nclassifier = 0

    for row in range(nrows):

        for col in range(ncols):

            name = names_classifiers[nclassifier][0]

            classifier = names_classifiers[nclassifier][1]

            indices = np.argsort(classifier.feature_importances_)[::-1][:40]

            g = sns.barplot(y=X_train.columns[indices][:40],

                            x=classifier.feature_importances_[indices][:40],

                            orient='h',

                            ax=axes[row][col])

            g.set_xlabel("Relative importance", fontsize=12)

            g.set_ylabel("Features", fontsize=12)

            g.tick_params(labelsize=9)

            g.set_title(name + " feature importance")

            nclassifier += 1
best_models
names_classifier = [('ExtraTreeClassifier', best_models[0]),

                   ('RandomForestClassifier', best_models[1])]

plot_feature_importance_of_tree_based_models(names_classifier, X_train)
def plot_ensemble_classifier_results(test, classifiers,

                                     X_train, Y_train):

    # [('rfc', RFC_best), ('extc', ExtC_best),

    #  ('svc', SVMC_best), ('adac', ada_best), ('gbc', GBC_best)]



    class_res = []

    for classifier in classifiers:

        class_res.append(pd.Series(classifier[1].predict(test), name=classifier[0]))

    ensemble_results = pd.concat(class_res, axis=1)

    sns.heatmap(ensemble_results.corr(), annot=True)

    votingC = VotingClassifier(estimators=classifiers,

                               voting='soft',

                               n_jobs=4)

    return votingC.fit(X_train, Y_train)
voting_classifier = plot_ensemble_classifier_results(X_test, names_classifier, X_train, y_train)
print(classification_report(y_test.values, voting_classifier.predict(X_test)))
train_df = train_df.drop(['chlorides', 'free sulfur dioxide', 'fixed acidity'], axis=1)

temp_df = temp_df.drop(['num chlorides', 'num free sulfur dioxide', 'num fixed acidity'], axis=1)
temp_df = temp_df.drop(['quality'], axis=1)
dataset = pd.concat([temp_df, train_df], axis=1)
X = dataset.iloc[:, :-1]

y = dataset.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
cat_features = [0, 1, 2, 3, 4, 5]
clf = CatBoostClassifier(iterations=100) #verbose=5)



clf.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test))
print(classification_report(y_test.values, clf.predict(X_test)))
def cat_boost_regressor(X_train, y_train,

                        X_val, y_val,

                        categorical_features_indices):

    """This trains the cat boost regressor on a training data set

    @param X_train: Training dataset input

    @param y_train: Training dataset output

    @param X_val: Validation dataset input

    @param y_val: Validation dataset output

    @param categorical_features_indices: List of indices

    which are categorical features in the X_train

    @return: The trained model

    """

    model = CatBoostRegressor(iterations=60,

                              depth=3,

                              learning_rate=0.1,

                              loss_function='RMSE')

    model.fit(X_train, y_train,

              cat_features=categorical_features_indices,

              eval_set=(X_val, y_val),

              plot=True)



    return model
cb_model = cat_boost_regressor(X_train, y_train, X_test, y_test, cat_features)
mean_squared_error(y_test.values, cb_model.predict(X_test))
def ridge_regression(X_train, y_train):

    """This trains the ridge regressor

    @param X_train: Training dataset input

    @param y_train: Training dataset output

    @return: Trained model, the best parameters after grid search and the

    best score of the model

    """

    ridge = Ridge()

    parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}

    ridge_regressor = GridSearchCV(ridge,

                                   parameters,

                                   scoring='neg_mean_squared_error',

                                   cv=5)

    ridge_regressor.fit(X_train, y_train)



    return ridge_regressor.best_estimator_ , ridge_regressor.best_params_, ridge_regressor.best_score_
ridge_model, _, _ = ridge_regression(X_train, y_train)
mean_squared_error(y_test.values, ridge_model.predict(X_test))