import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification

from sklearn.feature_selection import RFE, RFECV

from sklearn.model_selection import train_test_split, GridSearchCV, KFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.metrics import roc_curve



# Classifiers

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier

from sklearn.linear_model import RidgeClassifier, SGDClassifier

from sklearn.naive_bayes import BernoulliNB, GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import LinearSVC, NuSVC, SVC

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

import catboost as cb

import lightgbm as lgb

import xgboost as xgb



from sklearn.model_selection import cross_val_predict

from sklearn.metrics import accuracy_score 

from sklearn.metrics import f1_score

from imblearn.metrics import geometric_mean_score

from imblearn.metrics import sensitivity_score

from imblearn.metrics import specificity_score

from sklearn.metrics import roc_auc_score

###############################################################################

#                               4. Classifiers                                #

###############################################################################

# Create list of tuples with classifier label and classifier object

classifiers = {}

classifiers.update({"QDA": QuadraticDiscriminantAnalysis()})

classifiers.update({"AdaBoost": AdaBoostClassifier()})

classifiers.update({"Bagging": BaggingClassifier()})

classifiers.update({"Gradient Boosting": GradientBoostingClassifier()})

classifiers.update({"Random Forest": RandomForestClassifier()})

classifiers.update({"BNB": BernoulliNB()})

classifiers.update({"GNB": GaussianNB()})

classifiers.update({"KNN": KNeighborsClassifier()})

classifiers.update({"MLP": MLPClassifier()})

classifiers.update({"NuSVC": NuSVC()})

classifiers.update({"DTC": DecisionTreeClassifier()})

classifiers.update({"ETC": ExtraTreeClassifier()})

classifiers.update({"Catboost": cb.CatBoostClassifier()})

classifiers.update({"LightGBM": lgb.LGBMClassifier(boosting_type= 'gbdt', objective = 'binary')})

classifiers.update({"XGB": xgb.XGBClassifier(objective="binary:logistic", random_state=42)})

# Initiate parameter grid

parameters = {}



# Update dict with QDA

parameters.update({"QDA": {"classifier__reg_param":[0.34], 

                                         }})

# Update dict with AdaBoost

parameters.update({"AdaBoost": { 

                                "classifier__base_estimator": [DecisionTreeClassifier()],

                                "classifier__n_estimators": [200],

                                "classifier__learning_rate": [0.5]

                                 }})



# Update dict with Bagging

parameters.update({"Bagging": { 

                                "classifier__base_estimator": [DecisionTreeClassifier()],

                                "classifier__n_estimators": [200],

                                "classifier__n_jobs": [-1]

                                }})



# Update dict with Gradient Boosting

parameters.update({"Gradient Boosting": { 

                                        "classifier__learning_rate":[0.05], 

                                        "classifier__n_estimators": [200],

                                        "classifier__max_depth": [5],

                                        "classifier__min_samples_split": [0.05],

                                        "classifier__min_samples_leaf": [0.05],

                                        "classifier__max_features": ["log2"],

                                        "classifier__subsample": [0.8]

                                         }})





# Update dict with Random Forest Parameters

parameters.update({"Random Forest": { 

                                    "classifier__n_estimators": [200],

                                    "classifier__class_weight": ["balanced"],

                                    "classifier__max_features": ["auto"],

                                    "classifier__max_depth" : [6],

                                    "classifier__min_samples_split": [0.005],

                                    "classifier__min_samples_leaf": [0.005],

                                    "classifier__criterion" :["entropy"]     ,

                                    "classifier__n_jobs": [-1]

                                     }})





# Update dict with BernoulliNB Classifier

parameters.update({"BNB": { 

                            "classifier__alpha": [1e-09]

                             }})



# Update dict with GaussianNB Classifier

parameters.update({"GNB": { 

                            "classifier__var_smoothing": [1e-09]

                             }})



# Update dict with K Nearest Neighbors Classifier

parameters.update({"KNN": { 

                            "classifier__n_neighbors": [29],

                            "classifier__p": [1],

                            "classifier__leaf_size": [5],

                            "classifier__n_jobs": [-1]

                             }})





# Update dict with MLPClassifier

parameters.update({"MLP": { 

                            "classifier__hidden_layer_sizes": [ (10,10,10)],

                            "classifier__activation": ["relu"],

                            "classifier__learning_rate": ["invscaling"],

                            "classifier__max_iter": [200],

                            "classifier__alpha": [1e-06],

                             }})







parameters.update({"NuSVC": { 

                            "classifier__nu": [0.50],

                            "classifier__kernel": ["rbf"],

                            "classifier__degree": [1],

                             }})



# Update dict with Decision Tree Classifier

parameters.update({"DTC": { 

                            "classifier__criterion" :[ "entropy"],

                            "classifier__splitter": ["best"],

                            "classifier__class_weight": [None],

                            "classifier__max_features": ["auto"],

                            "classifier__max_depth" : [7],

                            "classifier__min_samples_split": [0.005],

                            "classifier__min_samples_leaf": [0.01],

                             }})





# Update dict with Extra Tree Classifier

parameters.update({"ETC": { 

                            "classifier__criterion" :["gini"],

                            "classifier__splitter": ["best"],

                            "classifier__class_weight": [None],

                            "classifier__max_features": ["sqrt"],

                            "classifier__max_depth" : [ 7],

                            "classifier__min_samples_split": [0.01],

                            "classifier__min_samples_leaf": [0.005],

                             }})

parameters.update({"LightGBM": {

    "classifier__learning_rate": [0.005],

    "classifier__n_estimators": [40],

    "classifier__num_leaves": [6,16],

    "classifier__boosting_type" : ['gbdt'],

    "classifier__objective" : ['binary'],

    "classifier__random_state" : [501],

    "classifier__colsample_bytree" : [0.65, 0.66],

    "classifier__subsample": [0.7,0.75],

    "classifier__reg_alpha" : [1.2],

    "classifier__reg_lambda" : [1]

    }})

parameters.update({"Catboost": {'classifier__depth': [10],

 'classifier__iterations': [300],

 'classifier__l2_leaf_reg': [9],

 'classifier__learning_rate': [0.03]}})



parameters.update({"XGB": {'classifier__colsample_bytree': [0.8],

 'classifier__gamma': [0.1],

 'classifier__max_depth': [5],

 'classifier__min_child_weight': [1],

 'classifier__reg_alpha': [0.01],

 'classifier__subsample': [0.8]}})
df = pd.read_csv("../input/promisee/promisin_couples.csv")

#df.replace(to_replace = -1 , value =np.nan)



X = df.iloc[:, 1:138].values

y = df.iloc[:, 139].values



#imputing missing values

from sklearn.impute import KNNImputer

imputer = KNNImputer()

#imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)

imputer.fit(X[:, 1:138])

X[:, 1:138] = imputer.transform(X[:, 1:138])



#Making all the values discrete

from sklearn.preprocessing import KBinsDiscretizer

est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

X = est.fit_transform(X)



# Filter Method: Spearman's Cross Correlation > 0.95

# Make correlation matrix

corr_matrix = pd.DataFrame(X).corr(method = "spearman").abs()



# Draw the heatmap

sns.set(font_scale = 1.0)

f, ax = plt.subplots(figsize=(11, 9))

sns.heatmap(corr_matrix, cmap= "YlGnBu", square=True, ax = ax)

f.tight_layout()

plt.savefig("correlation_matrix.png", dpi = 1080)



# Select upper triangle of matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]



# Drop features

X =pd.DataFrame(X).drop(to_drop, axis = 1)



###############################################################################

#                  8. Custom pipeline object to use with RFECV                #

###############################################################################

# Select Features using RFECV

class PipelineRFE(Pipeline):

    # Source: https://ramhiser.com/post/2018-03-25-feature-selection-with-scikit-learn-pipeline/

    def fit(self, X, y=None, **fit_params):

        super(PipelineRFE, self).fit(X, y, **fit_params)

        self.feature_importances_ = self.steps[-1][-1].feature_importances_

        return self



scaler = StandardScaler()

estimator = RandomForestClassifier(n_estimators= 200,

                                   class_weight ='balanced',

                                   max_features = 'auto',

                                   max_depth = 6,

                                   min_samples_split = 0.005,

                                   min_samples_leaf = 0.005,

                                   criterion = 'entropy',

                                   n_jobs = -1)

steps = [("scaler", scaler), ("classifier", estimator)]

pipe = PipelineRFE(steps = steps)



# Initialize RFECV object

feature_selector = RFECV(pipe, cv = 10, step = 1, min_features_to_select=10, scoring = "roc_auc", verbose = 1)



# Fit RFECV

X = feature_selector.fit_transform(X, y)



# Get selected features

feature_names = X.columns

selected_features = feature_names[feature_selector.support_].tolist()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,

                                                    random_state = 1000)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,

                                                    random_state = 1000)
print(X_train)
# Get Performance Data

performance_curve = {"Number of Features": list(range(1, len(feature_names) + 1)),

                    "AUC": feature_selector.grid_scores_}

performance_curve = pd.DataFrame(performance_curve)



# Performance vs Number of Features

# Set graph style

sns.set(font_scale = 1.75)

sns.set_style({"axes.facecolor": "1.0", "axes.edgecolor": "0.85", "grid.color": "0.85",

               "grid.linestyle": "-", 'axes.labelcolor': '0.4', "xtick.color": "0.4",

               'ytick.color': '0.4'})

colors = sns.color_palette("RdYlGn", 20)

line_color = colors[3]

marker_colors = colors[-1]



# Plot

f, ax = plt.subplots(figsize=(13, 6.5))

sns.lineplot(x = "Number of Features", y = "AUC", data = performance_curve,

             color = line_color, lw = 4, ax = ax)

sns.regplot(x = performance_curve["Number of Features"], y = performance_curve["AUC"],

            color = marker_colors, fit_reg = False, scatter_kws = {"s": 200}, ax = ax)



# Axes limits

plt.xlim(0.5, len(feature_names)+0.5)

plt.ylim(0.60, 0.925)



# Generate a bolded horizontal line at y = 0

ax.axhline(y = 0.625, color = 'black', linewidth = 1.3, alpha = .7)



# Turn frame off

ax.set_frame_on(False)



# Tight layout

plt.tight_layout()



# Save Figure

plt.savefig("performance_curve.png", dpi = 1080)



###############################################################################

#                  12. Visualizing Selected Features Importance               #

###############################################################################

# Get selected features data set

X_train = X_train[selected_features]

X_test = X_test[selected_features]



# Train classifier

classifier.fit(X_train, np.ravel(y_train))



# Get feature importance

feature_importance = pd.DataFrame(selected_features, columns = ["Feature Label"])

feature_importance["Feature Importance"] = classifier.feature_importances_



# Sort by feature importance

feature_importance = feature_importance.sort_values(by="Feature Importance", ascending=False)



# Set graph style

sns.set(font_scale = 1.75)

sns.set_style({"axes.facecolor": "1.0", "axes.edgecolor": "0.85", "grid.color": "0.85",

               "grid.linestyle": "-", 'axes.labelcolor': '0.4', "xtick.color": "0.4",

               'ytick.color': '0.4'})



# Set figure size and create barplot

f, ax = plt.subplots(figsize=(12, 9))

sns.barplot(x = "Feature Importance", y = "Feature Label",

            palette = reversed(sns.color_palette('YlOrRd', 15)),  data = feature_importance)



# Generate a bolded horizontal line at y = 0

ax.axvline(x = 0, color = 'black', linewidth = 4, alpha = .7)



# Turn frame off

ax.set_frame_on(False)



# Tight layout

plt.tight_layout()



# Save Figure

plt.savefig("feature_importance.png", dpi = 1080)




###############################################################################

#                       13. Classifier Tuning and Evaluation                  #

###############################################################################

# Initialize dictionary to store results

result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

results = pd.DataFrame(columns = ["Classifiers", "Best Parameters"

              "Training AUC",

              "Test AUC",

              "Accuracy",

              "F-Measure",

              "Geometric Mean",

              "Sensitivity",

              "Specificity"])



# Tune and evaluate classifiers

for classifier_label, classifier in classifiers.items():

    # Print message to user

    print(f"Now tuning {classifier_label}.")

    

    # Scale features via Z-score normalization

    scaler = StandardScaler()

    

    # Define steps in pipeline

    steps = [("scaler", scaler), ("classifier", classifier)]

    

    # Initialize Pipeline object

    pipeline = Pipeline(steps = steps)

      

    # Define parameter grid

    param_grid = parameters[classifier_label]

    

    # Initialize GridSearch object

    gscv = GridSearchCV(pipeline, param_grid, cv = 10,  n_jobs= -1, verbose = 1, scoring = "roc_auc")

                      

    # Fit gscv

    gscv.fit(X_train, np.ravel(y_train))  

    clf = gscv.best_estimator_

    y_pred = clf.predict(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_pred)

    auc = roc_auc_score(y_test, y_pred)

    

    result_table = result_table.append({'classifiers':classifier_label,

                                        'fpr':fpr, 

                                        'tpr':tpr, 

                                        'auc':auc}, ignore_index=True)

    

    # Evaluate model

    auc = metrics.roc_auc_score(y_test, y_pred)

    

    # Save results

    results = results.append({"Classifier": classifier_label,

              "Best Parameters": gscv.best_params_,

              "Training AUC": gscv.best_score_,

              "Test AUC": auc,

              "Accuracy": accuracy_score(y_test, y_pred),

              "F-Measure": f1_score(y_test, y_pred, average = 'weighted'),

              "Geometric Mean": geometric_mean_score(y_test, y_pred, average = 'weighted'),

              "Sensitivity" :   sensitivity_score(y_test, y_pred, average = 'weighted'),

              "Specificity" :   specificity_score(y_test, y_pred, average = 'weighted')}, ignore_index = True)



import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



import warnings

warnings.filterwarnings('ignore')







# Set name of the classifiers as index labels

result_table.set_index('classifiers', inplace=True)



fig = plt.figure(figsize=(20,18))



for i in result_table.index:

    plt.plot(result_table.loc[i]['fpr'], 

             result_table.loc[i]['tpr'], 

             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))

    

plt.plot([0,1], [0,1], color='orange', linestyle='--')



plt.xticks(np.arange(0.0, 1.1, step=0.1))

plt.xlabel("Flase Positive Rate", fontsize=15)



plt.yticks(np.arange(0.0, 1.1, step=0.1))

plt.ylabel("True Positive Rate", fontsize=15)



plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)

plt.legend(prop={'size':13}, loc='lower right')



plt.show()

fig.savefig('multiple_roc_curve.png')
###############################################################################

#                              14. Visualing Results                          #

###############################################################################

# Initialize auc_score dictionary

auc_scores = {

              "Classifier": [],

              "AUC": [],

              "AUC Type": []

              }



# Get AUC scores into dictionary

for classifier_label in results:

    auc_scores.update({"Classifier": [classifier_label] + auc_scores["Classifier"],

                       "AUC": [results[classifier_label]["Training AUC"]] + auc_scores["AUC"],

                       "AUC Type": ["Training"] + auc_scores["AUC Type"]})

    

    auc_scores.update({"Classifier": [classifier_label] + auc_scores["Classifier"],

                       "AUC": [results[classifier_label]["Test AUC"]] + auc_scores["AUC"],

                       "AUC Type": ["Test"] + auc_scores["AUC Type"]})



# Dictionary to PandasDataFrame

auc_scores = pd.DataFrame(auc_scores)



# Set graph style

sns.set(font_scale = 1.75)

sns.set_style({"axes.facecolor": "1.0", "axes.edgecolor": "0.85", "grid.color": "0.85",

               "grid.linestyle": "-", 'axes.labelcolor': '0.4', "xtick.color": "0.4",

               'ytick.color': '0.4'})



    

# Colors

training_color = sns.color_palette("RdYlBu", 10)[1]

test_color = sns.color_palette("RdYlBu", 10)[-2]

colors = [training_color, test_color]



# Set figure size and create barplot

f, ax = plt.subplots(figsize=(12, 9))



sns.barplot(x="AUC", y="Classifier", hue="AUC Type", palette = colors,

            data=auc_scores)



# Generate a bolded horizontal line at y = 0

ax.axvline(x = 0, color = 'black', linewidth = 4, alpha = .7)



# Turn frame off

ax.set_frame_on(False)



# Tight layout

plt.tight_layout()



# Save Figure

plt.savefig("AUC Scores.png", dpi = 1080)
results.to_csv('Accuracy.csv')