import time

import pandas as pd

import warnings

from datetime import datetime

warnings.filterwarnings("ignore")

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np

from sklearn.model_selection import train_test_split

from scipy.stats import pearsonr

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.legend_handler import HandlerLine2D

%matplotlib inline



from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier

from sklearn.feature_selection import SelectPercentile, RFE, RFECV

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.pipeline import Pipeline

from sklearn import metrics
df = pd.read_csv('/kaggle/input/kobe-bryant-shot-selection/data.csv', header=0, index_col="shot_id", parse_dates=['game_date'])

print("Size of data loaded:", len(df))



df.head()
df.dtypes
df["period"] = df["period"].astype('category')

df["season"] = df["season"].astype('category')

df["team_id"] = df["team_id"].astype('category')

df["game_id"] = df["game_id"].astype('category')

df["opponent"] = df["opponent"].astype('category')

df["playoffs"] = df["playoffs"].astype('category')

df["shot_type"] = df["shot_type"].astype('category')

df["action_type"] = df["action_type"].astype('category')

df["game_event_id"] = df["game_event_id"].astype('category')

df["shot_zone_area"] = df["shot_zone_area"].astype('category')

df["shot_zone_basic"] = df["shot_zone_basic"].astype('category')

df["shot_zone_range"] = df["shot_zone_range"].astype('category')

df["combined_shot_type"] = df["combined_shot_type"].astype('category')
df.isnull().sum()
plt.figure(figsize=(20,5))

sns.countplot('shot_zone_range',hue='shot_made_flag',data=df[df.shot_made_flag.notnull()])

plt.title('misses and baskets from each zone_range')

plt.show()
plt.figure(figsize=(20,5))

sns.countplot('shot_zone_basic',hue='shot_made_flag',data=df[df.shot_made_flag.notnull()])

plt.title('misses and baskets from each zone_basic')

plt.show()
cols = ["combined_shot_type", "period", "playoffs", "season", "shot_type", "shot_zone_area", "shot_zone_basic", "shot_zone_range", "team_id", "team_name", "opponent"]



for c in cols:

    plt.figure(figsize=(20, 8), dpi=80, facecolor='w', edgecolor='k')

    ax = plt.axes()

    sns.countplot(x=c, data=df, ax=ax);

    ax.set_title(c)

    plt.xticks(rotation=90)

    plt.show()
sns.set(style="white")



# Compute the correlation matrix

corr = df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(10, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)

plt.show()
print("Pearson correlation between lat and loc_y variables: %.2f" % pearsonr(df.lat, df.loc_y)[0])

df.drop('lat', axis=1, inplace=True)



print("Pearson correlation between lon and loc_x variables: %.2f" % pearsonr(df.lon, df.loc_x)[0])

df.drop('lon', axis=1, inplace=True)
df.describe(include=['object', 'category'])
print("Different values for team_name variable", set(df.team_name))

df.drop('team_name', axis=1, inplace=True)



print("Different values for team_id variable", set(df.team_id))

df.drop('team_id', axis=1, inplace=True)



df.drop('game_event_id', axis=1, inplace=True)

df.drop('game_id', axis=1, inplace=True)
df[["action_type", "combined_shot_type"]].head(25)
print("Uniques values for action_type:",format(str(len(set(df.action_type)))))

print("Uniques values for combined_shot_type:",format(str(len(set(df.combined_shot_type)))))
#df.drop('action_type', axis=1, inplace=True)

df.head(1)
matchups = list(set(df.matchup.str[-3:]))

print("Number of teams by matches column:", len(matchups))

opponent = list(set(df.opponent))

print("Number of teams by opponent column:", len(opponent))



main_list = list(set(matchups).difference(opponent))

print("\nThere are", len(main_list), "teams incongruous:")

print(main_list)
df[df["matchup"].str.endswith(main_list[0])].head(2) # PHO == PHX --> Phoenix Suns

df[df["matchup"].str.endswith(main_list[1])].head(2) # SAN == SAS --> San Antonio Spurs

df[df["matchup"].str.endswith(main_list[2])].head(2) # CHH == CHA --> Charlotte Horets

df[df["matchup"].str.endswith(main_list[3])].head(2) # UTH == UTA --> Utah Jazz

df[df["matchup"].str.endswith(main_list[4])].head(2) # NOK == NOP --> New Orleans Pelicans https://stats.nba.com/game/0020500903/scoring/
df["home"] = pd.np.where(df.matchup.str.contains("@"), 0, 1)

df["home"] = df["home"].astype('category')

df.drop('matchup', axis=1, inplace=True)
df['remain_time'] = 60*df['seconds_remaining'] + df['minutes_remaining']

df.drop('minutes_remaining', axis=1, inplace=True)

df.drop('seconds_remaining', axis=1, inplace=True)
df.head()
df['year'] = df.game_date.dt.year

df['month'] = df.game_date.dt.month

df['day'] = df.game_date.dt.day
df.drop('game_date', axis=1, inplace=True)
df.head(5)
categorial_cols = df.select_dtypes(include='category').columns



for cc in categorial_cols:

    dummies = pd.get_dummies(df[cc])

    dummies = dummies.add_prefix("{}#".format(cc))

    df.drop(cc, axis=1, inplace=True)

    df = df.join(dummies)

df.head(1)
# Splitting data into train-test

data = df[~df.shot_made_flag.isna()]

submit = df[df.shot_made_flag.isna()]



print("Split dataframe into data-submit: Data:", len(data), "; Submit:", len(submit))

print("\nPercentage for every class:\n", data.shot_made_flag.value_counts()/len(data))
plt.figure(figsize=(8, 5), dpi=80, facecolor='w', edgecolor='k')

ax = plt.axes()

sns.countplot(x='shot_made_flag', data=data, ax=ax);

ax.set_title('Target class distribution')

plt.show()
X_train, X_test, y_train, y_test = train_test_split(

    data.loc[:, data.columns != 'shot_made_flag'], data.shot_made_flag, 

    test_size=0.2, random_state=0, stratify=data.shot_made_flag)



print("Split data into train-test:\nTrain:", len(X_train), "\nTest:", len(X_test),"\n\n")



print("Percentage for train set:\n",y_train.value_counts()/len(y_train),"\n")

print("Percentage for train set:\n",y_test.value_counts()/len(y_test))
#sc = StandardScaler()

sc = MinMaxScaler()



X_train_sc = pd.DataFrame(sc.fit_transform(X_train.values), 

                          index=X_train.index, 

                          columns=X_train.columns)

X_test_sc = pd.DataFrame(sc.transform(X_test.values), 

                          index=X_test.index, 

                          columns=X_test.columns)



X_train_sc
# Create the RFE object and compute a cross-validated score.

ranker = GradientBoostingClassifier()

# The "accuracy" scoring is proportional to the number of correct classifications

rfecv = RFECV(estimator=ranker, step=1, cv=StratifiedKFold(2), n_jobs = 1, scoring='neg_log_loss')

rfecv.fit(X_train_sc, y_train)



print("Optimal number of features based on Gradient Boosting : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores

plt.figure(figsize=(13, 6.5))

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation -Log_Loss")

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()
feature_names = X_train_sc.columns

selected_features_gb = feature_names[rfecv.support_].tolist()

selected_features_gb



print("\nNumber of main features by Linear Discriminant Analysis: {}\n".format(len(selected_features_gb)))

#selected_features_gb
ranker = LinearDiscriminantAnalysis()

# The "accuracy" scoring is proportional to the number of correct classifications

rfecv = RFECV(estimator=ranker, step=1, cv=StratifiedKFold(2), scoring='neg_log_loss')

rfecv.fit(X_train_sc, y_train)



print("Optimal number of features based on LDA : %d" % rfecv.n_features_)



# Plot number of features VS. cross-validation scores

plt.figure(figsize=(13, 6.5))

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()
feature_names = X_train_sc.columns

selected_features_lda = feature_names[rfecv.support_].tolist()

selected_features_lda



print("\nNumber of main features by Linear Discriminant Analysis: {}\n".format(len(selected_features_lda)))
# Create the RFE object and compute a cross-validated score.

ranker = LogisticRegression()

# The "accuracy" scoring is proportional to the number of correct classifications

rfecv = RFECV(estimator=ranker, step=1, cv=StratifiedKFold(2),

              scoring='accuracy')

rfecv.fit(X_train_sc, y_train)



print("Optimal number of features based on Logistic Regression : %d" % rfecv.n_features_)



# Plot number of features VS. cross-validation scores

plt.figure(figsize=(13, 6.5))

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()
feature_names = X_train_sc.columns

selected_features_lr = feature_names[rfecv.support_].tolist()



print("\nNumber of main features by Logistic regression: {}\n".format(len(selected_features_lr)))
num_folds = 5

kfold = KFold(n_splits=num_folds, shuffle = True)



models = []

models.append(("LDA", LinearDiscriminantAnalysis()))

models.append(('Logistic regression', LogisticRegression()))

models.append(('Random Forest', RandomForestClassifier()))

models.append(('Ada Boost', AdaBoostClassifier()))

models.append(('Gradient Boosting', GradientBoostingClassifier()))

models.append(('XGBoost', XGBClassifier()))

models.append(("Bagging", BaggingClassifier()))

models.append(("KNN", KNeighborsClassifier()))

models.append(("MLP", MLPClassifier()))

models.append(("Gauss", GaussianNB()))

models.append(("Voting", VotingClassifier(estimators=[

                                                    ('lr', GradientBoostingClassifier()), 

                                                    ('rf', AdaBoostClassifier()), 

                                                    ('xgb', XGBClassifier())], voting='soft')))







start_time = time.time()

# Evaluate each model in turn

results = []

names = []

stds = []

means =[]

for name, model in models:

    cv_results = cross_val_score(model, X_train_sc[selected_features_gb], y_train, cv=kfold, scoring='neg_log_loss', n_jobs=2)

    print("Cross validation results for {0}: {1}".format(name, cv_results))

    print("{0}: ({1:.4f}) +/- ({2:.4f})".format(name, cv_results.mean(), cv_results.std()),"\n")

    results.append(cv_results)

    names.append(name)

    stds.append(cv_results.std())

    means.append(abs(cv_results.mean()))

    

    

print("--- %s seconds ---" % (time.time() - start_time))
pd.DataFrame({"Name":names, "Log Loss":means, "Standar Deviation": stds}).sort_values(by="Log Loss")
learning_rates = [0.0001, 0.001,0.005, 0.01, 0.05, 0.1, 0.15, 0.5, 1]

train_results = []

test_results = []

for eta in learning_rates:

    model = GradientBoostingClassifier(learning_rate=eta)

    model.fit(X_train_sc[selected_features_gb], y_train)

    train_pred = model.predict(X_train_sc[selected_features_gb])

    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_train, train_pred)

    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

    train_results.append(roc_auc)

    y_pred = model.predict(X_test_sc[selected_features_gb])

    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_pred)

    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

    test_results.append(roc_auc)



plt.figure(figsize=(15,8))

line1, = plt.plot(learning_rates, train_results, 'b', label='Train AUC')

line2, = plt.plot(learning_rates, test_results, 'r', label='Test AUC')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.xscale("log")

plt.ylabel('AUC score')

plt.xlabel('learning rate')

plt.show()
n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200, 500, 1000]

train_results = []

test_results = []

for estimator in n_estimators:

    model = GradientBoostingClassifier(n_estimators=estimator)

    model.fit(X_train_sc[selected_features_gb], y_train)

    train_pred = model.predict(X_train_sc[selected_features_gb])

    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_train, train_pred)

    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

    train_results.append(roc_auc)

    y_pred = model.predict(X_test_sc[selected_features_gb])

    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_pred)

    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

    test_results.append(roc_auc)

    

plt.figure(figsize=(15,8))

line1, = plt.plot(n_estimators, train_results, 'b', label='Train AUC')

line2, = plt.plot(n_estimators, test_results, 'r', label='Test AUC')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('n_estimators')

plt.show()
min_samples_splits = np.linspace(0.0001, 1.0, 10, endpoint=True)

train_results = []

test_results = []

for min_samples in min_samples_splits:

    model = GradientBoostingClassifier(min_samples_split = min_samples)

    model.fit(X_train_sc[selected_features_gb], y_train)

    train_pred = model.predict(X_train_sc[selected_features_gb])

    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_train, train_pred)

    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

    train_results.append(roc_auc)

    y_pred = model.predict(X_test_sc[selected_features_gb])

    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_pred)

    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

    test_results.append(roc_auc)



plt.figure(figsize=(15,8))

line1, = plt.plot(min_samples_splits, train_results, 'b', label='Train AUC')

line2, = plt.plot(min_samples_splits, test_results, 'r', label='Test AUC')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.xscale("log")

plt.ylabel('AUC score')

plt.xlabel('Min. samples split')

plt.show()
min_samples_leafs = [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 4, 8, 15]

train_results = []

test_results = []

for min_samples in min_samples_leafs:

    model = GradientBoostingClassifier(min_samples_leaf = min_samples)

    model.fit(X_train_sc[selected_features_gb], y_train)

    train_pred = model.predict(X_train_sc[selected_features_gb])

    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_train, train_pred)

    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

    train_results.append(roc_auc)

    y_pred = model.predict(X_test_sc[selected_features_gb])

    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_pred)

    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

    test_results.append(roc_auc)



plt.figure(figsize=(15,8))

line1, = plt.plot(min_samples_leafs, train_results, 'b', label='Train AUC')

line2, = plt.plot(min_samples_leafs, test_results, 'r', label='Test AUC')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.xscale("log")

plt.ylabel('AUC score')

plt.xlabel('Min. samples leaf')

plt.show()
max_features = list(range(1,X_train_sc[selected_features_gb].shape[1]))

train_results = []

test_results = []

for max_feature in max_features:

    model = GradientBoostingClassifier(max_features = max_feature)

    model.fit(X_train_sc[selected_features_gb], y_train)

    train_pred = model.predict(X_train_sc[selected_features_gb])

    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_train, train_pred)

    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

    train_results.append(roc_auc)

    y_pred = model.predict(X_test_sc[selected_features_gb])

    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_pred)

    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

    test_results.append(roc_auc)



plt.figure(figsize=(15,8))

line1, = plt.plot(max_features, train_results, 'b', label='Train AUC')

line2, = plt.plot(max_features, test_results, 'r', label='Test AUC')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('Max. features')

plt.show()
classifiers = {}

classifiers.update({"Gradient Boosting": GradientBoostingClassifier()})
parameters = {}

parameters.update({"Gradient Boosting": { 

                                        "classifier__learning_rate":[0.1,0.05,0.01,0.005], 

                                        "classifier__n_estimators": [500],

                                        "classifier__max_depth": [2,3,4,5,6],

                                        "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],

                                        "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],

                                        "classifier__subsample": [0.8, 0.9, 1]

                                         }})

results = {}

start_time = time.time()

# Tune and evaluate classifiers

for classifier_label, classifier in classifiers.items():

    

    # Print message to user

    print(f"Now tuning {classifier_label}.")



    # Initialize Pipeline object

    pipeline = Pipeline([("classifier", classifier)])



    # Define parameter grid

    param_grid = parameters[classifier_label]

    

    # Initialize GridSearch object

    rscv = RandomizedSearchCV(pipeline, param_grid, cv = 5,  n_jobs= -1, verbose = 1, scoring = 'neg_log_loss')



    # Fit gscv

    rscv.fit(X_train_sc[selected_features_gb], np.ravel(y_train))  



    # Get best parameters and score

    best_params = rscv.best_params_

    best_score = rscv.best_score_



    # Update classifier parameters and define new pipeline with tuned classifier

    tuned_params = {item[12:]: best_params[item] for item in best_params}

    classifier.set_params(**tuned_params)



    # Make predictions

    y_pred = rscv.predict_proba(X_test_sc[selected_features_gb])



    # Evaluate model

    log_loss = metrics.log_loss(y_test, y_pred)



    # Save results

    result = {"Classifier": rscv,

              "Best Parameters": best_params,

              "Training Log Loss": (-1) * best_score,

              "Test Log Loss": log_loss

             }



    results.update({classifier_label: result})



print("--- %s seconds ---" % (time.time() - start_time))
log_scores = {

              "Classifier": [],

              "Log Loss": [],

              "Log Loss Type": []

              }



# Get AUC scores into dictionary

for classifier_label in results:

    log_scores.update({"Classifier": [classifier_label] + log_scores["Classifier"],

                       "Log Loss": [results[classifier_label]["Training Log Loss"]] + log_scores["Log Loss"],

                       "Log Loss Type": ["Training"] + log_scores["Log Loss Type"]})

    

    log_scores.update({"Classifier": [classifier_label] + log_scores["Classifier"],

                       "Log Loss": [results[classifier_label]["Test Log Loss"]] + log_scores["Log Loss"],

                       "Log Loss Type": ["Test"] + log_scores["Log Loss Type"]})

    





# Dictionary to PandasDataFrame

log_scores = pd.DataFrame(log_scores)



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

f, ax = plt.subplots(figsize=(10, 5))



sns.barplot(x="Log Loss", y="Classifier", hue="Log Loss Type", palette = colors, data=log_scores)



# Generate a bolded horizontal line at y = 0

ax.axvline(x = 0, color = 'black', linewidth = 4, alpha = .7)



# Turn frame off

ax.set_frame_on(False)



# Tight layout

plt.tight_layout()

best_params
log_scores
train = pd.DataFrame(sc.fit_transform(data.loc[:, data.columns != 'shot_made_flag'].values), 

                          index=data.loc[:, data.columns != 'shot_made_flag'].index, 

                          columns=data.loc[:, data.columns != 'shot_made_flag'].columns)



object_variable = data.shot_made_flag



submit_ = pd.DataFrame(sc.transform(submit.loc[:, submit.columns != 'shot_made_flag'].values), 

                          index=submit.loc[:, submit.columns != 'shot_made_flag'].index, 

                          columns=submit.loc[:, submit.columns != 'shot_made_flag'].columns)



model = classifiers["Gradient Boosting"]

model.fit(train[selected_features_gb], object_variable)

y_pred = model.predict_proba(submit_[selected_features_gb])
predictions = pd.DataFrame({'shot_made_flag' : y_pred[:,1]},

                           index = df[df.shot_made_flag.isnull()].index)

predictions.index.name = 'shot_id'

predictions.to_csv('/kaggle/working/submission_{}.csv'.format(datetime.now().strftime('%Y_%m_%d')))