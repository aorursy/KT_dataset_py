import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, plot_roc_curve, accuracy_score
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-dark')

#import the rookies dataset
rookies_original = pd.read_excel("../input/nba-rookies-stats/NBA_Rookies_by_Year.xlsx")
rookies = rookies_original[rookies_original["Year Drafted"] < 2016]
rookies.index = range(0, len(rookies.index)) 
rookies.head()
rookies
#import players dataset
players_all = pd.read_csv("../input/nba-players-stats-19802017/player_df.csv")
players_all = players_all.drop(players_all.columns[0], axis=1)
players_all.head()
#dropping columns with irregularities
players_all = players_all.drop(["G","OWS","BPM","FG%","2P","FT","DRB","BLK"], axis=1)
players_all.head()
#converting year column to int
players_all = players_all.astype({"Year":int})
players_all
#we can disregard rookies drafted after 2013 because the players dataset only goes up to 2017
rookies = rookies[rookies["Year Drafted"] < 2014]
rookies
#storing rookie info in a dictionary
rkeys_list = list(rookies.loc[:, "Name"])
rval_list = list(rookies.loc[:, "Year Drafted"])
rookie_dict = {k:v for k,v in zip(rkeys_list, rval_list)}
#function that groups active players in a list based on the year
def active_players(year):
    players_year = players_all[players_all["Year"] == year]
    players_year = list(players_year.loc[:, "Player"])
    players_year = [s.strip('*') for s in players_year]
    return players_year

#creating a 2D list where one dimension is the year and the other dimension is the active players
players_by_year = [[None]] * 38
i=0
year = 1980
for year in range(1980, 2018):
    players = active_players(year)
    players_by_year[i] = players
    i+=1
#storing active player info in a dictionary where the key is the year and the value is the active players during that yera#

#keys
keys_list = [year for year in range(1980,2018)]

#creating dictionary
players_dict = {k:v for k,v in zip(keys_list, players_by_year)}
#creating list of players that spent at least 5 years in the league
fivyrs = []
for player, rookie_year in rookie_dict.items():
    target_year = rookie_year + 4
    if player in players_dict[target_year]:
        fivyrs.append(player)
#creating the target column by comparing fivyrs to rookie_dict
target_col = [None]*1424
rookie_names = list(rookies.loc[:, "Name"])
i = 0
for rookie in rookie_names:
    if rookie in fivyrs:
        target_col[i] = 1
    else:
        target_col[i] = 0
    i+=1
target_col = np.array(target_col)
print(target_col)
#adding the target column to the dataframe
target_col = pd.DataFrame(data=target_col, index=[i for i in range(0,len(rookies.index))], columns=["target"])
rookies.index = range(0,len(rookies.index))
rookies["target"] = target_col.loc[:, "target"]
rookies
pd.set_option('display.max_columns', None)
rookies.head(10)
rookies.tail(10)
#Let's find out how many of each class there is
rookies["target"].value_counts()
#Let's visualize this distribution
rookies["target"].value_counts().plot(kind="bar", color=["salmon", "lightblue"])
#Deleting the name column
rookies = rookies.drop(["Name"], axis=1)
rookies
#General description of data
rookies.describe()
#compare target column with year
yr_series = pd.Series(rookies.loc[:, "Year Drafted"])
target_series = pd.Series(rookies.loc[:, "target"])
pd.crosstab(target_series, yr_series)
#visualizing this info
pd.crosstab(yr_series, target_series).plot(kind="bar", figsize=(10,7), color=["salmon", "lightblue"])
plt.title("5yr Survival By Year")
plt.ylabel("Count")
rookies.head()
#PTS Distribution
rookies["PTS"].plot(kind="hist")
#MIN Distribution
rookies["MIN"].plot(kind="hist")
#FG% Distribution
rookies["FG%"].plot(kind="hist")
#3P% Distrbution
rookies["GP"].plot(kind="hist")
rookies.head()
#Correlaton matrix
corr_matrix = rookies.corr()
fig, ax = plt.subplots(figsize=(16,10))
ax = sns.heatmap(corr_matrix, annot=True, linewidths=0.5, fmt=".2f", cmap="YlGnBu")
#Cleaning the 3P% column
rookies["3P%"] = rookies["3P%"].map(lambda x:0 if x=="-" else x)
#Creating Matrix of Features
X = rookies.drop(["target"], axis = 1)
X
#creating target column
y = rookies.loc[:, "target"]
y
rookies.dtypes
# Splitting into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train
# Models dictionary
models = {"Logistic Regression": LogisticRegression(),
         "KNN": KNeighborsClassifier(),
         "Random Forest": RandomForestClassifier(),
         "XGBoost": XGBClassifier()}

#Function that will evaluate the model performance using various metrics
def evaluate_pred(y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    metric_dict = {"accuracy": round(accuracy, 2), "precision": round(precision, 2), "recall": round(recall, 2),
                  "f1": round(f1,2)}
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 score: {f1:.2f}")
    
    return metric_dict

# Function that will fit and score the models
def fit_and_score(models, X_train, X_test, y_train, y_test):
    np.random.seed(42)
    
    #Dictionary of model scores
    model_scores = {}
    
    #Loop through models
    for name, model in models.items():
        clf = model
        clf.fit(X_train, y_train)
        model_scores[name] = model.score(X_test, y_test)
    return model_scores
model_scores = fit_and_score(models, X_train, X_test, y_train, y_test)
model_scores
# Create hyperparameter options
grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}

# Apply grid search
log_clf = GridSearchCV(LogisticRegression(), grid, cv=5, verbose=0)

#Fit
log_clf.fit(X_train, y_train)
#print the best estimator
log_clf.best_estimator_
#evaluating the performance of the best estimator
log_clf1 = LogisticRegression(C=0.001, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
log_clf1.fit(X_train, y_train)
y_pred = log_clf1.predict(X_test)
accuracy_score(y_pred, y_test)
#negligible increase in accuracy
#Constructing the grid
param_test1 = {
 'n_estimators':range(50,200,10),
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}

#Apply grid search
xg_clf = GridSearchCV(XGBClassifier(), param_test1, cv=5, verbose=0)
xg_clf.fit(X_train, y_train)
#Print best estimator
xg_clf.best_estimator_
#evaluating the performance of the best estimator
xg_clf1 = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.300000012, max_delta_step=0, max_depth=7,
              min_child_weight=1, monotone_constraints='()',
              n_estimators=120, n_jobs=0, num_parallel_tree=1,
              objective='binary:logistic', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)

xg_clf1.fit(X_train, y_train)
y_pred = xg_clf1.predict(X_test)
accuracy_score(y_pred, y_test)
#decrease in accuracy
#Desired range for k parameter
k_range = list(range(19, 50))

#Creating grid
param_grid = dict(n_neighbors=k_range)

#Applying GridSearchCV
knn_clf = GridSearchCV(KNeighborsClassifier(), param_grid, cv=10, scoring='accuracy')
knn_clf.fit(X, y)
#printing best estimator
knn_clf.best_estimator_
#evaluating the performance of the best estimator
knn_clf1 = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=37, p=2,
                     weights='uniform')
knn_clf1.fit(X_train, y_train)
y_pred = knn_clf1.predict(X_test)
print(accuracy_score(y_pred, y_test))
#6% increase in accuracy achieved
#Creating the grid
param_grid = {
    'n_estimators'      : range(50,200,10),
    'max_depth'         : [8, 9, 10, 11, 12],
    'random_state'      : [0],
    #'max_features': ['auto'],
    #'criterion' :['gini']
}

#Applying grid search
cv_rfc = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv= 10, scoring='accuracy')
cv_rfc.fit(X_train, y_train)
#printing best estimator
cv_rfc.best_estimator_
# #evaluating performance of best estimator
rfc = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=9, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=340,
                       n_jobs=None, oob_score=False, random_state=0, verbose=0,
                       warm_start=False)

rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
accuracy_score(y_pred, y_test)
#negligible increase in accuracy
#Function that creates visualization for confusion matrix
sns.set(font_scale=1.0)

def plot_conf_mat(y_test, y_preds):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds),
                     annot=True,
                     cbar=False)
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
#function that calculates classification metrics using cross validation
cv_metrics = ["accuracy", "precision", "recall", "f1"]
def cv_calculator(cv_metrics, clf, X, y):
    cv_dict = {}
    for metric in cv_metrics:
        cv_dict[metric] = np.mean(cross_val_score(clf, X, y, cv=5, scoring=metric))
    return cv_dict
#Plot ROC Curve and calculate AUC for XGB
plot_roc_curve(xg_clf1, X_test, y_test)
#confusion matrix for XGB
y_pred1 = xg_clf1.predict(X_test)
plot_conf_mat(y_pred1, y_test)
#cross validated classification metrics for XGB
cv_dict = cv_calculator(cv_metrics, xg_clf1, X, y)
cv_dict
#visualize the cv metrics
cv_metrics1 = pd.DataFrame(cv_dict, index=["score"])
cv_metrics1.T.plot.bar(title="XGB CV Metrics", legend=False)
#feature importance XGB
plt.figure(figsize=(15, 5))
plt.bar(list(X_train.columns), xg_clf1.feature_importances_, align='edge', width=0.3)
plt.show()
#Plot ROC Curve and calculate AUC for Logistic Regression
plot_roc_curve(log_clf1, X_test, y_test)
#confusion matrix for Log Reg
y_pred2 = log_clf1.predict(X_test)
plot_conf_mat(y_pred2, y_test)
#cross validated classification metrics for Log Reg
cv_dict2 = cv_calculator(cv_metrics, log_clf, X, y)
cv_dict2
#visualize the cv metrics
cv_metrics2 = pd.DataFrame(cv_dict2, index=["score"])
cv_metrics2.T.plot.bar(title="Log Reg CV Metrics", legend=False)
#feature importance log reg#

#Match coefficients to corresponding columns
feature_dict = dict(zip(rookies.columns, list(log_clf1.coef_[0])))

#Visualize feature importance
plt.figure(figsize=(15, 5))
feature_df = pd.DataFrame(feature_dict, index=[0])
feature_df.T.plot.bar(title="Feature Importance", legend=False)
#Plot ROC Curve and calculate AUC KNeighbors
plot_roc_curve(knn_clf1, X_test, y_test)
#confusion matrix for KNN
y_pred3 = knn_clf1.predict(X_test)
plot_conf_mat(y_pred3, y_test)
#cross validated classification metrics for KNN
cv_dict3 = cv_calculator(cv_metrics, knn_clf1, X, y)
cv_dict3
#visualize the cv metrics
cv_metrics3 = pd.DataFrame(cv_dict3, index=["score"])
cv_metrics3.T.plot.bar(title="KNN CV Metrics", legend=False)
#Plot ROC Curve and calculate AUC for Random Forest
plot_roc_curve(rfc, X_test, y_test)
#confusion matrix for RFC
y_pred4 = rfc.predict(X_test)
plot_conf_mat(y_pred4, y_test)
#cross validated classification metrics RF
cv_dict4 = cv_calculator(cv_metrics, rfc, X, y)
cv_dict4
#visualize the cv metrics
cv_metrics4 = pd.DataFrame(cv_dict4, index=["score"])
cv_metrics4.T.plot.bar(title="Random Forest CV Metrics", legend=False)
#feature importance for random forest#

#creating feature importance dictionary
features_dict2 = dict(zip(rookies.columns , rfc.feature_importances_))

#visualizing feature importance
plt.figure(figsize=(15, 5))
feature_df = pd.DataFrame(feature_dict, index=[0])
feature_df.T.plot.bar(title="Random Forest Feature Importance", legend=False)