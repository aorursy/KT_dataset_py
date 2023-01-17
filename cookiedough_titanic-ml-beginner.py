import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

# Visualization
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
# Import data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

# Preview data
df_train.head()
plt.figure(figsize=(15, 9))

plt.subplot(3, 3, 1)
ax = df_train["Pclass"].hist(bins=3, density=True, stacked=True, color='teal', alpha=0.6)
df_train["Pclass"].plot(bw_method=0.5, kind='density', color='teal')
ax.set(xlabel='Pclass')
plt.xlim(1,3)

plt.subplot(3, 3, 2)
ax = df_train["Age"].hist(bins=19, density=True, stacked=True, color='teal', alpha=0.6)
df_train["Age"].plot(kind='density', color='teal')
ax.set(xlabel='Age')
plt.xlim(-10,85)

plt.subplot(3, 3, 3)
ax = df_train["SibSp"].hist(bins=8, density=True, stacked=True, color='teal', alpha=0.6)
df_train["SibSp"].plot(bw_method=1, kind='density', color='teal')
ax.set(xlabel='SibSp')
plt.xlim(df_train["SibSp"].min(), df_train["SibSp"].max())

plt.subplot(3, 3, 4)
ax = df_train["Parch"].hist(bins=6, density=True, stacked=True, color='teal', alpha=0.6)
df_train["Parch"].plot(bw_method=1, kind='density', color='teal')
ax.set(xlabel='Parch')
plt.xlim(df_train["Parch"].min(), df_train["Parch"].max())

plt.subplot(3, 3, 5)
cabins = ["A", "B", "C", "D", "E", "F", "G", "T"]
cabins_df = df_train
for cabin in cabins:
    cabins_df = cabins_df.replace({'Cabin': r''+cabin+'.*'}, {'Cabin': cabin}, regex=True)
cabins_df["Cabin"].value_counts().plot(kind='bar', color='teal', alpha=0.6)

plt.subplot(3, 3, 6)
df_train["Embarked"].value_counts().plot(kind='bar', color='teal', alpha=0.6)

plt.subplot(3, 3, 7)
ax = df_train["Fare"].hist(bins=50, density=True, stacked=True, color='teal', alpha=0.6)
df_train["Fare"].plot(bw_method=1, kind='density', color='teal')
ax.set(xlabel='Fare')
plt.xlim(df_train["Fare"].min(), df_train["Fare"].max())

plt.show()
def preprocess_features(dataframe):
    selected_features = dataframe[
        ["Pclass"]
    ]
    processed_features = selected_features.copy()
    processed_features["Male"] = (dataframe["Sex"] == 'male').astype(int)
    processed_features["Female"] = (dataframe["Sex"] == 'female').astype(int)
    processed_features["FamilySize"] = dataframe["SibSp"] + dataframe["Parch"]

    processed_features["Age_0-10"] = (dataframe["Age"] < 10).astype(int)
    processed_features["Age_10-25"] = ((dataframe["Age"] >= 10) & (dataframe["Age"] < 25)).astype(int)
    processed_features["Age_25-40"] = ((dataframe["Age"] >= 25) & (dataframe["Age"] < 40)).astype(int)
    processed_features["Age_40+"] = ((dataframe["Age"] >= 40)).astype(int)

    processed_features["Fare_0-5"] = (dataframe["Fare"] < 5).astype(int)
    processed_features["Fare_5-30"] = ((dataframe["Fare"] >= 5) & (dataframe["Age"] < 30)).astype(int)
    processed_features["Fare_30-90"] = ((dataframe["Fare"] >= 30) & (dataframe["Age"] < 90)).astype(int)
    processed_features["Fare_90+"] = ((dataframe["Fare"] >= 90)).astype(int)

    cabins = ["A", "B", "C", "D", "E", "F", "G", "T"]
    for cabin in cabins:
        processed_features["Cabin_%s" % cabin] = dataframe["Cabin"].apply(
            lambda x: 1 if not pd.isnull(x) and x.find(cabin) != -1 else 0
        )

    # Normalize data
    processed_features["Pclass"] = (processed_features["Pclass"] - processed_features["Pclass"].min()) / (processed_features["Pclass"].max() - processed_features["Pclass"].min())
    processed_features["FamilySize"] = (processed_features["FamilySize"] - processed_features["FamilySize"].min()) / (processed_features["FamilySize"].max() - processed_features["FamilySize"].min())

    return processed_features
def preprocess_targets(dataframe):
    return dataframe["Survived"]
# Shuffle data
df_train = df_train.reindex(np.random.permutation(df_train.index))

# Preprocess training data
x_train = preprocess_features(df_train)
y_train = preprocess_targets(df_train)

print("Data shape:\n", x_train.shape, y_train.shape)
print("\nData mean\n", np.mean(x_train))
print("\nData standard derivation\n", np.std(x_train))

# Preprocess test data
x_test = preprocess_features(df_test)
x_train.head()
# Cross validate model with Kfold stratified cross validation
skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2)
# Test different algorithms 
random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state, gamma='auto'))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())

cv_results = []
for classifier in classifiers:
    cv_results.append(cross_val_score(classifier, x_train, y_train, scoring = "accuracy", cv = skfold))
    
cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({
    "CrossValMeans": cv_means,
    "CrossValErrors": cv_std,
    "Algorithm": [
        "SVC",
        "DecisionTree",
        "AdaBoost",
        "RandomForest",
        "ExtraTrees",
        "GradientBoosting",
        "KNeighboors",
        "LogisticRegression",
        "LinearDiscriminantAnalysis"
    ]
})

print(cv_res)
fig, ax = plt.subplots()

algorithms = cv_res['Algorithm']
means = cv_res['CrossValMeans']
errors = cv_res['CrossValErrors']

y_pos = np.arange(len(means))

plt.barh(y_pos, means, xerr=errors, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(algorithms)
ax.invert_yaxis()
ax.set_xlabel('Mean Accuracy')
ax.set_ylabel('Algorithm')
ax.set_title('Cross validation scores')

plt.show()
ABC = AdaBoostClassifier()
abc_param_grid = {
    "learning_rate": [1.3, 1, 0.8, 0.5],
    "n_estimators" : [3, 4, 5, 10]
}

gsABC = GridSearchCV(
    ABC,
    param_grid = abc_param_grid,
    cv = skfold,
    scoring = "accuracy",
    n_jobs = 4,
    verbose = 1
)

gsABC.fit(x_train, y_train)

ABC_best = gsABC.best_estimator_

# Best score
gsABC.best_score_
RFC = RandomForestClassifier()
rf_param_grid = {
    "max_depth": [None],
    "max_features": [1, 3, 5],
    "min_samples_split": [3, 5, 10, 15],
    "min_samples_leaf": [5, 10],
    "bootstrap": [False],
    "n_estimators" : [100, 300, 500]
}

gsRFC = GridSearchCV(
    RFC,
    param_grid = rf_param_grid,
    cv = skfold,
    scoring = "accuracy",
    n_jobs = 4,
    verbose = 1
)

gsRFC.fit(x_train, y_train)

RFC_best = gsRFC.best_estimator_

# Best score
gsRFC.best_score_
ExtC = ExtraTreesClassifier()
ex_param_grid = {
    "max_depth": [None],
    "max_features": [1, 3, 5, 10],
    "min_samples_split": [30, 50, 80],
    "min_samples_leaf": [2, 3, 5],
    "bootstrap": [False],
    "n_estimators": [10, 30]
}

gsExtC = GridSearchCV(
    ExtC,
    param_grid = ex_param_grid,
    cv = skfold,
    scoring = "accuracy",
    n_jobs = 4,
    verbose = 1
)

gsExtC.fit(x_train, y_train)

ExtC_best = gsExtC.best_estimator_

# Best score
gsExtC.best_score_
GBC = GradientBoostingClassifier()
gb_param_grid = {
    'loss' : ["deviance"],
    'n_estimators' : [300, 500],
    'learning_rate': [3, 1, 0.8],
    'max_depth': [2, 4],
    'min_samples_leaf': [30, 50, 80],
    'max_features': [0.3, 0.1]
}

gsGBC = GridSearchCV(
    GBC,
    param_grid = gb_param_grid,
    cv = skfold,
    scoring = "accuracy",
    n_jobs = 4,
    verbose = 1
)

gsGBC.fit(x_train, y_train)

GBC_best = gsGBC.best_estimator_

# Best score
gsGBC.best_score_
LogReg = LogisticRegression()
lg_param_grid = {
    "C": [0.1, 0.2, 0.3],
    "tol": [0.3, 0.4, 0.5],
    "solver": ["liblinear", "saga"],
    "max_iter": [30, 60, 80, 100]
}

gsLogReg = GridSearchCV(
    LogReg,
    param_grid = lg_param_grid,
    cv = skfold,
    scoring = "accuracy",
    n_jobs = 4,
    verbose = 1
)

gsLogReg.fit(x_train, y_train)

LogReg_best = gsLogReg.best_estimator_

# Best score
gsLogReg.best_score_
def plot_learning_curve(estimator, title, x, y, cv=None):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(estimator, x, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()

    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r"
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g"
    )
    plt.plot(
        train_sizes,
        train_scores_mean,
        'o-',
        color="r",
        label="Training score"
    )
    plt.plot(
        train_sizes,
        test_scores_mean,
        'o-',
        color="g",
        label="Test score"
    )

    plt.legend(loc="best")
    return plt
plot_learning_curve(gsABC.best_estimator_, "Ada Boost learning curves", x_train, y_train, cv=skfold)
plot_learning_curve(gsRFC.best_estimator_, "Random Forest learning curves", x_train, y_train, cv=skfold)
plot_learning_curve(gsExtC.best_estimator_, "Extra Trees learning curves", x_train, y_train, cv=skfold)
plot_learning_curve(gsGBC.best_estimator_, "Gradient Boosting learning curves", x_train, y_train, cv=skfold)
plot_learning_curve(gsLogReg.best_estimator_, "Logistic Regression learning curves", x_train, y_train, cv=skfold)
null_accuracy = 1 - y_train.mean()
print("The null accuracy is %0.2f%%" % (null_accuracy * 100))
votingC = VotingClassifier(
    estimators = [('abc', ABC_best), ('rfc', RFC_best), ('extc', ExtC_best), ('gbc', GBC_best), ('logreg', LogReg_best)],
    voting = 'soft',
    n_jobs = 4
)

votingC = votingC.fit(x_train, y_train)

final_accuracy = votingC.score(x_train, y_train)
print("The final accuracy score for all chosen classifiers combined: %0.2f%%" % (final_accuracy * 100))
final_predictions = votingC.predict(x_test)
result = pd.DataFrame({
    "PassengerId": df_test["PassengerId"], 
    "Survived": final_predictions
})

# Save the submission file
result.to_csv('submission.csv', index=False)
