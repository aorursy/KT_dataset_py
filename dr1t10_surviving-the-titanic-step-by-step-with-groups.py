import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import jaccard_similarity_score

from scipy.stats import randint, uniform

from xgboost import XGBClassifier

import warnings


# Silence pesky deprecation warnings from sklearn
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
sns.set_palette('deep')
random_state = 1
_train = pd.read_csv("../input/train.csv")
_test = pd.read_csv("../input/test.csv")

# Make a copy to be modified
train = _train.copy()
test = _test.copy()
train_len = len(train)
print("Training dataset size = {}".format(train_len))
train.head()
test_len = len(test)
print("Test dataset size = {}".format(test_len))
test.head()
# Remove PassengerId since it's not a feature
train.drop(columns='PassengerId', inplace=True)
test.drop(columns='PassengerId', inplace=True)
# Concatenate the train and test datasets. Survived is not a feature so we drop it
dataset = pd.concat([train, test], sort=False).drop(columns='Survived')
pd.DataFrame({'No. NaN': dataset.isna().sum(), '%': dataset.isna().sum() / len(dataset)})
train.drop(columns='Cabin', inplace=True)
test.drop(columns='Cabin', inplace=True)
with sns.axes_style("darkgrid"):
    g = sns.FacetGrid(train, hue='Survived', height=5, aspect=2.5)
    g.map(sns.kdeplot, 'Age', shade=True)
    g.add_legend()
    g.set(xticks=np.arange(0, train['Age'].max() + 1, 5), xlim=(0, train['Age'].max()))
# Use KMeans to cluster `Age`
num_clusters = 4
X = train[[ 'Age', 'Survived']].dropna()
kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
kmeans.fit(X)
X['AgeCluster'] = kmeans.labels_

# Plot the decision boundary
# See http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
plt.figure(figsize=(10,5))
h = 0.01
x_min, x_max = X['Age'].min() - h, X['Age'].max() + h
y_min, y_max = X['Survived'].min() - h, X['Survived'].max() + h
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict the age cluster for each point in a mesh
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
cmap = sns.cubehelix_palette(start=2.8, rot=.1, as_cmap=True)
Z = Z.reshape(xx.shape)
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=cmap, aspect='auto')

# Plot the ages
sns.scatterplot(x='Age', y='Survived', hue='AgeCluster', data=X, palette=cmap)

# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w')
plt.yticks([0, 1])
plt.title("Age clusters and decision boundaries")
plt.show()
# Convert K-means clusters to age bands
age_bands = []
for k in range(num_clusters):
    age_bands.append(xx[Z==k].min())

# Since the clusters are not sorted we sort the intervals
age_bands.sort()

# Set the lower bound of the first interval to 0
age_bands[0] = 0

# Set the higher bound of the last interval to infinite just in case there are older older passengers in the test set
age_bands.append(np.inf)

# Convert list to numpy array
print("Age bands: {}".format(np.array(age_bands)))
# Use both the training and test dataset to fill the missing Age values
dataset = pd.concat([train, test], sort=True)
dataset['AgeBand'] = pd.cut(dataset['Age'], age_bands)

dataset.groupby('AgeBand')['Survived'].mean()
# Use both the training and test dataset to fill the missing Age values
fill_age_df = dataset[['Name', 'AgeBand', 'Pclass', 'SibSp', 'Parch', 'Sex']].copy()

# Get the titles of the passengers
fill_age_df['Title'] = fill_age_df['Name'].apply(lambda x: x[x.find(', ') + 2:x.find('.')])

pd.crosstab(fill_age_df['Title'], fill_age_df['Sex']).transpose()
# Join Mlle (Mademoiselle) and Ms with Miss - Mlle and Miss both indicate a unmarried status while Ms is more generic
# it can mean both Miss and Mrs, I chose Miss because it's more frequent
fill_age_df['Title'].replace(to_replace=['Mlle', 'Ms'], value='Miss', inplace=True)

# Join Mme (Madame) with Mrs - Mme and Mrs both indicate a married status
fill_age_df['Title'].replace(to_replace='Mme', value='Mrs', inplace=True)

# Join the remaining titles with low frequencies
fill_age_df['Title'].replace(to_replace=['Don', 'Rev', 'Dr', 'Major', 'Lady', 'Sir',
                                         'Col', 'Capt', 'the Countess', 'Jonkheer', 'Dona'],
                             value='Rare', inplace=True)

title_dummies = pd.get_dummies(fill_age_df['Title'], drop_first=True)
fill_age_df = pd.concat([fill_age_df, title_dummies], axis=1)
fill_age_df.head(2)
# Encode ages and sex
fill_age_df['IsMale'] = fill_age_df['Sex'].astype('category').cat.codes
fill_age_df['AgeBand'] = fill_age_df['AgeBand'].astype('category').cat.codes

# Drop columns we no longer need
fill_age_df.drop(columns=['Name', 'Sex', 'Title'], inplace=True)

fill_age_df.head(2)
# Drop all rows with unknown age bands (-1) from the training set
X_train = fill_age_df.loc[fill_age_df['AgeBand'] != -1].drop(columns='AgeBand')
Y_train = fill_age_df['AgeBand'].loc[fill_age_df['AgeBand'] != -1]

# Get all rows with unknown age bands (-1) for the test set
X = fill_age_df.loc[fill_age_df['AgeBand'] == -1].drop(columns='AgeBand')

# Some shapes to double-check
print("Training samples shape: {}".format(X_train.shape))
print("Training labels shape: {}".format(Y_train.shape))
print("Samples to predict shape: {}".format(X.shape))
logreg = LogisticRegression(random_state=random_state)
logreg_scores = cross_val_score(logreg, X_train, Y_train, cv=10)
print("Logistic Regression cross-validation scores: {:.3f}".format(logreg_scores.mean()))

knn = KNeighborsClassifier()
knn_scores = cross_val_score(knn, X_train, Y_train, cv=10)
print("KNeighbors cross-validation scores: {:.3f}".format(knn_scores.mean()))

tree = DecisionTreeClassifier(random_state=random_state)
tree_scores = cross_val_score(tree, X_train, Y_train, cv=10)
print("Tree Classifier cross-validation scores: {:.3f}".format(tree_scores.mean()))
# Make predictions using our best model (Decision Tree)
tree.fit(X_train, Y_train)
Y_pred = tree.predict(X_train)

# Compute the per-class/per-age band accuracy
total = np.bincount(Y_train.values, minlength=4)
correct = np.bincount(Y_pred[Y_pred == Y_train.values], minlength=4)
class_acc = correct / total
pd.DataFrame({'AgeBand': np.arange(4),
              'AgeInterval': dataset['AgeBand'].cat.categories,
              'PerClassAcc.': class_acc})
# Feature importance
g = sns.barplot(x=tree.feature_importances_, y=X.columns, orient='h')
_ = g.set_xlabel('Relative importance')
_ = g.set_ylabel('Features')
_ = g.set_title('Feature Importance')
Y = tree.predict(X)
fill_age_df.loc[fill_age_df['AgeBand'] == -1, 'AgeBand'] = Y
train['AgeBand'] = fill_age_df.iloc[:train_len, 0]
test['AgeBand'] = fill_age_df.iloc[train_len:, 0]

# Remove Age column
train.drop(columns='Age', inplace=True)
test.drop(columns='Age', inplace=True)

train.head(2)
_ = sns.countplot(x='Sex', hue='Survived', data=train)
train.groupby('Sex')['Survived'].mean()
# One-encode using dummies
train['IsMale'] = pd.get_dummies(train['Sex'], drop_first=True)
test['IsMale'] = pd.get_dummies(test['Sex'], drop_first=True)

# Drop the Sex column
train.drop(columns='Sex', inplace=True)
test.drop(columns='Sex', inplace=True)
_ = sns.countplot(x='Pclass', hue='Survived', data=train)
train.groupby('Pclass')['Survived'].mean()
_ = sns.countplot(x='Embarked', hue='Survived', data=train)
# Plot relationship between Embarked, Pclass, and IsMale
_ = sns.factorplot(x='Embarked', col='Pclass', row='IsMale', data=train, kind='count')
# Passengers grouped by Embarked, Pclass, and IsMale
embarked_corr = (train[['Survived', 'Embarked', 'Pclass', 'IsMale']].groupby(['Embarked', 'Pclass', 'IsMale'])                                            
                                                                    .agg(['count', 'sum', 'mean']))
embarked_corr.columns = embarked_corr.columns.droplevel(0)
embarked_corr.columns = ['Total', 'Survived', 'Rate']
embarked_corr
train.drop(columns='Embarked', inplace=True)
test.drop(columns='Embarked', inplace=True)
dataset = pd.concat([train, test], sort=True)
dataset['TicketFreq'] = dataset.groupby('Ticket')['Ticket'].transform('count')
dataset['PassengerFare'] = dataset['Fare'] / dataset['TicketFreq']

train.head(2)
num_fare_bins = 3
dataset['FareBand'], fare_bins = pd.qcut(dataset['PassengerFare'], num_fare_bins, retbins=True)
_ = sns.countplot(x='FareBand', hue='Survived', data=dataset)
_ = plt.xticks(rotation=30, ha='right')
# Plot relationship between FareBand and Pclass
g = sns.factorplot(x='FareBand', col='Pclass', data=dataset, kind='count')
_ = g.set_xticklabels(rotation=30, ha='right')
band = pd.Interval(left=7.775, right=13.0)
mask = (dataset['FareBand'] == band) & (dataset['Pclass'] != 1)

dataset.loc[mask, ['FareBand', 'Pclass', 'PassengerFare']].groupby(['FareBand', 'Pclass']).agg(['mean'])
band1 = pd.Interval(left=0, right=7.775)
band2 = pd.Interval(left=7.775, right=13.0)
mask = ((dataset['FareBand'] == band1) | (dataset['FareBand'] == band2)) & (dataset['Pclass'] == 3)

dataset.loc[mask, ['FareBand', 'Pclass', 'Survived']].groupby(['FareBand', 'Pclass']).agg(['mean'])
dataset.loc[mask, ['FareBand', 'Pclass', 'AgeBand']].groupby(['FareBand', 'Pclass']).agg(['value_counts']).sort_index()
train.drop(columns='Fare', inplace=True)
test.drop(columns='Fare', inplace=True)

train.head(2)
# Create the new feature for the whole dataset 
# (I am using _train and _test because train and test no longer contain 'Embarked')
dataset = pd.concat([_train, _test], sort=True, ignore_index=True)
surname = dataset['Name'].apply(lambda x: x[:x.find(',')])
ticket = dataset['Ticket'].apply(lambda x: x[:-1])

dataset['SPTE'] = (surname.astype(str) + '-' + dataset['Pclass'].astype(str) + '-'
           + ticket.astype(str) + '-' + dataset['Embarked'].astype(str))

spte_count = dataset['SPTE'].value_counts(sort=False)

def spte_group_lebeler(group):
    group_elements = dataset.loc[dataset['SPTE'] == group, 'PassengerId']
    if len(group_elements) == 1:
        return 0
    else:
        return group_elements.min()

dataset['GroupId'] = dataset['SPTE'].apply(spte_group_lebeler)
dataset.drop(columns='SPTE', inplace=True)
dataset.tail()
# Groups that share the same ticket number
def ticket_group_labeler(group):
    unique_groups = group.unique()
    if len(unique_groups) == 1:
        return unique_groups[0]
    elif len(unique_groups) == 2 and min(unique_groups) == 0:
        return dataset.loc[group.index, 'PassengerId'].min()
    else:
        raise ValueError("Found conflict between SPTE and ticket grouping:\n\n{}".format(dataset.loc[group.index]))

dataset['GroupId'] = dataset.groupby('Ticket')['GroupId'].transform(ticket_group_labeler)
dataset.tail()
# Calculate the size of each group
dataset['GroupSize'] = dataset.groupby('GroupId')['GroupId'].transform('count')
dataset.loc[dataset['GroupId'] == 0, 'GroupSize'] = 1

# InGroup is 1 for groups with more than one member
dataset['InGroup'] = (dataset['GroupSize'] > 1).astype(int)

# Add to the train and test datasets
train['InGroup'] = dataset.iloc[:train_len, -1]
test['InGroup'] = dataset.iloc[train_len:, -1].reset_index(drop=True)

_ = sns.countplot(x='InGroup', hue='Survived', data=train)
train.groupby('InGroup')['Survived'].mean()
# Get the titles of the passengers
dataset['Title'] = dataset['Name'].apply(lambda x: x[x.find(', ') + 2:x.find('.')])

# Create a mask to account only for females or boys in groups
mask = (dataset['GroupId'] != 0) & ((dataset['Title'] == 'Master') | (dataset['Sex'] == 'female'))

# Get the number of females and boys in each group, discard groups with only one member
wcg_groups = dataset.loc[mask, 'GroupId'].value_counts()
wcg_groups = wcg_groups[wcg_groups > 1]

# Update the mask to discard groups with only one female or boy
mask = mask & (dataset['GroupId'].isin(wcg_groups.index))

# Create the new feature using the updated mask
dataset['InWcg'] = 0
dataset.loc[mask, 'InWcg'] = 1

print("Number of woman-child-groups found:", len(wcg_groups))
print("Number of passengers in woman-child-groups:", len(dataset.loc[dataset['InWcg'] == 1]))

# Add to the train and test datasets
train['InWcg'] = dataset.iloc[:train_len, -1]
test['InWcg'] = dataset.iloc[train_len:, -1].reset_index(drop=True)
dataset['WcgAllSurvived'] = dataset.loc[dataset['InWcg'] == 1].groupby('GroupId')['Survived'].transform(np.nanmean)

# `np.nanmean` returns NaN for groups without survival information (test set only groups)
# Replace the NaN with 0
dataset.loc[dataset['WcgAllSurvived'].isna(), 'WcgAllSurvived'] = 0
dataset['WcgAllSurvived'] = dataset['WcgAllSurvived'].astype(int)

# Add to the train and test datasets
train['WcgAllSurvived'] = dataset.iloc[:train_len, -1]
test['WcgAllSurvived'] = dataset.iloc[train_len:, -1].reset_index(drop=True)
dataset['WcgAllDied'] = (1 - dataset.loc[dataset['InWcg'] == 1].groupby('GroupId')['Survived'].transform(np.nanmean))

# `np.nanmean` returns NaN for groups without survival information (test set only groups)
# Replace the NaN with 0
dataset.loc[dataset['WcgAllDied'].isna(), 'WcgAllDied'] = 0
dataset['WcgAllDied'] = dataset['WcgAllDied'].astype(int)

# Add to the train and test datasets
train['WcgAllDied'] = dataset.iloc[:train_len, -1]
test['WcgAllDied'] = dataset.iloc[train_len:, -1].reset_index(drop=True)
train.drop(columns=['Name', 'SibSp', 'Parch', 'Ticket'], inplace=True)
test.drop(columns=['Name', 'SibSp', 'Parch', 'Ticket'], inplace=True)
train.head()
test.head()
# Split the training set into samples and targets
X_train = train.drop(columns='Survived')
Y_train = train['Survived'].astype(int)

# Test set samples to predict
X_test = test

# Scale features such that the mean is 0 and standard deviation is 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Number of cross-validation folds
k_folds = 10

# Number of estimators for tree-based ensembles
n_estimators = 100

# Create a dictionary containing the instance of the models, scores, mean accuracy and standard deviation
classifiers = {
    'name': ['DecisionTree', 'RandomForest', 'ExtraTrees', 'AdaBoost', 'LogReg', 'KNN', 'SVC',
             'XGBoost', 'GradientBoost'],
    'models': [DecisionTreeClassifier(random_state=random_state),
               RandomForestClassifier(random_state=random_state, n_estimators=n_estimators),
               ExtraTreesClassifier(random_state=random_state, n_estimators=n_estimators),
               AdaBoostClassifier(random_state=random_state, n_estimators=n_estimators),
               LogisticRegression(random_state=random_state),
               KNeighborsClassifier(),
               SVC(random_state=random_state),
               XGBClassifier(random_state=random_state, n_estimators=n_estimators),
               GradientBoostingClassifier(random_state=random_state, n_estimators=n_estimators)], 
    'scores': [],
    'acc_mean': [],
    'acc_std': []
}

# Run cross-validation and store the scores
for model in classifiers['models']:
    score = cross_val_score(model, X_train, Y_train, cv=k_folds, n_jobs=4)
    classifiers['scores'].append(score)
    classifiers['acc_mean'].append(score.mean())
    classifiers['acc_std'].append(score.std())    

# Create a nice table with the results
classifiers_df = pd.DataFrame({
    'Model Name': classifiers['name'],
    'Accuracy': classifiers['acc_mean'],
    'Std': classifiers['acc_std']
}, columns=['Model Name', 'Accuracy', 'Std']).set_index('Model Name')

classifiers_df.sort_values('Accuracy', ascending=False)
# Utility function to report best scores
# Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
def report(results, n_top=3, limit=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        if limit is not None:
            candidates = candidates[:limit]
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.4f} (std: {1:.4f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print()

# Number of iterations
n_iter_search = 200
logreg = LogisticRegression(random_state=random_state)
rand_param = {
    'penalty': ['l1', 'l2'],
    'C': uniform(0.01, 10)
 }

logreg_search = RandomizedSearchCV(logreg, param_distributions=rand_param, n_iter=n_iter_search, cv=k_folds, n_jobs=4, verbose=1)
logreg_search.fit(X_train, Y_train)
report(logreg_search.cv_results_)

logreg_best = logreg_search.best_estimator_
svc = SVC(random_state=random_state, probability=True)
rand_param = {
    'C': uniform(0.01, 10),
    'gamma': uniform(0.01, 10)
 }

svc_search = RandomizedSearchCV(svc, param_distributions=rand_param, n_iter=n_iter_search, cv=k_folds, n_jobs=4, verbose=1)
svc_search.fit(X_train, Y_train)
report(svc_search.cv_results_)

svc_best = svc_search.best_estimator_
knn = KNeighborsClassifier()
rand_param = {
    'n_neighbors': randint(1, 25),
    'leaf_size': randint(1, 50),
    'weights': ['uniform', 'distance']
}

knn_search = RandomizedSearchCV(knn, param_distributions=rand_param, n_iter=n_iter_search, cv=k_folds, n_jobs=4, verbose=1)
knn_search.fit(X_train, Y_train)
report(knn_search.cv_results_)

knn_best = knn_search.best_estimator_
ada = AdaBoostClassifier(random_state=random_state, n_estimators=n_estimators)
rand_param = {
    'learning_rate': uniform(0.1, 10),
}

ada_search = RandomizedSearchCV(ada, param_distributions=rand_param, n_iter=n_iter_search, cv=k_folds, n_jobs=4, verbose=1)
ada_search.fit(X_train, Y_train)
report(ada_search.cv_results_)

ada_best = ada_search.best_estimator_
g = sns.barplot(x=ada_best.feature_importances_, y=test.columns, orient='h')
_ = g.set_xlabel('Relative importance')
_ = g.set_ylabel('Features')
etc = ExtraTreesClassifier(random_state=random_state, n_estimators=n_estimators)
rand_param = {
    'bootstrap': [True, False],
    'max_depth': np.append(randint(1, 10).rvs(10), None),
    'max_features': randint(1, X_train.shape[1]), # From 1 to number of features is a good range
    'min_samples_split': randint(2, 10)
}

etc_search = RandomizedSearchCV(etc, param_distributions=rand_param, n_iter=n_iter_search, cv=k_folds, n_jobs=4, verbose=1)
etc_search.fit(X_train, Y_train)
report(etc_search.cv_results_)

etc_best = etc_search.best_estimator_
g = sns.barplot(x=etc_best.feature_importances_, y=test.columns, orient='h')
_ = g.set_xlabel('Relative importance')
_ = g.set_ylabel('Features')
logreg_pred = logreg_best.predict(X_test)
svc_pred = svc_best.predict(X_test)
knn_pred = knn_best.predict(X_test)
ada_pred = ada_best.predict(X_test)
etc_pred = etc_best.predict(X_test)

# Make a data frame with the predictions from all models
pred_df = pd.DataFrame({
    'Logistic Regression': logreg_pred,
    'SVC': svc_pred,
    'KNN': knn_pred,
    'AdaBoost': ada_pred,
    'Extra Tress': etc_pred
})

jsim_df = pd.DataFrame(np.nan, columns=pred_df.columns, index=pred_df.columns)
for i in pred_df.columns:
    for j in pred_df.loc[:, i:].columns:
        jsim_df.loc[i, j] = jaccard_similarity_score(pred_df[i], pred_df[j])
        jsim_df.loc[j, i] = jsim_df.loc[i, j]

_ = sns.heatmap(jsim_df, linewidths=0.1, vmax=1.0, vmin=0, square=True, linecolor='white', annot=True, cmap='coolwarm')
estimators = [
    ('Logistic Regression', logreg_best),
    ('SVC', svc_best),
    ('KNN', knn_best),
    ('AdaBoost', ada_best),
    ('Extra Trees', etc_best)
]

eclf = VotingClassifier(estimators=estimators)
ensemble_param = {'voting': ['hard', 'soft']}

eclf_search = GridSearchCV(eclf, param_grid=ensemble_param, cv=k_folds, n_jobs=4, verbose=1)
eclf_search.fit(X_train, Y_train)
report(eclf_search.cv_results_)

eclf_best = eclf_search.best_estimator_
best_model = KNeighborsClassifier(leaf_size=3, weights='uniform', n_neighbors=19)
score = cross_val_score(best_model, X_train, Y_train, cv=k_folds, n_jobs=4)
best_model.fit(X_train, Y_train)
print("Cross-validation accuracy: {0:.4f}".format(score.mean()))

# Prediction
best_pred = best_model.predict(X_test)
submission_df = pd.DataFrame({'PassengerId': _test['PassengerId'], 'Survived': best_pred})
submission_df.to_csv("submission.csv", index=False)
train.to_csv("submission_train.csv", index=False)
test.to_csv("submission_test.csv", index=False)
