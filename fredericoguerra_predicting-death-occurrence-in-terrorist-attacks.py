import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#import visual as vs
GTD = pd.read_csv('../input/globalterrorismdb_0718dist.csv', engine = 'python')
GTD.shape
GTD.head(5)
print("{}% of incidents occured at least one death.".format(round(float(GTD[GTD.nkill > 0]['nkill'].count())/GTD.nkill.notna().sum()*100,2)))
plt.subplots(figsize=(18,6))
ax = sns.barplot(x="iyear", y="nkill", data=GTD, estimator = sum, palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7)).set_title('Number of deaths by year')
plt.xticks(rotation=90)
print("GTD length before: %.2d " % len(GTD))
print("Avoiding rows with missing values of deaths...")
GTD = GTD[GTD['nkill'].notnull()]
GTD['death'] = np.where(GTD.nkill > 0, 1, 0)
GTD = GTD.drop(['nkill'], axis = 1)
print("GTD length after: %.2d " % len(GTD))
plt.subplots(figsize=(18,8))
ax = sns.countplot(x='region_txt', hue='death', data=GTD)
ax.set_title('Count of death and not death occurrence by region')
plt.xticks(rotation=90)
plt.subplots(figsize=(18,8))
ax = sns.countplot(x='attacktype1_txt', hue='death', data=GTD)
ax.set_title('Count of death and not death occurrence by Attack type')
plt.xticks(rotation=90)
columns = GTD.columns
percent_missing = GTD.isnull().sum() * 100 / len(GTD)
unique = GTD.nunique()
dtypes = GTD.dtypes
missing_value_data = pd.DataFrame({'column_name': columns,
                                 'percent_missing': percent_missing,
                                 'unique': unique,
                                 'types': dtypes})
missing_value_data = missing_value_data[missing_value_data['percent_missing']>0]
missing_value_data=missing_value_data.sort_values(by=['percent_missing'], ascending=False)
plt.subplots(figsize=(20,10))
ax = sns.barplot(x="column_name", y="percent_missing", hue='types', data=missing_value_data)
ax.axhline(50, ls='--', color = 'r')
ax.text(90,51,"50% of missing values", color = 'r')
ax.set_title("Percentage of Missing Values by column")
plt.xticks(rotation=90)
def missing_values(data,mis_min):
    columns = data.columns
    percent_missing = data.isnull().sum() * 100 / len(data)
    unique = data.nunique()
    missing_value_data = pd.DataFrame({'column_name': columns,
                                 'percent_missing': percent_missing,
                                 'unique': unique})
    missing_drop = list(missing_value_data[missing_value_data.percent_missing>mis_min].column_name)
    return(missing_drop)
print("Number of features before dropping columns with >50%% of NAN: %.1d" % GTD.shape[1])
GTD['natlty1'].fillna(GTD['country'], inplace = True)

missing_drop = missing_values(GTD,50)
GTD = GTD.drop(missing_drop, axis=1)
GTD = GTD.drop(columns = ['nkillter'])
print("Number of features after dropping columns with >50%% of NAN: %.1d" % GTD.shape[1])
mode_fill = ['nwound','longitude','latitude','weapsubtype1','weapsubtype1_txt','targsubtype1','targsubtype1_txt','natlty1_txt','guncertain1','ishostkid', 'specificity','doubtterr','multiple', 'target1', 'city', 'provstate']
for col in mode_fill:
    GTD[col].fillna(GTD[col].mode()[0], inplace=True)

GTD['nperps'].fillna(GTD['nperps'].median(), inplace=True)
GTD['nperpcap'].fillna(GTD['nperpcap'].median(), inplace=True)
GTD['nwoundte'].fillna(GTD['nwoundte'].median(), inplace=True)
GTD['nwoundus'].fillna(GTD['nwoundus'].median(), inplace=True)
GTD['nkillus'].fillna(GTD['nkillus'].median(), inplace=True)
GTD = GTD.drop(columns = ['weapdetail', 'scite1', 'summary', 'corp1'])
GTD['claimed'].fillna(0, inplace=True)
print((GTD.isnull().sum(axis=0)/len(GTD)*100).sort_values(ascending=False).head(5))
print("Missing values successfully treated!!!")
GTD['suicide'] = GTD.suicide.astype('object')
num_features = ['nperps','nkillus','nwound','nwoundus', 'nwoundte']
matplotlib.style.use('seaborn')
GTD[num_features].hist(figsize=(16,10))
from scipy.stats import skew
import sys

skewed_feat = GTD[num_features].apply(lambda x: skew(x))
skewed_feat = skewed_feat[skewed_feat > 0.75]
skewed_feat = skewed_feat.index

GTD[skewed_feat] = np.log1p(GTD[skewed_feat])
GTD[skewed_feat].hist(figsize=(16,10))
duplicated_columns = [col for col in GTD.columns if "_txt" in col]
print("There are %.2d duplicated columns to be removed"%(len(duplicated_columns)))
GTD = GTD.drop(duplicated_columns, axis=1)
cat_features = GTD.dtypes[GTD.dtypes == 'object'].index

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

for col in cat_features:
    GTD[col] = le.fit_transform(GTD[col])
corr_matrix = GTD.corr()
abs(corr_matrix['death']).sort_values(ascending=False).head(10)

mask = np.zeros_like(corr_matrix, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True
plt.subplots(figsize=(25,10))
cmap = sns.diverging_palette(0,240, as_cmap = True)
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmin = -.3, vmax=.3, center=0, square=True, linewidths=.3, cbar_kws={"shrink": .5})

GTD[GTD == np.inf] = np.nan
GTD.fillna(GTD.mean(), inplace = True)
from sklearn.model_selection import train_test_split

income = GTD['death']

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(GTD.drop(['death'], axis=1),
                                                    income,
                                                    shuffle=True,
                                                    test_size = 0.2,
                                                    random_state = 43)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier

DTC = DecisionTreeClassifier(random_state = 43)

DTC = DTC.fit(X_train, y_train)
bench_acc = accuracy_score(y_test, DTC.predict(X_test))
bench_fsc = fbeta_score(y_test, DTC.predict(X_test), 0.5)

print('-'*40)
print("Benchmark Model:")
print("Accuracy: %.2f" %round(bench_acc*100,2))
print("F_score: %.2f" %round(bench_fsc*100,2))
print('-'*40)
import time

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    
    results = {}

    start = time.time()
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time.time()
    
    # Training time
    results['train_time'] = end - start
        
    # TODO: Get the predictions on the test set(X_test),
    #       then get predictions on the first 300 training samples(X_train) using .predict()
    start = time.time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time.time() # Get end time
    
    # TODO: Calculate the total prediction time
    results['pred_time'] = end - start
            
    # TODO: Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
        
    # TODO: Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    # TODO: Compute F-score on the the first 300 training samples using fbeta_score()
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta = 0.5, average = 'weighted')
        
    # TODO: Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test, predictions_test,  beta = 0.5, average = 'weighted')
       
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    # Return the results
    return results
# TODO: Import the three supervised learning models from sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import fbeta_score
from sklearn.svm import SVC

# TODO: Initialize the three models
clf_A = KNeighborsClassifier()
clf_B = AdaBoostClassifier(random_state = 41)
clf_C = RandomForestClassifier(random_state = 41)
#clf_D = GaussianNB()

# TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data
# HINT: samples_100 is the entire training set i.e. len(y_train)
    # HINT: samples_10 is 10% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
# HINT: samples_1 is 1% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
samples_100 = len(y_train)
samples_10 = int(samples_100/10)
samples_1 = int(samples_100/100)


# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_test, y_test)
accuracy = bench_acc
fbeta = bench_fsc
# Run metrics visualization for the three supervised learning models chosen
#vs.evaluate(results, accuracy, fbeta)
for k, v in results.items():
    print('-'*40)
    print("Model: %s" %(k))
    for i in v:
        print("Accuraccy: %.4f \n Fscore_test: %.4f" %(v[i]['acc_test'], v[i]['f_test']))
from sklearn.feature_selection import SelectFromModel

feat_labels = GTD.columns
sfm = SelectFromModel(clf, threshold=0.02)
sfm.fit(X_train, y_train)
important_feat = [feat_labels[col] for col in sfm.get_support(indices=True)]
#for feature_list_index in sfm.get_support(indices=True):
#    print(feat_labels[feature_list_index])
important_feat.append('death')

X_important_test = sfm.transform(X_test)
X_important_train = sfm.transform(X_train)

print('-'*40)
print('Feature Selection:')
print('before')
print(X_train.shape)
print('after')
print(X_important_train.shape)
print('-'*40)
rf = RandomForestClassifier(random_state = 41)
rf.fit(X_important_train, y_train)
important_pred = rf.predict(X_important_test)

print("Accuracy after Feature Selection: %.2f" %round((accuracy_score(y_test, important_pred)*100),2))
print("F_score after Feature Selection: %.2f" %round((fbeta_score(y_test, important_pred, 0.5)*100),2))
X_intermediate, X_test, y_intermediate, y_test = train_test_split(GTD.drop(['death'], axis=1),
                                                                  income,
                                                                  shuffle=True,
                                                                  test_size = 0.2,
                                                                  random_state = 43)

X_train, X_validation, y_train, y_validation = train_test_split(X_intermediate,
                                                                y_intermediate,
                                                                shuffle=False,
                                                                test_size=0.2,
                                                                random_state=43)
# delete intermediate variables
del X_intermediate, y_intermediate

# print proportions
print('train: {}% | validation: {}% | test {}%'.format(round(float(len(y_train))/len(income),2),
                                                       round(float(len(y_validation))/len(income),2),
                                                       round(float(len(y_test))/len(income),2)))
rf.fit(X_train, y_train)

print('-'*40)
print("Train:")
print("Accuracy: %.2f" %round((accuracy_score(y_train, rf.predict(X_train))*100),2))
print("F_score: %.2f" %round((fbeta_score(y_train, rf.predict(X_train), 0.5)*100),2))
print('-'*40)
print("Validation:")
print("Accuracy: %.2f" %round((accuracy_score(y_validation, rf.predict(X_validation))*100),2))
print("F_score: %.2f" %round((fbeta_score(y_validation, rf.predict(X_validation), 0.5)*100),2))
print('-'*40)
print("Test:")
print("Accuracy: %.2f" %round((accuracy_score(y_test, rf.predict(X_test))*100),2))
print("F_score: %.2f" %round((fbeta_score(y_test, rf.predict(X_test), 0.5)*100),2))
print('-'*40)
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 2, stop = 20, num = 5)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(1, 11, num = 5)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
"""rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model
rf_random.fit(X_train, y_train)
best_random = rf_random.best_estimator_
best_random"""
best_random = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=4, min_samples_split=5,
            min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=1,
            oob_score=False, random_state=41, verbose=0, warm_start=False)
new_rf = best_random
new_rf.fit(X_train, y_train)

print('-'*40)
print("Test:")
print("Accuracy: %.2f" %round((accuracy_score(y_test, new_rf.predict(X_test))*100),2))
print("F_score: %.2f" %round((fbeta_score(y_test, new_rf.predict(X_test), 0.5)*100),2))
print('-'*40)
from sklearn import metrics

conf_matrix = metrics.confusion_matrix(y_test, new_rf.predict(X_test))
TP = conf_matrix[1, 1]
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
conf_mat_df = pd.DataFrame(conf_matrix).astype(int)
conf_mat_df = conf_mat_df.rename(index={0:"no_death_true"})
conf_mat_df = conf_mat_df.rename(index={1:"death_true"})
conf_mat_df = conf_mat_df.rename(columns={0:"no_death_pred"})
conf_mat_df = conf_mat_df.rename(columns={1:"death_pred"})
print("Given death occurrance in attacks, the model mistakenly predicted no deathes in {}% cases.".format(round(float(FN)/(FN+TP)*100,2)))
grid_kws = {"height_ratios": (.8, .05), "hspace": .3}
ax = sns.heatmap(conf_mat_df, annot = True, fmt='d', cmap="YlGnBu",  cbar_kws={"orientation": "horizontal"})
ax.set_title("Confusion Matrix")
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
plt.style.use(['seaborn-whitegrid','seaborn-poster'])
y_pred_proba = new_rf.predict_proba(X_test)[::,1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr,color='g', label = "ROC curve (AUC = "+str(round(auc,2))+')')
plt.plot([0,1],[0,1], linestyle = '--', lw=2, color ='r', label='Chance', alpha=.8)
plt.legend(loc=4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve - Perfomance of the final classification model")
plt.show()
# Feature Importance

X = GTD.drop(['death'], axis=1)
y = GTD['death']

importances = new_rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in new_rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

feat_importances = [(feature, round(importance, 3)) for feature, importance in zip(GTD.columns, importances)]
feat_importances = sorted(feat_importances, key = lambda x: x[1], reverse = True)
feat_importances

#for f in range(X.shape[1]):
#    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation = 90)
plt.xlim([-1, X.shape[1]])
plt.show()
# display the relative importance of each attribute
#print(model.feature_importances_)