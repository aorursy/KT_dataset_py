# Importing liberaries for data preprocessing, visualization, modeling and scoring.

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split

from sklearn.linear_model import LogisticRegressionCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

import sklearn.ensemble as ens

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

import xgboost as xgb

import lightgbm as lgb

import sklearn.feature_selection

import sklearn.metrics

from sklearn.preprocessing import MaxAbsScaler, StandardScaler, Normalizer, LabelEncoder, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')





%matplotlib inline

sns.set_style('white')
labelled = pd.read_csv('../input/train.csv') # Labelled Data for training, validation, and model assessment. 
unlabelled = pd.read_csv('../input/test.csv') # Unlabelled Data for final submission.
# Keep PassengerId for final submission in seperate variable.

passengerID = unlabelled[['PassengerId']]
data = pd.concat([labelled, unlabelled], axis= 0, sort= False)
data.head()
sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap= 'viridis')
data.info()
sns.countplot(data = data, x= 'Survived')
sns.countplot(data = data, x= 'Survived', hue= 'Sex')

plt.legend(loc =(1.1,0.9)),
sns.countplot(data = data, x='Survived', hue='Pclass')
sns.distplot(data['Age'].dropna(), kde = False, bins = 35)
sns.countplot(x = 'SibSp', data = data)
sns.countplot(data= data.dropna(), x='Pclass')
sns.countplot(data= data, x='Pclass', hue= 'Sex')
sns.boxplot(data= data.dropna(), x='Pclass', y= 'Fare')
data.describe()
class_mean_age = data.pivot_table(values='Age', index='Pclass', aggfunc='median')
null_age = data['Age'].isnull()
data.loc[null_age,'Age'] = data.loc[null_age,'Pclass'].apply(lambda x: class_mean_age.loc[x] )
data.Age.isnull().sum()
class_mean_fare = data.pivot_table(values= 'Fare', index= 'Pclass', aggfunc='median')
null_fare = data['Fare'].isnull()
data.loc[null_fare, 'Fare'] = data.loc[null_fare, 'Pclass'].apply(lambda x: class_mean_fare.loc[x] )
data.Fare.isnull().sum()
data.Embarked.value_counts()
data['Embarked'] = data.Embarked.fillna('S')
data.Embarked.isnull().sum()
data['Title'] = data.Name.apply(lambda x : x[x.find(',')+2:x.find('.')])
data.Title.value_counts()
rare_titles = (data['Title'].value_counts() < 10)
data['Title'] = data['Title'].apply(lambda x : 'Other' if rare_titles.loc[x] == True else x)
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = 0
data['IsAlone'].loc[ data['FamilySize'] == 1] = 1
data['AgeBins'] = 0
data['AgeBins'].loc[(data['Age'] >= 11) & (data['Age'] < 20)] = 1

data['AgeBins'].loc[(data['Age'] >= 20) & (data['Age'] < 60)] = 2

data['AgeBins'].loc[data['Age'] >= 60] = 3
data['FareBins'] = pd.qcut(data['Fare'], 4)
data.columns
data.drop(columns=['PassengerId','Name','Ticket', 'Cabin', 'Age', 'Fare', 'SibSp', 'Parch'], inplace= True)
data = pd.get_dummies(

    data, columns=['Embarked', 'Sex', 'Title'], drop_first=True)
label = LabelEncoder()

data['FareBins'] = label.fit_transform(data['FareBins'])
data.head(7)
labelled = data[data.Survived.isnull() == False].reset_index(drop=True)

unlabelled = data[data.Survived.isnull()].drop(columns = ['Survived']).reset_index(drop=True)
labelled['Survived'] = labelled.Survived.astype('int64')
scalers = [MinMaxScaler(), MaxAbsScaler(), StandardScaler(), RobustScaler(),

            Normalizer(), QuantileTransformer(), PowerTransformer()]
scaler_score = {}

labelled_copy = labelled.copy(deep= True) # Creat a copy of the original Labelled DF.

for scaler in scalers:

    scaler.fit(labelled_copy[['FamilySize']])

    labelled_copy['FamilySize'] = scaler.transform(labelled_copy[['FamilySize']])

    lr = LogisticRegressionCV(cv = 10, scoring= 'accuracy')

    lr.fit(labelled_copy.drop(columns=['Survived']), labelled_copy.Survived)

    score = lr.score(labelled_copy.drop(columns=['Survived']), labelled_copy.Survived)

    scaler_score.update({scaler:score})
scaler_score
scaler = StandardScaler()

scaler.fit(labelled[['FamilySize']])

labelled['FamilySize'] = scaler.transform(labelled[['FamilySize']])

unlabelled['FamilySize'] = scaler.transform(unlabelled[['FamilySize']])
x_train, x_other, y_train, y_other = train_test_split(

                labelled.drop(columns=['Survived']), labelled.Survived, train_size=0.7)
x_valid, x_test, y_valid, y_test = train_test_split(

                                    x_other, y_other, train_size=0.5)
features = labelled.drop(columns=['Survived'])

target = labelled.Survived
logistic_reg = LogisticRegressionCV(cv= 7)
threshold = np.arange(1, 10, 0.5) *1e-1
scores = []

for i in threshold:

    selector = sklearn.feature_selection.VarianceThreshold(threshold= i)

    selected_features = selector.fit_transform(features)

    logistic_reg.fit(selected_features, target)

    y_pred = logistic_reg.predict(features.loc[:, selector.get_support()])

    scores.append(sklearn.metrics.accuracy_score(target, y_pred))



# Plot variance threshold VS. cross-validated scores for training sets.

plt.figure()

plt.xlabel("variance threshold")

plt.ylabel("Cross validated accuracy score")

plt.plot(np.arange(1, 10, 0.5) *1e-1, np.array(scores))
print('The highest accuracy score is:', np.max(np.array(scores)))
number_of_features = list(range(1,13))
scores_k = []

for i in number_of_features:

    selector = sklearn.feature_selection.SelectKBest(k=i)

    selected_features = selector.fit_transform(features, target)

    logistic_reg.fit(selected_features, target)

    y_pred = logistic_reg.predict(features.loc[:, selector.get_support()])

    scores_k.append(sklearn.metrics.accuracy_score(target, y_pred))



# Plot number of selected features VS. cross-validated scores for training sets.

plt.figure()

plt.xlabel("Number of Selected Features")

plt.ylabel("Cross validated accuracy score")    

plt.plot(list(range(1,13)), scores_k)
print("Maximum accuracy score is :", max(scores_k))
print("Optimal number of features :", np.argmax(np.array(scores_k)) + 1)
selector = sklearn.feature_selection.RFECV(logistic_reg, step= 1, cv= 5)

selector.fit(features, target)



# Plot number of features VS. cross-validation scores

plt.figure()

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
print("Optimal number of features : %d" % selector.n_features_)
print("Maximum accuracy score is :", np.max(selector.grid_scores_))
threshold = np.arange(1, 5, 0.1) *1e-1
scores_sfm = []

for i in threshold:

    selector = sklearn.feature_selection.SelectFromModel(logistic_reg, threshold= i)

    selector.fit(features, target)

    selected_features = features.loc[:, selector.get_support()]

    logistic_reg.fit(selected_features, target)

    y_pred = logistic_reg.predict(selected_features)

    scores_sfm.append(sklearn.metrics.accuracy_score(target, y_pred))



# Plot number of features VS. cross-validation scores

plt.figure()

plt.xlabel("Threshold Value")

plt.ylabel("Cross validation score")    

plt.plot(np.arange(1, 5, 0.1) *1e-1, scores_sfm)
print("Maximum accuracy score is :", np.max(np.array(scores_sfm)))
print("Optimal threshold :", threshold[np.argmax(np.array(scores_sfm))])
# Fit the model with features selected by SelectFromModel method and the training set

selector = sklearn.feature_selection.SelectFromModel(logistic_reg, threshold= 0.25)

selector.fit(features, target)

lr_selected_features = selector.get_support()
logistic_reg = LogisticRegressionCV(

    Cs=1, cv= 7, scoring='accuracy', max_iter=1000, refit=True)
lr_parameters_1 = {'solver': ['liblinear', 'saga'], 'penalty': ['l1']}

lr_parameters_2 = {'solver': ['newton-cg', 'lbfgs', 'sag'], 'penalty': ['l2']}
rs_lr = RandomizedSearchCV(logistic_reg, param_distributions= lr_parameters_2, n_iter= 100)
rs_lr.fit(x_train.loc[:, lr_selected_features], y_train)
print('Best Parameters are:\n', rs_lr.best_params_,

      '\nTraining accuracy score is:\n', rs_lr.best_score_)
print('Validation accuracy score is:\n', rs_lr.score(

    x_valid.loc[:, lr_selected_features], y_valid))
param_name = 'Cs'

param_range = [1, 10, 100, 1000]

train_score, valid_score = [], []

for cs in param_range:

    lr = LogisticRegressionCV(Cs=cs, cv=7, scoring='accuracy', solver= 'newton-cg',

                              penalty= 'l2', refit=True, max_iter=1000)

    lr.fit(x_train.loc[:, lr_selected_features], y_train)

    train_score.append(

        lr.score(x_train.loc[:, lr_selected_features], y_train))

    valid_score.append(

        lr.score(x_valid.loc[:, lr_selected_features], y_valid))
# Plot Regularization factor VS. cross-validated scores for training and Validation sets.

plt.figure()

plt.xlabel("Regularization factor")

plt.ylabel("Cross validated accuracy score")

plt.plot([1, 10, 100, 1000], train_score, color = 'blue')

plt.plot([1, 10, 100, 1000], valid_score, color = 'red')
train_test_diff = np.array(train_score) - np.array(valid_score)



# Plot number of folds VS. difference of cross-validated scores between train and Dev sets.

plt.figure()

plt.xlabel("Regularization Factor")

plt.ylabel("Diff. Cross validated accuracy score")

plt.plot([1, 10, 100, 1000], train_test_diff)
lr = LogisticRegressionCV(Cs= 10, cv= 7, solver= 'newton-cg', penalty= 'l2')

lr.fit(x_train.loc[:, lr_selected_features], y_train)
# Finding the ROC curve for different threshold values.

# probability estimates of the positive class.

y_scores_lr = lr.predict_proba(x_test.loc[:, lr_selected_features])[:, 1]

lr_fpr, lr_tpr, lr_thresholds = sklearn.metrics.roc_curve(y_test, y_scores_lr)
# Finding the AUC for the logistic classification model.

lr_auc = sklearn.metrics.auc(x=lr_fpr, y=lr_tpr)
# Model accuracy score on test data.

lr_acc = lr.score(x_test.loc[:, lr_selected_features], y_test)
print('For logistic Regression: \n Area Under Curve: {}, \n Test Accuracy score: {}'.format(

    lr_auc, lr_acc))
nb = GaussianNB()
threshold = np.arange(1, 10, 0.5) *1e-1
scores = []

for i in threshold:

    selector = sklearn.feature_selection.VarianceThreshold(threshold= i)

    selected_features = selector.fit_transform(features)

    nb.fit(selected_features, target)

    y_pred = nb.predict(features.loc[:, selector.get_support()])

    scores.append(sklearn.metrics.accuracy_score(target, y_pred))



# Plot variance threshold VS. cross-validated scores for training sets.

plt.figure()

plt.xlabel("variance threshold")

plt.ylabel("Cross validated accuracy score")

plt.plot(np.arange(1, 10, 0.5) *1e-1, np.array(scores))
print('The highest accuracy score is:', np.max(np.array(scores)))
number_of_features = list(range(1,13))
scores_k = []

for i in number_of_features:

    selector = sklearn.feature_selection.SelectKBest(k=i)

    selected_features = selector.fit_transform(features, target)

    nb.fit(selected_features, target)

    y_pred = nb.predict(features.loc[:, selector.get_support()])

    scores_k.append(sklearn.metrics.accuracy_score(target, y_pred))



# Plot number of selected features VS. cross-validated scores for training sets.

plt.figure()

plt.xlabel("Number of Selected Features")

plt.ylabel("Cross validated accuracy score")    

plt.plot(list(range(1,13)), scores_k)
print("Maximum accuracy score is :", max(scores_k))
print("Optimal number of features :", np.argmax(np.array(scores_k)) + 1)
# Fit the model with features selected by Variance threshold method and the training set

selector = sklearn.feature_selection.VarianceThreshold(threshold= 0.1)

selector.fit(features, target)

nb_selected_features = selector.get_support()
nb_params = {'priors': [[0.7, 0.3], [0.6, 0.4],

                        [0.5, 0.5], [0.4, 0.6], [0.3, 0.7]]}
rs_nb = RandomizedSearchCV(nb, param_distributions= nb_params,cv= 7 ,n_iter= 200)
rs_nb.fit(x_train.loc[:, nb_selected_features], y_train)
print('Best Parameters are:\n', rs_nb.best_params_,

      '\nTraining accuracy score is:\n', rs_nb.best_score_)
print('Validation accuracy score is:\n', rs_nb.score(

    x_valid.loc[:, nb_selected_features], y_valid))
nb = GaussianNB(priors= [0.4, 0.6])

nb.fit(x_train.loc[:, nb_selected_features], y_train)
# Finding the ROC curve for different threshold values.

# probability estimates of the positive class.

y_scores_nb = nb.predict_proba(x_test.loc[:, nb_selected_features])[:, 1]

nb_fpr, nb_tpr, nb_thresholds = sklearn.metrics.roc_curve(y_test, y_scores_nb)
# Finding the AUC for the naive bayes classification model.

nb_auc = sklearn.metrics.auc(x=nb_fpr, y=nb_tpr)
# Model Accuracy score on test data

nb_acc = nb.score(x_test.loc[:, nb_selected_features], y_test)
print('For Gaussian Naive Bayes: \n Area Under Curve: {}, \n Test Accuracy score: {}'.format(

    nb_auc, nb_acc))
knn = KNeighborsClassifier(n_neighbors= 5)
threshold = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
scores = []

for i in threshold:

    selector = sklearn.feature_selection.VarianceThreshold(threshold= i)

    selected_features = selector.fit_transform(features)

    knn.fit(selected_features, target)

    y_pred = knn.predict(selected_features)

    scores.append(sklearn.metrics.accuracy_score(target, y_pred))



# Plot variance threshold VS. cross-validated scores for training sets.

plt.figure()

plt.xlabel("variance threshold")

plt.ylabel("Cross validated accuracy score")

plt.plot([0.001, 0.005, 0.01, 0.05, 0.1, 0.2], np.array(scores))
np.max(np.array(scores))
number_of_features = list(range(1,13))
scores_k = []

for i in number_of_features:

    selector = sklearn.feature_selection.SelectKBest(k=i)

    selected_features = selector.fit_transform(features, target)

    knn.fit(selected_features, target)

    y_pred = knn.predict(selected_features)

    scores_k.append(sklearn.metrics.accuracy_score(target, y_pred))



# Plot number of selected features VS. cross-validated scores for training sets.

plt.figure()

plt.xlabel("Number of Selected Features")

plt.ylabel("Cross validated accuracy score")    

plt.plot(list(range(1,13)), scores_k)
print("Maximum accuracy score is :", max(scores_k))
print("Optimal number of features :", np.argmax(np.array(scores_k)) + 1)
selector = sklearn.feature_selection.VarianceThreshold(threshold= 0.1)

selector.fit(features, target)

knn_selected_features = selector.get_support()
knn_params = {'n_neighbors': [5, 7, 9] , 'weights': [

    'uniform', 'distance'], 'leaf_size': [5, 10, 20], 'p': [1, 2, 3]}
rs_knn = RandomizedSearchCV(knn, param_distributions= knn_params,

                      scoring='accuracy', cv= 7, n_iter= 200, refit=True)
rs_knn.fit(x_train.loc[:, knn_selected_features], y_train)
print('Best Parameters are:\n', rs_knn.best_params_,

      '\nTraining accuracy score is:\n', rs_knn.best_score_)
print('Validation accuracy score is:\n', rs_knn.score(

    x_valid.loc[:, knn_selected_features], y_valid))
param_name = 'n_neighbors'

param_range = np.arange(3,21)

train_score, valid_score = [], []

for k in param_range:

    knn = KNeighborsClassifier(n_neighbors= k, weights= 'uniform', p= 2,leaf_size= 5)

    knn.fit(x_train.loc[:, knn_selected_features], y_train)

    train_score.append(

        knn.score(x_train.loc[:, knn_selected_features], y_train))

    valid_score.append(

        knn.score(x_valid.loc[:, knn_selected_features], y_valid))
# Plot number of neighbours VS. cross-validated scores for training and Validation sets.

plt.figure()

plt.xlabel("Number of Neighbours")

plt.ylabel("Cross validated accuracy score")

plt.plot(np.arange(3,21), train_score, color = 'blue')

plt.plot(np.arange(3,21), valid_score, color = 'red')
train_test_diff = np.array(train_score) - np.array(valid_score)



# Plot Number of Neighbours VS. difference of cross-validated scores between train and validation sets.

plt.figure()

plt.xlabel("Number of Neighbours")

plt.ylabel("Diff. Cross validated accuracy score")

plt.plot(np.arange(3,21), train_test_diff)
knn = KNeighborsClassifier(n_neighbors= 4, weights= 'uniform', p= 2, leaf_size= 5)
knn.fit(x_train.loc[:, knn_selected_features], y_train)
# Finding the ROC curve for different threshold values.

# probability estimates of the positive class.

y_scores_knn = knn.predict_proba(x_test.loc[:, knn_selected_features])[:, 1]

knn_fpr, knn_tpr, knn_thresholds = sklearn.metrics.roc_curve(y_test, y_scores_knn)
# Finding the AUC for the naive bayes classification model.

knn_auc = sklearn.metrics.auc(x=knn_fpr, y=knn_tpr)
# Model Accuracy score on test data

knn_acc = knn.score(x_test.loc[:, knn_selected_features], y_test)
print('Area Under Curve: {}, Accuracy: {}'.format(knn_auc, knn_acc))
svm = SVC(probability=True)
threshold = np.arange(1, 10, 0.5) *1e-1
scores = []

for i in threshold:

    selector = sklearn.feature_selection.VarianceThreshold(threshold= i)

    selected_features = selector.fit_transform(features)

    svm.fit(selected_features, target)

    y_pred = svm.predict(features.loc[:, selector.get_support()])

    scores.append(sklearn.metrics.accuracy_score(target, y_pred))



# Plot variance threshold VS. cross-validated scores for training sets.

plt.figure()

plt.xlabel("variance threshold")

plt.ylabel("Cross validated accuracy score")

plt.plot(np.arange(1, 10, 0.5) *1e-1, np.array(scores))
print('The highest accuracy score is:', np.max(np.array(scores)))
number_of_features = list(range(1,13))
scores_k = []

for i in number_of_features:

    selector = sklearn.feature_selection.SelectKBest(k=i)

    selected_features = selector.fit_transform(features, target)

    svm.fit(selected_features, target)

    y_pred = svm.predict(features.loc[:, selector.get_support()])

    scores_k.append(sklearn.metrics.accuracy_score(target, y_pred))



# Plot number of selected features VS. cross-validated scores for training sets.

plt.figure()

plt.xlabel("Number of Selected Features")

plt.ylabel("Cross validated accuracy score")    

plt.plot(list(range(1,13)), scores_k)
print("Maximum accuracy score is :", max(scores_k))
print("Optimal number of features :", np.argmax(np.array(scores_k)) + 1)
# Fit the model with features selected by SelectFromModel method and the training set

selector = sklearn.feature_selection.VarianceThreshold(threshold= 0.1)

selector.fit(features, target)

svm_selected_features = selector.get_support()
svm = SVC(probability=True)
svm_parameters = {'kernel': ['linear', 'rbf', 'sigmoid'], 'gamma': [

    'auto', 'scale'], 'shrinking': [True, False]}
rs_svm = RandomizedSearchCV(svm, cv= 7, param_distributions= svm_parameters, n_iter= 200)
rs_svm.fit(x_train.loc[:, svm_selected_features], y_train)
print('Best Parameters are:\n', rs_svm.best_params_,

      '\nTraining accuracy score is:\n', rs_svm.best_score_)
print('Validation accuracy score is:\n', rs_svm.score(

    x_valid.loc[:, svm_selected_features], y_valid))
param_name = 'C'

param_range = np.arange(1,31)

train_score, valid_score = [], []

for c in param_range:

    svm = SVC(C= c,probability= True)

    svm.fit(x_train.loc[:, svm_selected_features], y_train)

    train_score.append(

        svm.score(x_train.loc[:, svm_selected_features], y_train))

    valid_score.append(

        svm.score(x_valid.loc[:, svm_selected_features], y_valid))
# Plot Regularization factor VS. cross-validated scores for training and Validation sets.

plt.figure()

plt.xlabel("Regularization factor")

plt.ylabel("Cross validated accuracy score")

plt.plot(np.arange(1,31), train_score, color = 'blue')

plt.plot(np.arange(1,31), valid_score, color = 'red')
train_test_diff = np.array(train_score) - np.array(valid_score)



# Plot number of folds VS. difference of cross-validated scores between train and Dev sets.

plt.figure()

plt.xlabel("Regularization Factor")

plt.ylabel("Diff. Cross validated accuracy score")

plt.plot(np.arange(1,31), train_test_diff)
svm = SVC(C=3, probability= True)

svm.fit(features.loc[:, svm_selected_features], target)
# Finding the ROC curve for different threshold values.

# probability estimates of the positive class.

y_scores_svm = svm.predict_proba(x_test.loc[:, svm_selected_features])[:, 1]

svm_fpr, svm_tpr, svm_thresholds = sklearn.metrics.roc_curve(y_test, y_scores_svm)
# Finding the AUC for the logistic classification model.

svm_auc = sklearn.metrics.auc(x=svm_fpr, y=svm_tpr)
# Model accuracy score on test data.

svm_acc = svm.score(x_test.loc[:, svm_selected_features], y_test)
print('For logistic Regression: \n Area Under Curve: {}, \n Test Accuracy score: {}'.format(

    svm_auc, svm_acc))
dt = DecisionTreeClassifier()
threshold = np.arange(1, 10, 0.5) *1e-1
scores = []

for i in threshold:

    selector = sklearn.feature_selection.VarianceThreshold(threshold= i)

    selected_features = selector.fit_transform(features)

    dt.fit(selected_features, target)

    y_pred = dt.predict(features.loc[:, selector.get_support()])

    scores.append(sklearn.metrics.accuracy_score(target, y_pred))



# Plot variance threshold VS. cross-validated scores for training sets.

plt.figure()

plt.xlabel("variance threshold")

plt.ylabel("Cross validated accuracy score")

plt.plot(threshold, np.array(scores))
print('The highest accuracy score is:', np.max(np.array(scores)))
number_of_features = list(range(1,13))
scores_k = []

for i in number_of_features:

    selector = sklearn.feature_selection.SelectKBest(k=i)

    selected_features = selector.fit_transform(features, target)

    dt.fit(selected_features, target)

    y_pred = dt.predict(features.loc[:, selector.get_support()])

    scores_k.append(sklearn.metrics.accuracy_score(target, y_pred))



# Plot number of selected features VS. cross-validated scores for training sets.

plt.figure()

plt.xlabel("Number of Selected Features")

plt.ylabel("Cross validated accuracy score")    

plt.plot(number_of_features, scores_k)
print("Maximum accuracy score is :", max(scores_k))
print("Optimal number of features :", np.argmax(np.array(scores_k)) + 1)
selector = sklearn.feature_selection.RFECV(dt, step= 1, cv= 5)

selector.fit(features, target)



# Plot number of features VS. cross-validation scores

plt.figure()

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
print("Optimal number of features : %d" % selector.n_features_)
print("Maximum accuracy score is :", np.max(selector.grid_scores_))
threshold = [0.001, 0.0025, 0.005, 0.01, 0.025 ,0.05, 0.1]
scores_sfm = []

for i in threshold:

    selector = sklearn.feature_selection.SelectFromModel(dt, threshold= i)

    selector.fit(features, target)

    selected_features = features.loc[:, selector.get_support()]

    dt.fit(selected_features, target)

    y_pred = dt.predict(selected_features)

    scores_sfm.append(sklearn.metrics.accuracy_score(target, y_pred))



# Plot number of features VS. cross-validation scores

plt.figure()

plt.xlabel("Threshold Value")

plt.ylabel("Cross validation score")    

plt.plot(threshold, scores_sfm)
print("Maximum accuracy score is :", np.max(np.array(scores_sfm)))
print("Optimal threshold :", threshold[np.argmax(np.array(scores_sfm))])
dt_params = {'criterion': ['gini'], 'min_samples_split': [

     21, 22, 23], 'max_features': ['auto', 'log2', None]}
rs_dt = RandomizedSearchCV(dt, param_distributions= dt_params,

                     scoring='accuracy', cv= StratifiedKFold(7), refit=True, n_iter= 500)
rs_dt.fit(x_train, y_train)
print('Best Parameters are:\n', rs_dt.best_params_,

      '\nTraining accuracy score is:\n', rs_dt.best_score_)
print('Validation accuracy score is:\n', rs_dt.score(x_valid, y_valid))
param_name = 'max_depth'

param_range = np.arange(1, 21)

train_score, valid_score = [], []

for depth in param_range:

    dt = DecisionTreeClassifier(

        criterion='gini', max_features=None, min_samples_split=22, max_depth= depth)

    dt.fit(x_train, y_train)

    train_score.append(dt.score(x_train, y_train))

    valid_score.append(dt.score(x_valid, y_valid))
# Plot Regularization factor VS. cross-validated scores for training and Validation sets.

plt.figure()

plt.xlabel("Regularization factor")

plt.ylabel("Cross validated accuracy score")

plt.plot(param_range, train_score, color = 'blue')

plt.plot(param_range, valid_score, color = 'red')
train_test_diff = np.array(train_score) - np.array(valid_score)



# Plot number of folds VS. difference of cross-validated scores between train and Dev sets.

plt.figure()

plt.xlabel("Regularization Factor")

plt.ylabel("Diff. Cross validated accuracy score")

plt.plot(param_range, train_test_diff)
dt = DecisionTreeClassifier(criterion='gini', max_features=None, min_samples_split=22, max_depth= 3)

dt.fit(features,target)
# Finding the ROC curve for different threshold values.

# probability estimates of the positive class.

y_scores_dt = dt.predict_proba(x_test)[:, 1]

dt_fpr, dt_tpr, dt_thresholds = sklearn.metrics.roc_curve(y_test, y_scores_dt)

# Finding the AUC for the Decision Tree classification model.

dt_auc = sklearn.metrics.auc(x=dt_fpr, y=dt_tpr)
dt_acc = dt.score(x_test, y_test)
print('Area Under Curve: {}, Accuracy: {}'.format(dt_auc, dt_acc))
rf = ens.RandomForestClassifier()
threshold = np.arange(1, 10, 0.5) *1e-1
scores = []

for i in threshold:

    selector = sklearn.feature_selection.VarianceThreshold(threshold= i)

    selected_features = selector.fit_transform(features)

    rf.fit(selected_features, target)

    y_pred = rf.predict(features.loc[:, selector.get_support()])

    scores.append(sklearn.metrics.accuracy_score(target, y_pred))



# Plot variance threshold VS. cross-validated scores for training sets.

plt.figure()

plt.xlabel("variance threshold")

plt.ylabel("Cross validated accuracy score")

plt.plot(threshold, np.array(scores))
print('The highest accuracy score is obtained after execluding features whose variance is less than: ', 

              np.round(threshold[np.argmax(np.array(scores))],3))
print('The highest accuracy score is:', np.max(np.array(scores)))
number_of_features = list(range(1,13))
scores_k = []

for i in number_of_features:

    selector = sklearn.feature_selection.SelectKBest(k=i)

    selected_features = selector.fit_transform(features, target)

    rf.fit(selected_features, target)

    y_pred = rf.predict(features.loc[:, selector.get_support()])

    scores_k.append(sklearn.metrics.accuracy_score(target, y_pred))



# Plot number of selected features VS. cross-validated scores for training sets.

plt.figure()

plt.xlabel("Number of Selected Features")

plt.ylabel("Cross validated accuracy score")    

plt.plot(number_of_features, scores_k)
print("Maximum accuracy score is :", max(scores_k))
print("Optimal number of features :", np.argmax(np.array(scores_k)) + 1)
selector = sklearn.feature_selection.RFECV(rf, step= 1, cv= 5)

selector.fit(features, target)



# Plot number of features VS. cross-validation scores

plt.figure()

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
print("Optimal number of features : %d" % selector.n_features_)
print("Maximum accuracy score is :", np.max(selector.grid_scores_))
threshold = [0.001, 0.0025, 0.005, 0.01, 0.025 ,0.05, 0.1, 0.15]
scores_sfm = []

for i in threshold:

    selector = sklearn.feature_selection.SelectFromModel(rf, threshold= i)

    selector.fit(features, target)

    selected_features = features.loc[:, selector.get_support()]

    rf.fit(selected_features, target)

    y_pred = rf.predict(selected_features)

    scores_sfm.append(sklearn.metrics.accuracy_score(target, y_pred))



# Plot number of features VS. cross-validation scores

plt.figure()

plt.xlabel("Threshold Value")

plt.ylabel("Cross validation score")    

plt.plot(threshold, scores_sfm)
print("Maximum accuracy score is :", np.max(np.array(scores_sfm)))
print("Optimal threshold :", threshold[np.argmax(np.array(scores_sfm))])
rf_params = {'n_estimators': [200, 300, 400], 'criterion': ['gini'], 'min_samples_split': [

    22, 20, 25], 'max_features': ['auto', 'log2', None], 'class_weight': [{0: 0.6, 1: 0.4}, {0: 0.6, 1: 0.4}, {0: 0.5, 1: 0.5}]}
rs_rf = RandomizedSearchCV(rf, param_distributions= rf_params,

                     scoring='accuracy', cv= StratifiedKFold(7), refit=True, n_iter= 200)
rs_rf.fit(x_train, y_train)
print('Best Parameters are:\n', rs_rf.best_params_,

      '\nTraining accuracy score is:\n', rs_rf.best_score_)
print('Validation accuracy score is:\n', rs_rf.score(x_valid, y_valid))
param_name = 'max_depth'

param_range = np.arange(1, 31)

train_score, valid_score = [], []

for depth in param_range:

    rf = ens.RandomForestClassifier(n_estimators= 300,

        criterion='gini', max_features= 'auto', min_samples_split=22, 

                                    class_weight= {0: 0.5, 1: 0.5},max_depth= depth)

    rf.fit(x_train, y_train)

    train_score.append(rf.score(x_train, y_train))

    valid_score.append(rf.score(x_valid, y_valid))
# Plot Regularization factor VS. cross-validated scores for training and Validation sets.

plt.figure()

plt.xlabel("Regularization factor")

plt.ylabel("Cross validated accuracy score")

plt.plot(param_range, train_score, color = 'blue')

plt.plot(param_range, valid_score, color = 'red')
train_test_diff = np.array(train_score) - np.array(valid_score)



# Plot number of folds VS. difference of cross-validated scores between train and Dev sets.

plt.figure()

plt.xlabel("Regularization Factor")

plt.ylabel("Diff. Cross validated accuracy score")

plt.plot(param_range, train_test_diff)
rf = ens.RandomForestClassifier(n_estimators= 300,

        criterion='gini', max_features= 'auto', min_samples_split=22, 

                                class_weight= {0: 0.5, 1: 0.5}, max_depth= 6)

rf.fit(features,target)
# Finding the ROC curve for different threshold values.

# probability estimates of the positive class.

y_scores_rf = rf.predict_proba(x_test)[:, 1]

rf_fpr, rf_tpr, rf_thresholds = sklearn.metrics.roc_curve(y_test, y_scores_rf)

# Finding the AUC for the Decision Tree classification model.

rf_auc = sklearn.metrics.auc(x=rf_fpr, y=rf_tpr)
rf_acc = rf.score(x_test, y_test)
print('Area Under Curve: {}, Accuracy: {}'.format(rf_auc, rf_acc))
bg = ens.BaggingClassifier()
threshold = np.arange(1, 10, 0.5) *1e-1
scores = []

for i in threshold:

    selector = sklearn.feature_selection.VarianceThreshold(threshold= i)

    selected_features = selector.fit_transform(features)

    bg.fit(selected_features, target)

    y_pred = bg.predict(features.loc[:, selector.get_support()])

    scores.append(sklearn.metrics.accuracy_score(target, y_pred))



# Plot variance threshold VS. cross-validated scores for training sets.

plt.figure()

plt.xlabel("variance threshold")

plt.ylabel("Cross validated accuracy score")

plt.plot(threshold, np.array(scores))
print('The highest accuracy score is obtained after execluding features whose variance is less than: ', 

              np.round(threshold[np.argmax(np.array(scores))],3))
print('The highest accuracy score is:', np.max(np.array(scores)))
number_of_features = list(range(1,13))
scores_k = []

for i in number_of_features:

    selector = sklearn.feature_selection.SelectKBest(k=i)

    selected_features = selector.fit_transform(features, target)

    bg.fit(selected_features, target)

    y_pred = bg.predict(features.loc[:, selector.get_support()])

    scores_k.append(sklearn.metrics.accuracy_score(target, y_pred))



# Plot number of selected features VS. cross-validated scores for training sets.

plt.figure()

plt.xlabel("Number of Selected Features")

plt.ylabel("Cross validated accuracy score")    

plt.plot(number_of_features, scores_k)
print("Maximum accuracy score is :", max(scores_k))
print("Optimal number of features :", np.argmax(np.array(scores_k)) + 1)
bg_params = {'n_estimators': [20, 25, 100], 'base_estimator': [

    None, svm], 'max_features': [0.6, 0.7, 0.8], 'oob_score' : [True, False], 

            'max_samples': [0.6,0.7,0.8]}
rs_bg = RandomizedSearchCV(bg, param_distributions= bg_params,

                     scoring='accuracy', cv=StratifiedKFold(7), n_iter= 2000,refit=True)
rs_bg.fit(x_train, y_train)
print('Best Parameters are:\n', rs_bg.best_params_,

      '\nTraining accuracy score is:\n', rs_bg.best_score_)
print('Validation accuracy score is:\n', rs_bg.score(x_valid, y_valid))
bg = ens.BaggingClassifier(n_estimators= 25,

        max_features= 0.8, base_estimator= svm, oob_score= True, max_samples= 0.8)

bg.fit(features,target)
# Finding the ROC curve for different threshold values.

# probability estimates of the positive class.

y_scores_bg = bg.predict_proba(x_test)[:, 1]

bg_fpr, bg_tpr, bg_thresholds = sklearn.metrics.roc_curve(y_test, y_scores_bg)

# Finding the AUC for the Decision Tree classification model.

bg_auc = sklearn.metrics.auc(x=bg_fpr, y=bg_tpr)
bg_acc = bg.score(x_test, y_test)
print('Area Under Curve: {}, Accuracy: {}'.format(bg_auc, bg_acc))
ada = ens.AdaBoostClassifier()
threshold = np.arange(1, 10, 0.5) *1e-1
scores = []

for i in threshold:

    selector = sklearn.feature_selection.VarianceThreshold(threshold= i)

    selected_features = selector.fit_transform(features)

    ada.fit(selected_features, target)

    y_pred = ada.predict(features.loc[:, selector.get_support()])

    scores.append(sklearn.metrics.accuracy_score(target, y_pred))



# Plot variance threshold VS. cross-validated scores for training sets.

plt.figure()

plt.xlabel("variance threshold")

plt.ylabel("Cross validated accuracy score")

plt.plot(threshold, np.array(scores))
print('The highest accuracy score is obtained after execluding features whose variance is less than: ', 

              np.round(threshold[np.argmax(np.array(scores))],3))
print('The highest accuracy score is:', np.max(np.array(scores)))
number_of_features = list(range(1,13))
scores_k = []

for i in number_of_features:

    selector = sklearn.feature_selection.SelectKBest(k=i)

    selected_features = selector.fit_transform(features, target)

    ada.fit(selected_features, target)

    y_pred = ada.predict(features.loc[:, selector.get_support()])

    scores_k.append(sklearn.metrics.accuracy_score(target, y_pred))



# Plot number of selected features VS. cross-validated scores for training sets.

plt.figure()

plt.xlabel("Number of Selected Features")

plt.ylabel("Cross validated accuracy score")    

plt.plot(number_of_features, scores_k)
print("Maximum accuracy score is :", max(scores_k))
print("Optimal number of features :", np.argmax(np.array(scores_k)) + 1)
selector = sklearn.feature_selection.RFECV(ada, step= 1, cv= 5)

selector.fit(features, target)



# Plot number of features VS. cross-validation scores

plt.figure()

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
print("Optimal number of features : %d" % selector.n_features_)
print("Maximum accuracy score is :", np.max(selector.grid_scores_))
threshold = [0.001, 0.0025, 0.005, 0.01, 0.025 ,0.05, 0.1, 0.15]
scores_sfm = []

for i in threshold:

    selector = sklearn.feature_selection.SelectFromModel(ada, threshold= i)

    selector.fit(features, target)

    selected_features = features.loc[:, selector.get_support()]

    ada.fit(selected_features, target)

    y_pred = ada.predict(selected_features)

    scores_sfm.append(sklearn.metrics.accuracy_score(target, y_pred))



# Plot number of features VS. cross-validation scores

plt.figure()

plt.xlabel("Threshold Value")

plt.ylabel("Cross validation score")    

plt.plot(threshold, scores_sfm)
print("Maximum accuracy score is :", np.max(np.array(scores_sfm)))
print("Optimal threshold :", threshold[np.argmax(np.array(scores_sfm))])
ada_params = {'n_estimators': [90, 100, 110], 'base_estimator': [None, svm],

             'learning_rate': [0.09 ,0.1, 0.11]}
rs_ada = RandomizedSearchCV(ada, param_distributions= ada_params,

                     scoring='accuracy', cv= StratifiedKFold(7), refit=True, n_iter= 500)
rs_ada.fit(x_train, y_train)
print('Best Parameters are:\n', rs_ada.best_params_,

      '\nTraining accuracy score is:\n', rs_ada.best_score_)
print('Validation accuracy score is:\n', rs_ada.score(x_valid, y_valid))
ada = ens.AdaBoostClassifier(n_estimators= 110, learning_rate= 0.09)

ada.fit(features, target)
# Finding the ROC curve for different threshold values.

# probability estimates of the positive class.

y_scores_ada = ada.predict_proba(x_test)[:, 1]

ada_fpr, ada_tpr, ada_thresholds = sklearn.metrics.roc_curve(y_test, y_scores_ada)

# Finding the AUC for the Decision Tree classification model.

ada_auc = sklearn.metrics.auc(x=ada_fpr, y=ada_tpr)
ada_acc = ada.score(x_test, y_test)
print('Area Under Curve: {}, Accuracy: {}'.format(ada_auc, ada_acc))
gb = ens.GradientBoostingClassifier()
threshold = np.arange(1, 10, 0.5) *1e-1
scores = []

for i in threshold:

    selector = sklearn.feature_selection.VarianceThreshold(threshold= i)

    selected_features = selector.fit_transform(features)

    gb.fit(selected_features, target)

    y_pred = gb.predict(features.loc[:, selector.get_support()])

    scores.append(sklearn.metrics.accuracy_score(target, y_pred))



# Plot variance threshold VS. cross-validated scores for training sets.

plt.figure()

plt.xlabel("variance threshold")

plt.ylabel("Cross validated accuracy score")

plt.plot(threshold, np.array(scores))
print('The highest accuracy score is obtained after execluding features whose variance is less than: ', 

              np.round(threshold[np.argmax(np.array(scores))],3))
print('The highest accuracy score is:', np.max(np.array(scores)))
number_of_features = list(range(1,13))
scores_k = []

for i in number_of_features:

    selector = sklearn.feature_selection.SelectKBest(k=i)

    selected_features = selector.fit_transform(features, target)

    gb.fit(selected_features, target)

    y_pred = gb.predict(features.loc[:, selector.get_support()])

    scores_k.append(sklearn.metrics.accuracy_score(target, y_pred))



# Plot number of selected features VS. cross-validated scores for training sets.

plt.figure()

plt.xlabel("Number of Selected Features")

plt.ylabel("Cross validated accuracy score")    

plt.plot(number_of_features, scores_k)
print("Maximum accuracy score is :", max(scores_k))
print("Optimal number of features :", np.argmax(np.array(scores_k)) + 1)
selector = sklearn.feature_selection.RFECV(gb, step= 1, cv= 5)

selector.fit(features, target)



# Plot number of features VS. cross-validation scores

plt.figure()

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
print("Optimal number of features : %d" % selector.n_features_)
print("Maximum accuracy score is :", np.max(selector.grid_scores_))
threshold = [0.001, 0.0025, 0.005, 0.01, 0.025 ,0.05, 0.1, 0.15]
scores_sfm = []

for i in threshold:

    selector = sklearn.feature_selection.SelectFromModel(gb, threshold= i)

    selector.fit(features, target)

    selected_features = features.loc[:, selector.get_support()]

    gb.fit(selected_features, target)

    y_pred = gb.predict(selected_features)

    scores_sfm.append(sklearn.metrics.accuracy_score(target, y_pred))



# Plot number of features VS. cross-validation scores

plt.figure()

plt.xlabel("Threshold Value")

plt.ylabel("Cross validation score")    

plt.plot(threshold, scores_sfm)
print("Maximum accuracy score is :", np.max(np.array(scores_sfm)))
print("Optimal threshold :", threshold[np.argmax(np.array(scores_sfm))])
# Fit the model with features selected by SelectFromModel method and the training set

selector = sklearn.feature_selection.SelectKBest(k= 11)

selector.fit(features, target)

gb_selected_features = selector.get_support()
gb_params = {'n_estimators': [150, 160, 170], 'loss': ['deviance', 'exponential'],

             'subsample': [0.7, 0.8, 0.9], 'max_features': ['auto', 'log2', None]}
rs_gb = RandomizedSearchCV(gb, param_distributions= gb_params,

                     scoring='accuracy', cv= StratifiedKFold(7), refit=True, n_iter= 2000)
rs_gb.fit(x_train.loc[:,gb_selected_features], y_train)
print('Best Parameters are:\n', rs_gb.best_params_,

      '\nTraining accuracy score is:\n', rs_gb.best_score_)
print('Validation accuracy score is:\n',

      rs_gb.score(x_valid.loc[:,gb_selected_features], y_valid))
param_name = 'max_depth'

param_range = np.arange(1, 31)

train_score, valid_score = [], []

for depth in param_range:

    gb = ens.GradientBoostingClassifier(n_estimators= 170,

        subsample= 0.9, max_features= 'auto', loss= 'exponential',max_depth= depth)

    gb.fit(x_train.loc[:,gb_selected_features], y_train)

    train_score.append(gb.score(x_train.loc[:,gb_selected_features], y_train))

    valid_score.append(gb.score(x_valid.loc[:,gb_selected_features], y_valid))
# Plot Regularization factor VS. cross-validated scores for training and Validation sets.

plt.figure()

plt.xlabel("Maximum Depth")

plt.ylabel("Cross validated accuracy score")

plt.plot(param_range, train_score, color = 'blue')

plt.plot(param_range, valid_score, color = 'red')
train_test_diff = np.abs(np.array(train_score) - np.array(valid_score))



# Plot number of folds VS. difference of cross-validated scores between train and Dev sets.

plt.figure()

plt.xlabel("Maximum depth")

plt.ylabel("Diff. Cross validated accuracy score")

plt.plot(param_range, train_test_diff)
gb = ens.GradientBoostingClassifier(n_estimators= 170, subsample= 0.9, max_features= 'auto',

                                    loss= 'exponential',max_depth= 4)

gb.fit(features.loc[:, gb_selected_features],target)
# Finding the ROC curve for different threshold values.

# probability estimates of the positive class.

y_scores_gb = gb.predict_proba(x_test.loc[:, gb_selected_features])[:, 1]

gb_fpr, gb_tpr, gb_thresholds = sklearn.metrics.roc_curve(y_test, y_scores_gb)

# Finding the AUC for the Decision Tree classification model.

gb_auc = sklearn.metrics.auc(x=gb_fpr, y=gb_tpr)
gb_acc = gb.score(x_test.loc[:, gb_selected_features], y_test)
print('Area Under Curve: {}, Accuracy: {}'.format(gb_auc, gb_acc))
xgboost = xgb.XGBClassifier()
threshold = np.arange(1, 10, 0.5) *1e-2
scores = []

for i in threshold:

    selector = sklearn.feature_selection.VarianceThreshold(threshold= i)

    selected_features = selector.fit_transform(features)

    xgboost.fit(selected_features, target)

    y_pred = xgboost.predict(selected_features)

    scores.append(sklearn.metrics.accuracy_score(target, y_pred))



# Plot variance threshold VS. cross-validated scores for training sets.

plt.figure()

plt.xlabel("variance threshold")

plt.ylabel("Cross validated accuracy score")

plt.plot(threshold, np.array(scores))
print('The highest accuracy score is obtained after execluding features whose variance is less than: ', 

              np.round(threshold[np.argmax(np.array(scores))],3))
print('The highest accuracy score is:', np.max(np.array(scores)))
number_of_features = list(range(1,13))
scores_k = []

for i in number_of_features:

    selector = sklearn.feature_selection.SelectKBest(k=i)

    selected_features = selector.fit_transform(features, target)

    xgboost.fit(selected_features, target)

    y_pred = xgboost.predict(selected_features)

    scores_k.append(sklearn.metrics.accuracy_score(target, y_pred))



# Plot number of selected features VS. cross-validated scores for training sets.

plt.figure()

plt.xlabel("Number of Selected Features")

plt.ylabel("Cross validated accuracy score")    

plt.plot(number_of_features, scores_k)
print("Maximum accuracy score is :", max(scores_k))
print("Optimal number of features :", np.argmax(np.array(scores_k)) + 1)
selector = sklearn.feature_selection.RFECV(xgboost, step= 1, cv= 5)

selector.fit(features, target)



# Plot number of features VS. cross-validation scores

plt.figure()

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
print("Optimal number of features : %d" % selector.n_features_)
print("Maximum accuracy score is :", np.max(selector.grid_scores_))
threshold = [0.001, 0.0025, 0.005, 0.01, 0.025 ,0.05, 0.1, 0.15]
scores_sfm = []

for i in threshold:

    selector = sklearn.feature_selection.SelectFromModel(xgboost, threshold= i)

    selector.fit(features, target)

    selected_features = features.loc[:, selector.get_support()]

    xgboost.fit(selected_features, target)

    y_pred = xgboost.predict(selected_features)

    scores_sfm.append(sklearn.metrics.accuracy_score(target, y_pred))



# Plot number of features VS. cross-validation scores

plt.figure()

plt.xlabel("Threshold Value")

plt.ylabel("Cross validation score")    

plt.plot(threshold, scores_sfm)
print("Maximum accuracy score is :", np.max(np.array(scores_sfm)))
print("Optimal threshold :", threshold[np.argmax(np.array(scores_sfm))])
xgboost = xgb.XGBClassifier()

xgboost.fit(features, target)
# Finding the ROC curve for different threshold values.

# probability estimates of the positive class.

y_scores_xgb = xgboost.predict_proba(x_test)[:, 1]

xgb_fpr, xgb_tpr, xgb_thresholds = sklearn.metrics.roc_curve(y_test, y_scores_xgb)

# Finding the AUC for the Decision Tree classification model.

xgb_auc = sklearn.metrics.auc(x=xgb_fpr, y=xgb_tpr)
xgb_acc = xgboost.score(x_test, y_test)
print('Area Under Curve: {}, Accuracy: {}'.format(xgb_auc, xgb_acc))
lgboost = lgb.LGBMClassifier()
threshold = [0.001, 0.01,0.1,0.5]
scores = []

for i in threshold:

    selector = sklearn.feature_selection.VarianceThreshold(threshold= i)

    selected_features = selector.fit_transform(features)

    lgboost.fit(selected_features, target)

    y_pred = lgboost.predict(selected_features)

    scores.append(sklearn.metrics.accuracy_score(target, y_pred))

plt.plot([0.001, 0.01,0.1,0.5], np.array(scores))
np.max(np.array(scores))
number_of_features = list(range(1,13))
scores_k = []

for i in number_of_features:

    selector = sklearn.feature_selection.SelectKBest(k=i)

    selected_features = selector.fit_transform(features, target)

    lgboost.fit(selected_features, target)

    y_pred = lgboost.predict(selected_features)

    scores_k.append(sklearn.metrics.accuracy_score(target, y_pred))

plt.plot(list(range(1,13)), scores_k)
max(scores_k)
print("Optimal number of features :", np.argmax(np.array(scores_k)) + 1)
selector = sklearn.feature_selection.RFECV(lgboost, step= 1, cv= 7)

selector.fit(features, target)



# Plot number of features VS. cross-validation scores

plt.figure()

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
print("Optimal number of features : %d" % selector.n_features_)
np.max(selector.grid_scores_)
threshold = [0.001, 0.01, 0.05, 0.1 , 0.5]
scores_sfm = []

for i in threshold:

    selector = sklearn.feature_selection.SelectFromModel(lgboost, threshold= i)

    selector.fit(features, target)

    selected_features = features.loc[:, selector.get_support()]

    lgboost.fit(selected_features, target)

    y_pred = lgboost.predict(selected_features)

    scores_sfm.append(sklearn.metrics.accuracy_score(target, y_pred))



# Plot number of features VS. cross-validation scores

plt.figure()

plt.xlabel("Threshold Value")

plt.ylabel("Cross validation score")    

plt.plot([0.001, 0.01, 0.05, 0.1 , 0.5], scores_sfm)
print("Maximum accuracy score is :", np.max(np.array(scores_sfm)))
print("Optimal threshold :", threshold[np.argmax(np.array(scores_sfm))])
# Fit the model with the best 11 features selected.

selector = sklearn.feature_selection.SelectKBest(k= 11)

selector.fit(features, target)

lgb_selected_features = selector.get_support()
lgboost = lgb.LGBMClassifier()

lgboost.fit(features.loc[:,lgb_selected_features], target)
# Finding the ROC curve for different threshold values.

# probability estimates of the positive class.

y_scores_lgb = lgboost.predict_proba(x_test.loc[:,lgb_selected_features])[:, 1]

lgb_fpr, lgb_tpr, lgb_thresholds = sklearn.metrics.roc_curve(y_test, y_scores_lgb)

# Finding the AUC for the Decision Tree classification model.

lgb_auc = sklearn.metrics.auc(x=lgb_fpr, y=lgb_tpr)
lgb_acc = lgboost.score(x_test.loc[:,lgb_selected_features], y_test)
print('Area Under Curve: {}, Accuracy: {}'.format(lgb_auc, lgb_acc))
v = ens.VotingClassifier(estimators=[

    ('lr', lr),('NB', nb),('KNN', knn),('SVM', svm),('DT', dt),

    ('RF', rf), ('BG', bg),('AdaBoost', ada),('GBM', gb),

    ('XGBM', xgboost),('LightGBM', lgboost)], 

                         voting='soft', 

                         weights= [1,1,1, 1.25, 1.25, 1.25, 1.25, 1.25, 1.75, 1.5, 1.5])
# Fit the model with the best 11 features selected.

selector = sklearn.feature_selection.SelectKBest(k= 11)

selector.fit(features, target)

voting_selected_features = selector.get_support()
v.fit(features.loc[:, voting_selected_features], target)
# Finding the ROC curve for different threshold values.

# probability estimates of the positive class.

y_scores_v = v.predict_proba(features.loc[:, voting_selected_features])[:, 1]

v_fpr, v_tpr, v_thresholds = sklearn.metrics.roc_curve(target, y_scores_v)

# Finding the AUC for the Voting classification model.

v_auc = sklearn.metrics.auc(x=v_fpr, y=v_tpr)
v_acc = v.score(x_test.loc[:,voting_selected_features], y_test)
print('Area Under Curve: {}, Accuracy: {}'.format(v_auc, v_acc))
pd.DataFrame([(lr_auc, lr_acc), (nb_auc, nb_acc), (knn_auc, knn_acc), (dt_auc, dt_acc),

              (rf_auc, rf_acc), (svm_auc, svm_acc), (bg_auc, bg_acc), (ada_auc, ada_acc),

              (v_auc, v_acc), (gb_auc, gb_acc), (xgb_auc, xgb_acc), (lgb_auc, lgb_acc)],

             columns=['AUC', 'Accuracy'],

             index=['Logistic Regression', 'Naive Bayes', 'KNN', 'Decision Tree',

                    'Random Forest', 'SVM', 'Bagging', 'AdaBoost', 'Voting',

                   'Gradient Boost', 'XGBoost', 'Light Boost'])
plt.figure(figsize=(8, 5))

plt.title('Receiver Operating Characteristic Curve')

plt.plot(lr_fpr, lr_tpr, label='LR_AUC = %0.2f' % lr_auc)

plt.plot(nb_fpr, nb_tpr, label='NB_AUC = %0.2f' % nb_auc)

plt.plot(knn_fpr, knn_tpr, label='KNN_AUC = %0.2f' % knn_auc)

plt.plot(svm_fpr, svm_tpr, label='SVM_AUC = %0.2f' % svm_auc)

plt.plot(dt_fpr, dt_tpr, label='DT_AUC = %0.2f' % dt_auc)

plt.plot(rf_fpr, rf_tpr, label='RF_AUC = %0.2f' % rf_auc)

plt.plot(bg_fpr, bg_tpr, label='BG_AUC = %0.2f' % bg_auc)

plt.plot(ada_fpr, ada_tpr, label='Ada_AUC = %0.2f' % ada_auc)

plt.plot(v_fpr, v_tpr, label='Voting_AUC = %0.2f' % v_auc)

plt.plot(lgb_fpr, lgb_tpr, label='LBoost_AUC = %0.2f' % lgb_auc)

plt.legend(loc='lower right')

plt.plot([0, 1], [0, 1], 'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.title('ROC Curve')

plt.show()
y_pred_v = pd.DataFrame(v.predict(unlabelled.loc[:, voting_selected_features]), columns=[

                        'Survived'], dtype='int64')
v_model = pd.concat([passengerID, y_pred_v], axis=1)
v_model.to_csv('voting.csv', index= False)