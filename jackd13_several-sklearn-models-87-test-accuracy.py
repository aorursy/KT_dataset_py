%matplotlib inline
import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import itertools
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import LabelBinarizer, StandardScaler

from sklearn.pipeline import FeatureUnion, Pipeline

from sklearn.svm import LinearSVC, SVC
dataset = pd.read_csv("../input/adult.csv")

print(dataset.shape)

print()

print(dataset.columns.values)
dataset.hist(bins=50, figsize=(20,15))
def createScatterSubPlot(x_feature, y_feature):

    fig, ax = plt.subplots(figsize=(20, 10))

    

    for name, group in dataset.groupby('income'):

        ax.plot(group[x_feature], group[y_feature], marker='o', linestyle='', ms=5, label=name, alpha=0.1)

    ax.legend()

    ax.set_xlabel(x_feature)

    ax.set_ylabel(y_feature)



createScatterSubPlot("age", "hours.per.week")

plt.title('Age vs Hours Worked per Week')

createScatterSubPlot("capital.loss", "capital.gain")

plt.title('Capital Loss vs Capital Gain')

createScatterSubPlot("education.num", "age")

plt.title('Years in Education vs Age')



plt.show()
incomeCategories = dict()

for key in ["<=50K", ">50K"]:

    incomeCategories[key] = int(dataset[dataset.income == key].shape[0])

 

print(incomeCategories)



print()



fig, ax = plt.subplots(figsize=(20, 10))

ax.pie(list(incomeCategories.values()), labels=incomeCategories.keys())

ax.legend()

ax.axis('equal') 

plt.title('Proportions of each income group in the dataset')

plt.show()
def countGroupOfGroups(topGroupName, subGroupName):

    """Divide the dataset using two of its attributes, return a dictionary containing the nested counts of samples

    that fall into each group"""

    

    topCategories = dict()



    for topKey in dataset[topGroupName].unique():

        subcategories = dict()

        subgroup = dataset[dataset[topGroupName] == topKey]

        

        for subKey in subgroup[subGroupName].unique():

            subcategories[str(subKey)] = int(subgroup[subgroup[subGroupName] == subKey].shape[0])



        topCategories[str(topKey)] = subcategories

    

    return topCategories



# Plot how income is divided by gender

genderCategories = countGroupOfGroups("income", "sex")

print(genderCategories)



ind = np.arange(len(genderCategories["<=50K"])) 

barWidth = 0.35



plt.subplots(figsize=(20, 10))

p1 = plt.bar(ind, genderCategories[">50K"].values(), barWidth)

p2 = plt.bar(ind, genderCategories["<=50K"].values(), barWidth, bottom=genderCategories[">50K"].values())



plt.ylabel('Number of individuals')

plt.title('Proportions of each gender in each income group')

plt.xticks(ind, genderCategories["<=50K"].keys())

plt.legend((p1[0], p2[0]), [">50K", "<=50K"])

plt.show()
# Plot how income is divided by race

raceCategories = countGroupOfGroups("income", "race")

print(raceCategories)



ind = np.arange(len(raceCategories[">50K"])) 

barWidth = 0.35



plt.subplots(figsize=(20, 10))

p1 = plt.bar(ind, raceCategories[">50K"].values(), barWidth)

p2 = plt.bar(ind, raceCategories["<=50K"].values(), barWidth, bottom=raceCategories[">50K"].values())



plt.ylabel('Number of individuals')

plt.title('Proportions of each race in each income group')

plt.xticks(ind, raceCategories[">50K"].keys())

plt.legend((p1[0], p2[0]), [">50K", "<=50K"])

plt.show()



# Plot the same thing again, but trim the y axis so that some areas of the graph are more visible

plt.subplots(figsize=(20, 10))

p1 = plt.bar(ind, raceCategories[">50K"].values(), barWidth)

p2 = plt.bar(ind, raceCategories["<=50K"].values(), barWidth, bottom=raceCategories[">50K"].values())



plt.ylabel('Number of individuals')

plt.title('Proportions of each race in each income group (trimmed)')

plt.xticks(ind, raceCategories[">50K"].keys())

plt.legend((p1[0], p2[0]), [">50K", "<=50K"])

plt.ylim([0, 5000])

plt.show()
X = dataset.drop("income", axis=1)

y = dataset["income"]



y_binarizer = LabelBinarizer()

y = y_binarizer.fit_transform(y).ravel()



print("X:\n", X.head(5))

print()

print("y:\n", y[0:5])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
NUMERICAL_FEATURES = list()

CATEGORICAL_FEATURES = list()



for featureNumber in range(X.values.shape[1]):

    if type(dataset.values[0, featureNumber]) is str:

        CATEGORICAL_FEATURES.append(dataset.columns.values[featureNumber])

    else:

        NUMERICAL_FEATURES.append(dataset.columns.values[featureNumber])



print("Numerical features:", NUMERICAL_FEATURES)

print()

print("Categorical features:", CATEGORICAL_FEATURES)



class FeatureSeparator(BaseEstimator, TransformerMixin): 

    """Class will drop features that do not match the provided names."""

    _featureNames = list()

    

    def __init__(self, featureNames):

        self._featureNames = featureNames



    def fit(self, X, y=None):

        return self



    def transform(self, X):

        return X[self._featureNames].values



    

class MultiFeatureBinarizer(BaseEstimator, TransformerMixin):

    """Perform the binarization for each feature, returning a large matrix of all features."""

    _binarizers = list()

    

    def  __init__(self):

        return None

    

    def fit(self, X, y=None):

        """Builds and fits a list of LabelBinarizers, one for each feature"""

        NUM_FEATURES = X.shape[1]

        

        for featureNumber in range(NUM_FEATURES):

            binarizer = LabelBinarizer()

            binarizer.fit(X[:, featureNumber])

            self._binarizers.append(binarizer)

        

        return self

            

        

    def transform(self, X):

        """Performs the binarization and returns a matrix of all the new binary features."""

        retVal = np.empty((0, 0))

        

        for featureNumber in range(len(self._binarizers)):

            if (featureNumber == 0):

                retVal = self._binarizers[featureNumber].transform(X[:, featureNumber])

            else:

                retVal = np.concatenate((retVal, self._binarizers[featureNumber].transform(X[:, featureNumber])),

                                        axis=1)



        return retVal



    



numPipeline = Pipeline([ ('selector', FeatureSeparator(NUMERICAL_FEATURES)),

                         ('std_scaler', StandardScaler()) ])



catPipeline = Pipeline([ ('selector', FeatureSeparator(CATEGORICAL_FEATURES)),

                         ('multi_binarizer', MultiFeatureBinarizer()) ])



prepPipeline = FeatureUnion(transformer_list=[

    ('num_pipeline', numPipeline),

    ('cat_pipeline', catPipeline)

])



X_train_prepared = prepPipeline.fit_transform(X_train)



print("First row of our prepared dataset:\n", X_train_prepared[0, :])

print("Samples, Features:", X_train_prepared.shape)
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)

rnd_clf.fit(X_train_prepared, y_train)



featureImportances = dict()

for name, score in zip(dataset.columns.values, rnd_clf.feature_importances_):

    featureImportances[name] = score

    

importancesSortedKeys = sorted(featureImportances,

                               key=featureImportances.get,

                               reverse=True)



print("Feature importance:")

for key in importancesSortedKeys:

    print(key, featureImportances[key])
class Model:

    classifier = None

    gridSearch = None

    

    def __init__(self, classifier, paramGrid):

        self.classifier = classifier

    

        self.gridSearch = GridSearchCV(self.classifier,

                                      paramGrid,

                                      cv=5,

                                      scoring="accuracy",

                                      verbose=2,

                                      n_jobs=-1)

        



        

        



# Create a dictionary of classifiers and grid search parameters

classifiers = dict()

classifiers["knn"] = Model(KNeighborsClassifier(),

                           [ {'n_jobs': [-1], 'n_neighbors': [2, 4, 6]} ])



classifiers["randomForest"] = Model(RandomForestClassifier(),

                                    [ {'n_jobs': [-1], 'n_estimators': [10, 100, 500]},

                                      {'n_jobs': [-1], 'n_estimators': [100], 'max_depth': [2, 20, 100]} ])



classifiers["sgd"] = Model(SGDClassifier(),

                           [ {"loss": ["hinge", "log"], 'penalty': ["l2", "l1"]} ])



classifiers["linearSVC"] = Model(LinearSVC(),

                                 [ {"C": [0.1, 1, 10]} ])



classifiers["rbfSVC"] = Model(SVC(),

                              [ {"C": [0.1, 1, 10], "kernel": ["rbf"]},

                                {"C": [0.1, 1, 10], "kernel": ["poly"], "degree": [2, 3]}])



classifiers["adaboost"] = Model(AdaBoostClassifier(),

                                [ {"n_estimators": [10, 50, 150], "learning_rate": [0.1, 1]}])
classifiers["knn"].gridSearch.fit(X_train_prepared, y_train)
classifiers["randomForest"].gridSearch.fit(X_train_prepared, y_train)
classifiers["sgd"].gridSearch.fit(X_train_prepared, y_train)
classifiers["linearSVC"].gridSearch.fit(X_train_prepared, y_train)
classifiers["rbfSVC"].gridSearch.fit(X_train_prepared, y_train)
classifiers["adaboost"].gridSearch.fit(X_train_prepared, y_train)
for key in classifiers.keys():

    print("\nFor classifier:", key)

    print(classifiers[key].gridSearch.best_estimator_)

    print("Accuracy:", classifiers[key].gridSearch.best_score_)
classifiers["randomForest"] = Model(RandomForestClassifier(),

                                    [ {'n_jobs': [-1], 'n_estimators': [90, 100, 110], 'max_depth': [15, 20, 25]} ])



classifiers["rbfSVC"] = Model(SVC(),

                              [ {"C": [9, 10, 11], "kernel": ["rbf"]} ])



classifiers["adaboost"] = Model(AdaBoostClassifier(),

                                [ {"n_estimators": [150, 200, 300]} ])



classifiers["randomForest"].gridSearch.fit(X_train_prepared, y_train)

classifiers["rbfSVC"].gridSearch.fit(X_train_prepared, y_train)

classifiers["adaboost"].gridSearch.fit(X_train_prepared, y_train)
for key in list(["randomForest", "rbfSVC", "adaboost"]): 

    print("\nFor classifier:", key)

    print(classifiers[key].gridSearch.best_estimator_)

    print("Accuracy:", classifiers[key].gridSearch.best_score_)
y_pred_forest = classifiers["randomForest"].gridSearch.predict(X_train_prepared)

y_pred_rbfSVC = classifiers["rbfSVC"].gridSearch.predict(X_train_prepared)

y_pred_adaboost = classifiers["adaboost"].gridSearch.predict(X_train_prepared)
def plotConfusionMatrix(confusionMatrix, title, classNames):

    """Normalises and plots a grid of the confusion matrix (mostly borrowed from an sklearn tutorial)"""

    

    # Scale it, plot the coloured grid

    confusionMatrix = confusionMatrix.astype('float') / confusionMatrix.sum(axis=1)[:, np.newaxis]

    plt.imshow(confusionMatrix, interpolation='nearest', cmap=plt.cm.Purples)

    

    

    # Labels

    plt.title(title)

    tickMarks = np.arange(len(classNames))

    plt.xticks(tickMarks, classNames, rotation=45)

    plt.yticks(tickMarks, classNames)

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    

    # Plot the numbers

    for i, j in itertools.product(range(confusionMatrix.shape[0]), range(confusionMatrix.shape[1])):

        plt.text(j, i, format(confusionMatrix[i, j], ".2f"),

                 horizontalalignment="center",

                 color="white")



    



plotConfusionMatrix(confusion_matrix(y_train, y_pred_forest), "Random Forest", ["<=50K", ">50K"])

plt.show()



plotConfusionMatrix(confusion_matrix(y_train, y_pred_rbfSVC), "RBF SVC", ["<=50K", ">50K"])

plt.show()



plotConfusionMatrix(confusion_matrix(y_train, y_pred_adaboost), "Adaboost", ["<=50K", ">50K"])

plt.show()
classifiers["adaboost"] = Model(AdaBoostClassifier(),

                                [ {"n_estimators": [1450, 1500, 1700]}])

classifiers["adaboost"].gridSearch.fit(X_train_prepared, y_train)



print(classifiers["adaboost"].gridSearch.best_estimator_)

print("Accuracy:", classifiers["adaboost"].gridSearch.best_score_)
y_pred_adaboost = classifiers["adaboost"].gridSearch.predict(X_train_prepared)

plotConfusionMatrix(confusion_matrix(y_train, y_pred_adaboost), "Adaboost", ["<=50K", ">50K"])

plt.show()
X_test_prepared = prepPipeline.transform(X_test)

y_test_predictions = classifiers["adaboost"].gridSearch.predict(X_test_prepared)

print(accuracy_score(y_test, y_test_predictions))