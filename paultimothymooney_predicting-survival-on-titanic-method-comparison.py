import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import itertools
from __future__ import print_function
from sklearn import model_selection
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, StratifiedKFold, GridSearchCV, train_test_split, learning_curve
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
%matplotlib inline
#os.chdir('/Users/ptm/desktop/Current_working_directory')

#trainingData = pd.read_csv('train.csv')
#testingData = pd.read_csv('test.csv')
trainingData = pd.read_csv('../input/train.csv')
testingData = pd.read_csv('../input/test.csv')

def describeTheData(input):
    """ 
    Describe: (1) the name of each column; (2) the number of values in each column; 
    (3) the number of missing/NaN values in each column; (4) the contents of the first 5 rows; and
    (5) the contents of the last 5 rows.
    """  
    print('\nColumn Values:\n')
    print(input.columns.values)
    print('\nValue Counts:\n')
    print(input.info())
    print('\nNull Value Counts:\n')
    print(input.isnull().sum())
    print('\nFirst Few Values:\n')
    print(input.head())
    print('\nLast Few Values:\n')
    print(input.tail())
    print('')
describeTheData(trainingData)
def plotAgeDistribution(input):
    """ 
    Plot the distribution of ages for passengers that either did or did not survive the sinking of the Titanic.
    """  
    sns.set_style("whitegrid")
    distributionOne = sns.FacetGrid(input, hue="Survived",aspect=2)
    distributionOne.map(plt.hist, 'Age', bins=12)
    distributionOne.add_legend()
    distributionOne.set_axis_labels('Age', 'Count')
    distributionOne.fig.suptitle('Survival Probability vs Age (Blue = Died; Orange = Survived)')
    distributionTwo = sns.FacetGrid(input, hue="Survived",aspect=2)
    distributionTwo.map(sns.kdeplot,'Age',shade= True)
    distributionTwo.set(xlim=(0, input['Age'].max()))
    distributionTwo.add_legend()
    distributionTwo.set_axis_labels('Age', 'Proportion')
    distributionTwo.fig.suptitle('Survival Probability vs Age (Blue = Died; Orange = Survived)')
plotAgeDistribution(trainingData)
trainingData = trainingData.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Embarked', 'Cabin'], axis=1)
testingData = testingData.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Embarked', 'Cabin'], axis=1)
def replaceMissingValuesWithMedianValues(input):
    input['Fare'].fillna(input['Fare'].dropna().median(), inplace=True)   
    input['Age'].fillna(input['Fare'].dropna().median(), inplace=True)
replaceMissingValuesWithMedianValues(trainingData)
replaceMissingValuesWithMedianValues(testingData)

def sexToBinary(input):
    """ 0 = "female" and 1 = "male".""" 
    trainingData["Sex"] = trainingData["Sex"].astype("category")
    trainingData["Sex"].cat.categories = [0,1]
    trainingData["Sex"] = trainingData["Sex"].astype("int")
sexToBinary(trainingData)
sexToBinary(testingData)

def ageToCategory(input):
    """ 
    0 = "ages between 0 and 4", 1 = "ages between 4 and 12",
    2 = "ages between 12 and 18", 3 = "ages between 18 and 60", and 4 = "ages between 60 and 150".
    """ 
    input['Age'] = input.Age.fillna(-0.5)
    bins = (-0.01, 4, 12, 18, 60, 150)
    categories = pd.cut(input.Age, bins, labels=False)
    input.Age = categories
ageToCategory(trainingData)
ageToCategory(testingData)

def fareToCategory(input):
    """ 
    0 = "ticket price < $10", 1 = "$10<X<$20", 2 = "$20<X<$30", 
    and 3 = "ticket price > $30".
    """ 
    input['Fare'] = input.Fare.fillna(-0.5)
    bins = (-0.01, 10, 20, 30, 1000)
    categories = pd.cut(input.Fare, bins, labels=False)
    input.Fare = categories
fareToCategory(trainingData)
fareToCategory(testingData)

# Next we will need to split up our training data, setting aside 20% of the training data for cross-validation testing, such that we can avoid potentially overfitting the data.
xValues = trainingData.drop(['Survived'], axis=1)
yValues = trainingData['Survived']
X_train, X_test, Y_train, Y_test = train_test_split(xValues, yValues, test_size=0.2, random_state=23)
def describeDataAgain(input):
    """ 
    The output is as follows: (1) the name of each column; (2) the contents of the first 5 rows; and
    (3) the number of missing/NaN values in each column; 
    """ 
    print('\nNew summary of data after making changes:\n')
    print('Column Values:\n')
    print(input.columns.values)
    print('\nFirst Few Values:\n')
    print(input.head())
    print('\nNull Value Counts:\n')
    print(input.isnull().sum())
    print("")
describeDataAgain(trainingData)
def makeAHeatMap(input):
    """  heatmap showing the relationship between each numerical feature """  
    plt.figure(figsize=[8,6])
    heatmap = sns.heatmap(input.corr(), vmax=1.0, square=True, annot=True)
    heatmap.set_title('Pearson Correlation Coefficients')
makeAHeatMap(trainingData)
def pivotTheData(input):
    print('\nPivot Tables:\n')
    print(input[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
    print('')
    print(input[["Fare", "Survived"]].groupby(['Fare'], as_index=False).mean().sort_values(by='Survived', ascending=False))
    print('')
    print(input[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
    print('')
    return
pivotTheData(trainingData)
def plotTheData(input):
    """ 
    The output is as follows: (1) survival probability vs gender; (2) survival probability vs ticket class; 
    and (3) survival probability vs gender vs ticket class.
    """  
    plt.figure(figsize=[10,6])
    plt.subplot(221)
    plotOne = sns.barplot('Sex', 'Survived', data=input, capsize=.1)
    plotOne.set_title('Survival Probability vs Gender (Blue=Female, Green=Male)')
    plt.subplot(222)
    plotTwo = sns.barplot('Pclass', 'Survived', data=input, capsize=.1, linewidth=2.5, facecolor=(1, 1, 1, 0), errcolor=".2", edgecolor=".2")
    plotTwo.set_title('Survival Probability vs Ticket Class')
plotTheData(trainingData)
def plotTheDataAgain(input):
    """ 
    The output is as follows: (1) survival probability vs gender; (2) survival probability vs ticket class; 
    and (3) survival probability vs gender vs ticket class.
    """  
    plt.figure(figsize=[8,5])
    plotThree = sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=input,
                  palette={1: "green", 0: "blue"},
                  markers=["*", "o"], linestyles=["-", "--"]);
    plotThree.set_title('Survival Probability vs Gender vs Ticket Class (Blue=Female, Green=Male)')
plotTheDataAgain(trainingData)
def compareABunchOfDifferentModelsAccuracy(a,b,c,d):
    print('\nCompare Multiple Classifiers:\n')
    print('K-Fold Cross-Validation Accuracy:\n')
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('RF', RandomForestClassifier()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('SVM', SVC()))
    models.append(('LSVM', LinearSVC()))
    models.append(('GNB', GaussianNB()))
    models.append(('DTC', DecisionTreeClassifier()))
    models.append(('GBC', GradientBoostingClassifier()))
    models.append(('LDA', LinearDiscriminantAnalysis())) 
    resultsAccuracy = []
    names = []
    for name, model in models:
        model.fit(a, b)
        kfold = model_selection.KFold(n_splits=10, random_state=7)
        accuracy_results = model_selection.cross_val_score(model, a,b, cv=kfold, scoring='accuracy')
        resultsAccuracy.append(accuracy_results)
        names.append(name)
        accuracyMessage = "%s: %f (%f)" % (name, accuracy_results.mean(), accuracy_results.std())
        print(accuracyMessage)
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison: Accuracy')
    ax = fig.add_subplot(111)
    plt.boxplot(resultsAccuracy)
    ax.set_xticklabels(names)
    ax.set_ylabel('Cross-Validation: Accuracy Score')
    plt.show()
compareABunchOfDifferentModelsAccuracy(X_train, Y_train, X_test, Y_test)

def defineModels():
    print('\nLR = LogisticRegression')
    print('RF = RandomForestClassifier')
    print('KNN = KNeighborsClassifier')
    print('SVM = Support Vector Machine SVC')
    print('LSVM = LinearSVC')
    print('GNB = GaussianNB')
    print('DTC = DecisionTreeClassifier')
    print('GBC = GradientBoostingClassifier')
    print('LDA = LinearDiscriminantAnalysis\n')
defineModels()
def compareABunchOfDifferentModelsF1Score(a,b,c,d):
    print('\nCompare Multiple Classifiers:\n')
    print('F1 Score:\n')
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('RF', RandomForestClassifier()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('SVM', SVC()))
    models.append(('LSVM', LinearSVC()))
    models.append(('GNB', GaussianNB()))
    models.append(('DTC', DecisionTreeClassifier()))
    models.append(('GBC', GradientBoostingClassifier()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    resultsF1 = []
    names = []
    for name, model in models:
        model.fit(a, b)
        kfold = model_selection.KFold(n_splits=10, random_state=7)
        f1_results = model_selection.cross_val_score(model, a,b, cv=kfold, scoring='f1_macro')
        resultsF1.append(f1_results)
        names.append(name)
        f1Message = "%s: %f (%f)" % (name, f1_results.mean(), f1_results.std())
        print(f1Message)
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison: F1 Score')
    ax = fig.add_subplot(111)
    plt.boxplot(resultsF1)
    ax.set_xticklabels(names)
    ax.set_ylabel('Cross-Validation: F1 Score')
    plt.show()
compareABunchOfDifferentModelsF1Score(X_train, Y_train, X_test, Y_test)
defineModels()
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Plots a learning curve. http://scikit-learn.org/stable/modules/learning_curve.html
    """
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
plot_learning_curve(LogisticRegression(), 'Learning Curve For Logistic Regression Classifier', X_train, Y_train, (0.75,0.95), 10)
plot_learning_curve(KNeighborsClassifier(), 'Learning Curve For K Neighbors Classifier', X_train, Y_train, (0.75,0.95), 10)
plot_learning_curve(SVC(), 'Learning Curve For SVM Classifier', X_train, Y_train, (0.75,0.95), 10)
def selectParametersForLR(a, b, c, d):
    model = LogisticRegression()
    parameters = {'C': [0.01, 0.1, 0.5, 1.0, 5.0, 10, 25, 50, 100],
                  'solver' : ['newton-cg', 'lbfgs', 'liblinear']}
    accuracy_scorer = make_scorer(accuracy_score)
    grid_obj = GridSearchCV(model, parameters, scoring=accuracy_scorer, error_score = 0.01)
    grid_obj = grid_obj.fit(a, b)
    model = grid_obj.best_estimator_
    model.fit(a, b)
    print('\nSelected Parameters for LR:\n')
    print(model)
    print('')
#    predictions = model.predict(c)
#    print(accuracy_score(d, predictions))
#    print('Logistic Regression - Training set accuracy: %s' % accuracy_score(d, predictions))
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    accuracy = model_selection.cross_val_score(model, a,b, cv=kfold, scoring='accuracy')
    mean = accuracy.mean() 
    stdev = accuracy.std()
    print('\nLogistic Regression - Training set accuracy: %s (%s)' % (mean, stdev))
selectParametersForLR(X_train, Y_train, X_test, Y_test)

def selectParametersForSVM(a, b, c, d):
    model = SVC()
    parameters = {'C': [0.01, 0.1, 0.5, 1.0, 5.0, 10, 25, 50, 100],
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    accuracy_scorer = make_scorer(accuracy_score)
    grid_obj = GridSearchCV(model, parameters, scoring=accuracy_scorer)
    grid_obj = grid_obj.fit(a, b)
    model = grid_obj.best_estimator_
    model.fit(a, b)
    print('\nSelected Parameters for SVM:\n')
    print(model)
#    predictions = model.predict(c)
#    print(accuracy_score(d, predictions))
#    print('Logistic Regression - Training set accuracy: %s' % accuracy_score(d, predictions))
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    accuracy = model_selection.cross_val_score(model, a,b, cv=kfold, scoring='accuracy')
    mean = accuracy.mean() 
    stdev = accuracy.std()
    print('\nSupport Vector Machine - Training set accuracy: %s (%s)' % (mean, stdev))
selectParametersForSVM(X_train, Y_train, X_test, Y_test)

def selectParametersForKNN(a, b, c, d):

    model = KNeighborsClassifier()
    parameters = {'n_neighbors': [5, 10, 25, 50],
                  'algorithm': ['ball_tree', 'kd_tree'],
                  'leaf_size': [5, 10, 25, 50]}
    accuracy_scorer = make_scorer(accuracy_score)
    grid_obj = GridSearchCV(model, parameters, scoring=accuracy_scorer)
    grid_obj = grid_obj.fit(a, b)
    model = grid_obj.best_estimator_
    model.fit(a, b)
    print('\nSelected Parameters for KNN:\n')
    print(model)
#    predictions = model.predict(c)
#    print(accuracy_score(d, predictions))
#    print('Logistic Regression - Training set accuracy: %s' % accuracy_score(d, predictions))
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    accuracy = model_selection.cross_val_score(model, a,b, cv=kfold, scoring='accuracy')
    mean = accuracy.mean() 
    stdev = accuracy.std()
    print('\nK-Nearest Neighbors Classifier - Training set accuracy: %s (%s)' % (mean, stdev))
    print('')
selectParametersForKNN(X_train, Y_train,  X_test, Y_test)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize = (5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
dict_characters = {0: 'Did Not Survive', 1: 'Survived'}

def runSVMconfusion(a,b,c,d):
    classifier = SVC(C=50, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
    classifier.fit(a, b)
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    accuracy = model_selection.cross_val_score(classifier, a,b, cv=kfold, scoring='accuracy')
    mean = accuracy.mean() 
    stdev = accuracy.std()
    print('SKlearn Multi-layer Perceptron NN - Training set accuracy: %s (%s)\n' % (mean, stdev))
    prediction = classifier.predict(c)
    cnf_matrix = confusion_matrix(d, prediction)
    np.set_printoptions(precision=2)
    class_names = dict_characters 
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix')
plot_learning_curve(SVC(C=50, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False), 'Learning Curve For SVM Classifier', X_train, Y_train, (0.75,0.95), 10)
runSVMconfusion(X_train, Y_train,  X_test, Y_test)
#trainingData = pd.read_csv('train.csv')
#testingData = pd.read_csv('test.csv')
trainingData = pd.read_csv('../input/train.csv')
testingData = pd.read_csv('../input/test.csv')
trainingData['is_test'] = 0 # this will be helpful when we split up the data again later
testingData['is_test'] = 1 
fullData = pd.concat((trainingData, testingData), axis=0)
fullData = fullData.drop(['PassengerId'], axis=1)

def replaceMissingWithMedian(dataframe):
    """ Replace missing values wiht median values"""
    dataframe['Fare'].fillna(fullData['Fare'].dropna().median(), inplace=True)   
    dataframe['Age'].fillna(fullData['Age'].dropna().median(), inplace=True)
    dataframe['Cabin'].fillna('Z', inplace=True)
    dataframe['Embarked'].fillna('S', inplace=True)
    dataframe = dataframe.fillna(-0.5)
replaceMissingWithMedian(fullData)

def replaceAgeNumbersWithCategories(dataframe):
    """Replace age numbers with age categories"""
    dataframe['Age'] = dataframe.Age.fillna(-0.5)
    bins = (-0.01, 4, 12, 18, 60, 150)
    categories = pd.cut(dataframe.Age, bins, labels=False)
    dataframe.Age = categories
replaceAgeNumbersWithCategories(fullData)

def replaceFareNumbersWithCategories(dataframe):
    """Replace fare numbers with fare categories"""
    dataframe['Fare'] = dataframe.Fare.fillna(-0.5)
    bins = (-0.01, 10, 20, 30, 1000)
    categories = pd.cut(dataframe.Fare, bins, labels=False)
    dataframe.Fare = categories
replaceFareNumbersWithCategories(fullData)
def makeFeatureNameTitle(dataframe):
    """ make feature Name Title from out feature Name"""
    name = dataframe['Name']
    full_name = name.str.split(', ', n=0, expand=True)
    last_name = full_name[0]
    titles = full_name[1].str.split('.', n=0, expand=True)
    titles = titles[0]
    dataframe['Name'] = titles
    newTitles = titles.replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataframe['Name'] = newTitles
makeFeatureNameTitle(fullData)

def extractCabinLetterDeleteOther(dataframe):
    """ Extract Cabin Letter from Cabin Name and Delete all other info"""
    dataframe['Cabin'] = dataframe['Cabin'].str[0]    
    # Extract Family Size from SibSp + Parch and then make new family size groups
    dataframe['FamilySize'] = dataframe['SibSp'] + dataframe['Parch'] + 1
    dataframe['FamilySize'] = dataframe.FamilySize.fillna(-0.5)
    bins = (-1, 1, 2, 4, 6, 1000)
    categories = pd.cut(dataframe.FamilySize, bins, labels=False)
    dataframe['FamilySize'] = categories
extractCabinLetterDeleteOther(fullData)

def extractTicketPrefix(dataframe):
    """Extract whether or not the Ticket Number has a Prefix (possibly indicating special privileges)"""
    Ticket = []
    for i in list(dataframe.Ticket):
        if not i.isdigit() :
            Ticket.append("1")
        else:
            Ticket.append("0")     
    dataframe["Ticket"] = Ticket
extractTicketPrefix(fullData)

# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html
fullData = pd.get_dummies(fullData, columns=['Pclass', 'Sex', 'Embarked', 'Age', 'Fare', 'Cabin', 'Name', 'FamilySize', 'Ticket'])
# Now we split the combined data back into training and testing data since we have finished with the feature engineering
trainingData = fullData[fullData['is_test'] == 0]
testingData = fullData[fullData['is_test'] == 1]
# Describe the data
describeDataAgain(trainingData)

#http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
#http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
X = trainingData.drop(['Survived', 'is_test'], axis=1)
y = trainingData['Survived']
xValues = X
yValues = y.values.ravel()
X_train, X_test, Y_train, Y_test = train_test_split(xValues, yValues, test_size=0.2)
compareABunchOfDifferentModelsAccuracy(X_train, Y_train, X_test, Y_test)
defineModels()
# http://scikit-learn.org/stable/modules/learning_curve.html
def plotLotsOfLearningCurves(a,b):
    """Now let's plot a bunch of learning curves
    # http://scikit-learn.org/stable/modules/learning_curve.html
    """
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('RF', RandomForestClassifier()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('SVM', SVC()))
    models.append(('LSVM', LinearSVC()))
    #models.append(('GNB', GaussianNB()))
    #models.append(('DTC', DecisionTreeClassifier()))
    models.append(('GBC', GradientBoostingClassifier()))
    #models.append(('LDA', LinearDiscriminantAnalysis()))
    for name, model in models:
        plot_learning_curve(model, 'Learning Curve For %s Classifier'% (name), a,b, (0.75,0.95), 10)
plotLotsOfLearningCurves(X_train, Y_train)
def determineOptimalFeatureNumber(a,b):
    """
    #http://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html
    """
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('RF', RandomForestClassifier()))
    #models.append(('KNN', KNeighborsClassifier()))
    #models.append(('SVM', SVC()))
    models.append(('LSVM', LinearSVC()))
    #models.append(('GNB', GaussianNB()))
    models.append(('DTC', DecisionTreeClassifier()))
    models.append(('GBC', GradientBoostingClassifier()))
    #models.append(('LDA', LinearDiscriminantAnalysis()))
    for name, model in models:
        # Create the RFE object and compute a cross-validated score.
        currentModel = model
        # The "accuracy" scoring is proportional to the number of correct
        # classifications
        rfecv = RFECV(estimator=currentModel, step=1, cv=StratifiedKFold(2), scoring='accuracy')
        rfecv.fit(a,b)
        print("Optimal number of features : %d" % rfecv.n_features_)
        # Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Number of features selected for %s" % (name))
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.show()
determineOptimalFeatureNumber(X_train, Y_train)
#Run LinearSVC
def runLinearSVC(a,b,c,d):
    """Run LinearSVC w/ Kfold CV"""
    model = LinearSVC()
    model.fit(a,b)
    kfold = model_selection.KFold(n_splits=10)
    accuracy = model_selection.cross_val_score(model, a,b, cv=kfold, scoring='accuracy')
    mean = accuracy.mean() 
    stdev = accuracy.std()
    print('LinearSVC - Training set accuracy: %s (%s)' % (mean, stdev))
    print('')
runLinearSVC(X_train, Y_train, X_test, Y_test)
# Identify best feature coefficients (coef_) and/or feature importance (feature_importances_)
model = LinearSVC()
model.fit(X_train,Y_train) # Needed to initialize coef_
columns = X_train.columns
coefficients = model.coef_.reshape(X_train.columns.shape[0], 1)
absCoefficients = abs(coefficients)
fullList = pd.concat((pd.DataFrame(columns, columns = ['Variable']), pd.DataFrame(absCoefficients, columns = ['absCoefficient'])), axis = 1).sort_values(by='absCoefficient', ascending = False)
print('LinearSVC - Feature Importance:\n')
print(fullList)
print('')
# Remove all but the most helpful features
topTwenty = fullList[:15]
featureList = topTwenty.values
featureList = pd.DataFrame(featureList)
featuresOnly = featureList[0]
featuresOnly = list(featuresOnly)
featuresOnly += ['is_test', 'Survived']
fullData = fullData[featuresOnly]
trainingData = fullData[fullData['is_test'] == 0]
testingData = fullData[fullData['is_test'] == 1]
#g = sns.heatmap(trainingData[featuresOnly].corr(),cmap="BrBG",annot=False)

# Let's see if we improved our accuracy scores
X = trainingData.drop(['Survived', 'is_test'], axis=1)
y = trainingData['Survived']
xValues = X
yValues = y.values.ravel()
X_train, X_test, Y_train, Y_test = train_test_split(xValues, yValues, test_size=0.2)
print('\nDataset reduced to the following columns:\n')
print(X_train.columns)
print('\nAfter feature selection:\n')
runLinearSVC(X_train, Y_train, X_test, Y_test)
print('After Feature Selection\n')
compareABunchOfDifferentModelsAccuracy(X_train, Y_train, X_test, Y_test)
defineModels()
g = sns.heatmap(X_train.corr(),cmap="BrBG",annot=False)
# Minimum percentage of variance we want to be described by the resulting transformed components
variance_pct = .99
# Create PCA object
pca = PCA(n_components=variance_pct)
# Transform the initial features
X_transformed = pca.fit_transform(X,y)
# Create a data frame from the PCA'd data
pcaDataFrame = pd.DataFrame(X_transformed) 
#print(pcaDataFrame.shape[1], " components describe ", str(variance_pct)[1:], "% of the variance")
# Redefine X_train, X_test, Y_train, Y_test
xValues = pcaDataFrame
yValues = y.values.ravel()
X_train, X_test, Y_train, Y_test = train_test_split(xValues, yValues, test_size=0.2)
# Now do it to the test data as well
testingData = testingData.drop(['Survived', 'is_test'], axis=1)
testingData = pca.fit_transform(testingData)
testingData = pd.DataFrame(testingData) 
# There are fewer numbers of features now (dimensionality reduction)
# The features are no longer correlated, as illustrated below:
g = sns.heatmap(X_train.corr(),cmap="BrBG",annot=False)
print('\nAfter feature selection + PCA:\n')
runLinearSVC(X_train, Y_train, X_test, Y_test)
print('After Feature Selection + PCA\n')
compareABunchOfDifferentModelsAccuracy(X_train, Y_train, X_test, Y_test)
defineModels()
plot_learning_curve(LinearSVC(), 'Learning Curve For %s Classifier'% ('LinearSVC'), X_train, Y_train, (0.75,0.95), 10)
plot_learning_curve(LogisticRegression(), 'Learning Curve For %s Classifier'% ('LogisticRegression'), X_train, Y_train, (0.75,0.95), 10)
def selectParametersForLSVM(a, b, c, d):
    """http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    http://scikit-learn.org/stable/modules/grid_search.html#grid-search"""
    model = LinearSVC()
    parameters = {'C': [0.00001, 0.001, .01, 0.1, 0.5, 1.0, 5.0, 10, 25, 50, 100, 1000]}
    accuracy_scorer = make_scorer(accuracy_score)
    grid_obj = GridSearchCV(model, parameters, scoring=accuracy_scorer)
    grid_obj = grid_obj.fit(a, b)
    model = grid_obj.best_estimator_
    model.fit(a, b)
    print('Selected Parameters for LSVM:')
    print('')
    print(model)
    print('')
#    predictions = model.predict(c)
#    print(accuracy_score(d, predictions))
#    print('Logistic Regression - Training set accuracy: %s' % accuracy_score(d, predictions))
    kfold = model_selection.KFold(n_splits=10)
    accuracy = model_selection.cross_val_score(model, a,b, cv=kfold, scoring='accuracy')
    mean = accuracy.mean() 
    stdev = accuracy.std()
    print('Linear Support Vector Machine - Training set accuracy: %s (%s)' % (mean, stdev))
    print('')
    return
selectParametersForLSVM(X_train, Y_train, X_test, Y_test)
runLinearSVC(X_train, Y_train, X_test, Y_test)
def runMLPC(a,b,c,d):
    classifier = MLPC(activation='relu', max_iter=1000)
    classifier.fit(a, b)
    kfold = model_selection.KFold(n_splits=10)
    accuracy = model_selection.cross_val_score(classifier, a,b, cv=kfold, scoring='accuracy')
    mean = accuracy.mean() 
    stdev = accuracy.std()
    print('SKlearn Multi-layer Perceptron NN - Training set accuracy: %s (%s)' % (mean, stdev))
    print('')
runMLPC(X_train, Y_train,  X_test, Y_test)
def selectParametersForMLPC(a, b, c, d):
    """http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    http://scikit-learn.org/stable/modules/grid_search.html#grid-search"""
    model = MLPC()
    parameters = {'verbose': [False],
                  'activation': ['logistic', 'relu'],
                  'max_iter': [1000, 2000], 'learning_rate': ['constant', 'adaptive']}
    accuracy_scorer = make_scorer(accuracy_score)
    grid_obj = GridSearchCV(model, parameters, scoring=accuracy_scorer)
    grid_obj = grid_obj.fit(a, b)
    model = grid_obj.best_estimator_
    model.fit(a, b)
    print('Selected Parameters for Multi-Layer Perceptron NN:\n')
    print(model)
    print('')
#    predictions = model.predict(c)
#    print(accuracy_score(d, predictions))
#    print('Logistic Regression - Training set accuracy: %s' % accuracy_score(d, predictions))
    kfold = model_selection.KFold(n_splits=10)
    accuracy = model_selection.cross_val_score(model, a,b, cv=kfold, scoring='accuracy')
    mean = accuracy.mean() 
    stdev = accuracy.std()
    print('SKlearn Multi-Layer Perceptron - Training set accuracy: %s (%s)' % (mean, stdev))
    print('')
selectParametersForMLPC(X_train, Y_train,  X_test, Y_test)
def runVotingClassifier(a,b,c,d):
    """http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
    http://scikit-learn.org/stable/modules/ensemble.html#voting-classifier"""
    global votingC, mean, stdev # eventually I should get rid of these global variables and use classes instead.  in this case i need these variables for the submission function.
    votingC = VotingClassifier(estimators=[('LSVM', LinearSVC(C=0.0001, class_weight=None, dual=True, fit_intercept=True,
         intercept_scaling=1, loss='squared_hinge', max_iter=1000,
         multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
         verbose=0)), ('MLPC', MLPC(activation='logistic', alpha=0.0001, batch_size='auto',
           beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(100,), learning_rate='constant',
           learning_rate_init=0.001, max_iter=2000, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=None,
           shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
           verbose=False, warm_start=False))], voting='hard')  
    votingC = votingC.fit(a,b)   
    kfold = model_selection.KFold(n_splits=10)
    accuracy = model_selection.cross_val_score(votingC, a,b, cv=kfold, scoring='accuracy')
    meanC = accuracy.mean() 
    stdevC = accuracy.std()
    print('Ensemble Voting Method - Training set accuracy: %s (%s)' % (meanC, stdevC))
    print('')
    return votingC, meanC, stdevC
runVotingClassifier(X_train,Y_train,X_test,Y_test)
model = votingC
model.fit(X_train, Y_train)
prediction = model.predict(X_test)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test, prediction)
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
class_names = ["Survived", "Did Not Survive"]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.show()
# # Submission with Ensemble Voting Classification Method
# testingData2 = pd.read_csv('../input/test.csv')
# model = votingC
# model.fit(X_train, Y_train)
# prediction = model.predict(testingData)
# prediction = prediction.astype(int)
# submission = pd.DataFrame({
#     "PassengerId": testingData2["PassengerId"],
#     "Survived": prediction})
# submission.to_csv('_new_submission_ensemble.csv', index=False)
# # to finish the submission process, upload the file '_new_submission_.csv' to Kaggle
