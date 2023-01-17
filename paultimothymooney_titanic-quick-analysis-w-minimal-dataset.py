import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import itertools
from sklearn.metrics import make_scorer, accuracy_score
from sklearn import model_selection
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
#from sklearn.metrics import make_scorer, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
%matplotlib inline

# os.chdir('/Users/ptm/desktop/Current_working_directory')
# trainingData = pd.read_csv('train.csv')
# testingData = pd.read_csv('test.csv')
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
        accuracy_results = model_selection.cross_val_score(model, a, b, cv=kfold, scoring='accuracy')
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
        f1_results = model_selection.cross_val_score(model, a, b, cv=kfold, scoring='f1_macro')
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
    accuracy = model_selection.cross_val_score(model, a, b, cv=kfold, scoring='accuracy')
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
    accuracy = model_selection.cross_val_score(model, a, b, cv=kfold, scoring='accuracy')
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
    accuracy = model_selection.cross_val_score(model, a, b, cv=kfold, scoring='accuracy')
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
    accuracy = model_selection.cross_val_score(classifier, a, b, cv=kfold, scoring='accuracy')
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
# It looks like our model can predict with about 70-75% accuracty whether or not a given
# passenger survived the sinking of the Titanic despite using only a minimal dataset.  That is pretty good!

#testingData2 = pd.read_csv('test.csv')
#model = SVC()s
#model.fit(X_train, Y_train)
#prediction = model.predict(testingData)
#submission = pd.DataFrame({
#    "PassengerId": testingData2["PassengerId"],
#    "Survived": prediction})
#submission.to_csv('new_submission.csv', index=False)

# to finish the submission process, upload the file 'new_submission.csv' to Kaggle
