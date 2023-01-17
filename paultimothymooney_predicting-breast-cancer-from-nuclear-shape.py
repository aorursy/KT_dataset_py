import cv2
import matplotlib.pyplot as plt
PNG_location = "/kaggle/input/webster-2009/webster 2009.png"
def plotPNG(a):
    """
    Plot a PNG Image w/ Matplotlib
    """
    PNG = cv2.imread(PNG_location)
    PNG = cv2.resize(PNG, (512,256))
    plt.imshow(cv2.cvtColor(PNG, cv2.COLOR_BGR2RGB)); plt.axis('off')
    return
plotPNG(PNG_location)
print("Webster, M., Witkin, K.L., and Cohen-Fix, O. (2009). Sizing up the nucleus: nuclear shape, size and nuclear-envelope assembly. J. Cell Sci. 122, 1477–1486.")
#from __future__ import print_function
#import os
import itertools
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score 
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier as MLPC
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
%matplotlib inline
sizeMeasurements = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
def describeData(a):
    """ 
    Print column titles, first few values, and null value counts
    """  
    print('\n Column Values: \n\n', a.columns.values, "\n")
    print('\n First Few Values: \n\n', a.head(), "\n")
    print('\n Null Value Counts: \n\n', a.isnull().sum(), "\n")
describeData(sizeMeasurements)
def plotSizeDistribution(a):
    """ 
    Plot size distribution for benign vs malignant samples
    """  
    sns.set_style("whitegrid")
    distributionOne = sns.FacetGrid(a, hue="diagnosis",aspect=2.5)
    distributionOne.map(plt.hist, 'area_mean', bins=30)
    distributionOne.add_legend()
    distributionOne.set_axis_labels('area_mean', 'Count')
    distributionOne.fig.suptitle('Area vs Diagnosis (Blue = Malignant; Orange = Benign)')
    distributionTwo = sns.FacetGrid(a, hue="diagnosis",aspect=2.5)
    distributionTwo.map(sns.kdeplot,'area_mean',shade=True)
    distributionTwo.set(xlim=(0, a['area_mean'].max()))
    distributionTwo.add_legend()
    distributionTwo.set_axis_labels('area_mean', 'Proportion')
    distributionTwo.fig.suptitle('Area vs Diagnosis (Blue = Malignant; Orange = Benign)')
plotSizeDistribution(sizeMeasurements)
def plotConcaveDistribution(a):
    """ 
    Plot shape distribution for benign vs malignant samples
    """  
    sns.set_style("whitegrid")
    distributionOne = sns.FacetGrid(a, hue="diagnosis",aspect=2.5)
    distributionOne.map(plt.hist, 'concave points_mean', bins=30)
    distributionOne.add_legend()
    distributionOne.set_axis_labels('concave points_mean', 'Count')
    distributionOne.fig.suptitle('# of Concave Points vs Diagnosis (Blue = Malignant; Orange = Benign)')
    distributionTwo = sns.FacetGrid(a, hue="diagnosis",aspect=2.5)
    distributionTwo.map(sns.kdeplot,'concave points_mean',shade= True)
    distributionTwo.set(xlim=(0, a['concave points_mean'].max()))
    distributionTwo.add_legend()
    distributionTwo.set_axis_labels('concave points_mean', 'Proportion')
    distributionTwo.fig.suptitle('# of Concave Points vs Diagnosis (Blue = Malignant; Orange = Benign)')
plotConcaveDistribution(sizeMeasurements)
def diagnosisToBinary(a):
    """ 
    convert diagnosis to binary label
    """ 
    a["diagnosis"] = a["diagnosis"].astype("category")
    a["diagnosis"].cat.categories = [0,1]
    a["diagnosis"] = a["diagnosis"].astype("int")
diagnosisToBinary(sizeMeasurements)

xValues = sizeMeasurements.drop(['diagnosis', 'Unnamed: 32', 'id'], axis=1)
yValues = sizeMeasurements['diagnosis']
xValuesScaled = preprocessing.scale(xValues)
xValuesScaled = pd.DataFrame(xValuesScaled, columns = xValues.columns)
variance_pct = .99 # Minimum percentage of variance we want to be described by the resulting transformed components
pca = PCA(n_components=variance_pct) # Create PCA object
X_transformed = pca.fit_transform(xValuesScaled,yValues) # Transform the initial features
xValuesScaledPCA = pd.DataFrame(X_transformed) # Create a data frame from the PCA'd data

X_trainOriginal, X_testOriginal, Y_trainOriginal, Y_testOriginal = train_test_split(xValues, yValues, test_size=0.2)
X_trainScaled, X_testScaled, Y_trainScaled, Y_testScaled = train_test_split(xValuesScaled, yValues, test_size=0.2)
X_trainScaledPCA, X_testScaledPCA, Y_trainScaledPCA, Y_testScaledPCA = train_test_split(xValuesScaledPCA, yValues, test_size=0.2)
print("\nFeature Correlation Before PCA:\n")
g = sns.heatmap(X_trainOriginal.corr(),cmap="BrBG",annot=False)
print("\nFeature Correlation After PCA:\n")
i = sns.heatmap(X_trainScaledPCA.corr(),cmap="BrBG",annot=False)
print('\n First Few Values, Original: \n\n', xValues.head(), "\n\n")
print('First Few Values, Scaled: \n\n,',xValuesScaled.head(),'\n\n')
print('First Few Values, After PCA: \n\n,',xValuesScaledPCA.head(),'\n\n')
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

dict_characters = {0: 'Malignant', 1: 'Benign'}


def compareABunchOfDifferentModelsAccuracy(a, b, c, d):
    """
    compare performance of classifiers on X_train, X_test, Y_train, Y_test
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    http://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score
    """    
    print('\nCompare Multiple Classifiers: \n')
    print('K-Fold Cross-Validation Accuracy: \n')
    names = []
    models = []
    resultsAccuracy = []
    models.append(('LR', LogisticRegression()))
    models.append(('RF', RandomForestClassifier()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('SVM', SVC()))
    models.append(('LSVM', LinearSVC()))
    models.append(('GNB', GaussianNB()))
    models.append(('DTC', DecisionTreeClassifier()))
    models.append(('GBC', GradientBoostingClassifier()))
    #models.append(('LDA', LinearDiscriminantAnalysis()))
    for name, model in models:
        model.fit(a, b)
        kfold = model_selection.KFold(n_splits=10, random_state=7)
        accuracy_results = model_selection.cross_val_score(model, a,b, cv=kfold, scoring='accuracy')
        resultsAccuracy.append(accuracy_results)
        names.append(name)
        accuracyMessage = "%s: %f (%f)" % (name, accuracy_results.mean(), accuracy_results.std())
        print(accuracyMessage) 
    # Boxplot
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison: Accuracy')
    ax = fig.add_subplot(111)
    plt.boxplot(resultsAccuracy)
    ax.set_xticklabels(names)
    ax.set_ylabel('Cross-Validation: Accuracy Score')
    plt.show()

print("Before Data Scaling:")
compareABunchOfDifferentModelsAccuracy(X_trainOriginal, Y_trainOriginal, X_testOriginal, Y_testOriginal)
print("After Data Scaling:")
compareABunchOfDifferentModelsAccuracy(X_trainScaled, Y_trainScaled, X_testScaled, Y_testScaled)
print("After PCA:")
compareABunchOfDifferentModelsAccuracy(X_trainScaledPCA, Y_trainScaledPCA, X_testScaledPCA, Y_testScaledPCA)

def defineModels():
    """
    This function just defines each abbreviation used in the previous function (e.g. LR = Logistic Regression)
    """
    print('\nLR = LogisticRegression')
    print('RF = RandomForestClassifier')
    print('KNN = KNeighborsClassifier')
    print('SVM = Support Vector Machine SVC')
    print('LSVM = LinearSVC')
    print('GNB = GaussianNB')
    print('DTC = DecisionTreeClassifier')
    print('GBC = GradientBoostingClassifier \n\n')
    #print('LDA = LinearDiscriminantAnalysis')
defineModels()
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
    models.append(('GNB', GaussianNB()))
    models.append(('DTC', DecisionTreeClassifier()))
    models.append(('GBC', GradientBoostingClassifier()))
    #models.append(('LDA', LinearDiscriminantAnalysis()))
    #models.append(('MLP', MLPC()))
    for name, model in models:
        plot_learning_curve(model, 'Learning Curve For %s Classifier'% (name), a,b, (0.8,1), 10)
plotLotsOfLearningCurves(X_trainScaledPCA, Y_trainScaledPCA)
def selectParametersForSVM(a, b, c, d):
    model = SVC()
    parameters = {'C': [0.01, 0.1, 0.5, 1.0, 5.0, 10, 25, 50, 100],
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    accuracy_scorer = make_scorer(accuracy_score)
    grid_obj = GridSearchCV(model, parameters, scoring=accuracy_scorer)
    grid_obj = grid_obj.fit(a, b)
    model = grid_obj.best_estimator_
    model.fit(a, b)
    print('Selected Parameters for SVM:\n')
    print(model,"\n")
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    accuracy = model_selection.cross_val_score(model, a,b, cv=kfold, scoring='accuracy')
    mean = accuracy.mean() 
    stdev = accuracy.std()
    print('Support Vector Machine - Training set accuracy: %s (%s)' % (mean, stdev))
    print('')
    prediction = model.predict(c)
    cnf_matrix = confusion_matrix(d, prediction)
    np.set_printoptions(precision=2)
    class_names = dict_characters 
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
    plot_learning_curve(model, 'Learning Curve For SVM Classifier', X_trainScaledPCA, Y_trainScaledPCA, (0.85,1), 10)
print("\nAfter Data Scaling:\n")
selectParametersForSVM(X_trainScaled, Y_trainScaled,  X_testScaled, Y_testScaled)
print("\nAfter PCA:\n")
selectParametersForSVM(X_trainScaledPCA, Y_trainScaledPCA,  X_testScaledPCA, Y_testScaledPCA)
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
    kfold = model_selection.KFold(n_splits=10)
    accuracy = model_selection.cross_val_score(model, a,b, cv=kfold, scoring='accuracy')
    mean = accuracy.mean() 
    stdev = accuracy.std()
    print('SKlearn Multi-Layer Perceptron - Training set accuracy: %s (%s)' % (mean, stdev))
    print('')
    prediction = model.predict(c)
    cnf_matrix = confusion_matrix(d, prediction)
    np.set_printoptions(precision=2)
    class_names = dict_characters 
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
    plot_learning_curve(model, 'Learning Curve For MLPC Classifier', a, b, (0.85,1), 10)
print("Before Data Scaling:\n")
selectParametersForMLPC(X_trainOriginal, Y_trainOriginal,  X_testOriginal, Y_testOriginal)
print("After Data Scaling:\n")
selectParametersForMLPC(X_trainScaled, Y_trainScaled,  X_testScaled, Y_testScaled)
print("After PCA:\n")
selectParametersForMLPC(X_trainScaledPCA, Y_trainScaledPCA,  X_testScaledPCA, Y_testScaledPCA)
def runSimpleKeras(a,b,c,d):
    """ Build and run Two different NNs using Keras"""
    #global kerasModelOne # eventually I should get rid of these global variables and use classes instead.  in this case i need these variables for the submission function.
    # kerasModelOne: simple network consisting of only two fully connected layers.
    Adagrad(lr=0.00001, epsilon=1e-08, decay=0.0)
    model = Sequential()
    model.add(Dense(input_dim=np.array(a).shape[1], units=128, kernel_initializer='normal', bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dense(units=1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    model.fit(np.array(a), np.array(b), epochs=10, verbose=2, validation_split=0.2)
    score = model.evaluate(np.array(c),np.array(d), verbose=0)
    print('\nLoss, Accuracy:\n', score)
    #kerasModelOne = model  
    #return kerasModelOne
print("Before Data Scaling:\n")
runSimpleKeras(X_trainOriginal,Y_trainOriginal,X_testOriginal,Y_testOriginal)
print("After Data Scaling:\n")
runSimpleKeras(X_trainScaled,Y_trainScaled,X_testScaled,Y_testScaled)
print("After PCA:\n")
runSimpleKeras(X_trainScaledPCA,Y_trainScaledPCA,X_testScaledPCA,Y_testScaledPCA)
def runVotingClassifier(a,b,c,d):
    """http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
    http://scikit-learn.org/stable/modules/ensemble.html#voting-classifier"""
    #global votingC, mean, stdev # eventually I should get rid of these global variables and use classes instead.  in this case i need these variables for the submission function.
    votingC = VotingClassifier(estimators=[('SVM', SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)), ('MLPC', MLPC(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
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
    print('Ensemble Voting Classifier - Training set accuracy: %s (%s)' % (meanC, stdevC))
    print('')
    #return votingC, meanC, stdevC
    prediction = votingC.predict(c)
    cnf_matrix = confusion_matrix(d, prediction)
    np.set_printoptions(precision=2)
    class_names = dict_characters 
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
    plot_learning_curve(votingC, 'Learning Curve For Ensemble Voting Classifier', X_trainScaledPCA, Y_trainScaledPCA, (0.85,1), 10)
print("\nAfter Data Scaling:\n")
runVotingClassifier(X_trainScaled, Y_trainScaled,  X_testScaled, Y_testScaled)
print("\nAfter PCA:\n")
runVotingClassifier(X_trainScaledPCA, Y_trainScaledPCA,  X_testScaledPCA, Y_testScaledPCA)