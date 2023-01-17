"""

Author : Shahar Barak

Date : 06 Janudary 2016

Revised: 06 Janudary 2016



some groups of people were more likely to survive than others, such as women, children, and the upper-class.

complete the analysis of what sorts of people were likely to survive.

In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

"""

from IPython.core.display import HTML

HTML("""

<style>

.output_png {

    display: table-cell;

    text-align: center;

    vertical-align: middle;

}

</style>

""")



# remove warnings

import warnings

warnings.filterwarnings('ignore')

# ---



#Data analysis

import itertools

import numpy as np

import pandas as pd

from pandas import Series,DataFrame

from time import time

#Graphics

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns # statistical data visualization

sns.set_style('whitegrid')

%matplotlib inline

matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)

#Machine learning

import sklearn

from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.pipeline import make_pipeline

from sklearn.cross_validation import StratifiedKFold

from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.linear_model import LogisticRegression, RidgeClassifier, Perceptron, PassiveAggressiveClassifier, SGDClassifier

from sklearn.naive_bayes import BernoulliNB, MultinomialNB

from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

from sklearn.utils.extmath import density

from sklearn.svm import LinearSVC
def ReadData():

## Read Data

    test_file = 'test.csv'

    train_file  = 'train.csv'

    test = pd.read_csv(test_file, index_col='PassengerId')

    train = pd.read_csv(train_file, index_col='PassengerId')

    return test, train



def SurvivalStats(df):

    # VARIABLE DESCRIPTIONS: survival Survival (0 = No; 1 = Yes) pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)

    # name Name sex Sex age Age sibsp Number of Siblings/Spouses Aboard parch Number of Parents/Children Aboard 

    # ticket Ticket Number fare Passenger Fare cabin Cabin 

    # embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

    

    n_passengers = df.shape[0]

    n_survivors = df['Survived'][df['Survived'] == 1].count()

    n_female = df['Sex_female'][df['Sex_female'] == 1].count()

    n_male = df['Sex_male'][df['Sex_male'] == 1].count()

    n_pc1 = df['Pclass_1'][df['Pclass_1'] == 1].count()

    n_pc2 = df['Pclass_2'][df['Pclass_2'] == 1].count()

    n_pc3 = df['Pclass_3'][df['Pclass_3'] == 1].count()

    n_female_srv = df['Sex_female'][df['Sex_female'] == 1][df['Survived'] == 1].count()

    n_male_srv = df['Sex_male'][df['Sex_male'] == 1][df['Survived'] == 1].count()

    n_pc1_srv = df['Pclass_1'][df['Pclass_1'] == 1][df['Survived'] == 1].count()

    n_pc2_srv = df['Pclass_2'][df['Pclass_2'] == 1][df['Survived'] == 1].count()

    n_pc3_srv = df['Pclass_3'][df['Pclass_3'] == 1][df['Survived'] == 1].count() 

    

    print('Number of passengers:', n_passengers)

    print('Number of survivors: %d (%.0f%%)' % (n_survivors, 100*(n_survivors/n_passengers)))

    print('Number of female: %d (%.0f%%)' % (n_female, 100*(n_female/n_passengers)))

    print('Number of male: %d (%.0f%%)' % (n_male, 100*(n_male/n_passengers)))

    print('Number of Class 1: %d (%.0f%%)' % (n_pc1, 100*(n_pc1/n_passengers)))

    print('Number of Class 2: %d (%.0f%%)' % (n_pc2, 100*(n_pc2/n_passengers)))

    print('Number of Class 3: %d (%.0f%%)' % (n_pc3, 100*(n_pc3/n_passengers)))

    print('Number of female survivors: %d (%.0f%%)' % (n_female_srv, 100*(n_female_srv/n_survivors)))

    print('Number of male survivors: %d (%.0f%%)' % (n_male_srv, 100*(n_male_srv/n_survivors)))

    print('Number of Class 1 survivors: %d (%.0f%%)' % (n_pc1_srv, 100*(n_pc1_srv/n_survivors)))

    print('Number of Class 2 survivors: %d (%.0f%%)' % (n_pc2_srv, 100*(n_pc2_srv/n_survivors)))

    print('Number of Class 3 survivors: %d (%.0f%%)' % (n_pc3_srv, 100*(n_pc3_srv/n_survivors)))

    print('')

    

def PrepareData(df):

## Prepare Data

    

    #df = get_titles(df)



    # Fill missing values and bin

    df = FillMissingAge(df)

    df = BinData(df, 'Age', [0, 14, 80, 100], ['Child', 'Adult', 'Senior'])

    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    #df = BinData(df, 'Fare', [df["Fare"].min(), df["Fare"].median(), df["Fare"].max()], ['Low', 'High'])

    df['Embarked'].fillna('S', inplace=True)

    df.Cabin.fillna('T',inplace=True)    

    # mapping each Cabin value with the cabin letter

    df['Cabin'] = df['Cabin'].map(lambda c : c[0])

     

    # Convert categorical variables into indicator variables

    df = pd.get_dummies(df, columns=['Sex'])

    

    #df = NewFeature(df)

    #df.drop(['Fare_High', 'F_C1', 'Pclass_2', 'M_C3', 'F_0-10', 'F_60-100', 'Age_60-100', 'Fare_Low', 'Age_0-10', 'M_C1', 'M_C2', 'M_0-10', 'F_C2','M_60-100','Age_10-60','F_C3'], axis=1, inplace=True)

    #df['Family_Size'] = df['Parch'] + df['SibSp']

    #df['NOT_ADULT'] = df['Age_Child'] + df['Age_Senior']

    #df['Not_Class1'] = df['Pclass_2'] + df['Pclass_3']

    #df['Not_Class3'] = df['Pclass_2'] + df['Pclass_1']

    # drop unnecessary columns, these columns won't be useful in analysis and prediction

    df.drop(['Name', 'Cabin', 'Ticket', 'Name', 'Parch', 'SibSp', 'Embarked','Pclass'], axis=1, inplace=True)

    return df



def get_titles(df):

    

    # we extract the title from each name

    df['Title'] = df['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

    

    # a map of more aggregated titles

    Title_Dictionary = {

                        "Capt":       "VIP",

                        "Col":        "VIP",

                        "Major":      "VIP",

                        "Jonkheer":   "VIP",

                        "Don":        "VIP",

                        "Sir" :       "VIP",

                        "Dr":         "VIP",

                        "Rev":        "VIP",

                        "the Countess":"VIP",

                        "Dona":       "VIP",

                        "Mme":        "Ms",

                        "Mlle":       "Ms",

                        "Ms":         "Ms",

                        "Mr" :        "Mr",

                        "Mrs" :       "Ms",

                        "Miss" :      "Ms",

                        "Master" :    "VIP",

                        "Lady" :      "VIP"



                        }

    

    # we map each title

    df['Title'] = df.Title.map(Title_Dictionary)

    return df

    

def BinData(df, feature, bins, group_names):

## Bin Data

 

    # Bin data

    df[feature] = pd.cut(df[feature], bins, labels=group_names)

    # Convert categorical variable into indicator variables

    df = pd.get_dummies(df, columns=[feature])

    

    return df



def FillMissingAge(df):

# Age 

# Fill missing "Age" values with random numbers that fall within the first standard deviation of the mean



    # get average, std, and number of NaN values in titanic_df

    average_age   = df["Age"].mean()

    std_age       = df["Age"].std()

    count_nan_age = df["Age"].isnull().sum()



    # generate random numbers between (mean - std) & (mean + std)

    rand = np.random.randint(average_age - std_age, average_age + std_age, size = count_nan_age)



    # fill NaN values in Age column with random values generated

    df["Age"][np.isnan(df["Age"])] = rand



    # convert from float to int

    df['Age'] = df['Age'].astype(int)

    

    return df



def NewFeature(df):

    ## combine age and gender

    ## combine class and gender

    df['F_0-10'] = df['Sex_female'] * df['Age_0-10']

    df['F_10-60'] = df['Sex_female'] * df['Age_10-60']

    df['F_60-100'] = df['Sex_female'] * df['Age_60-100']

    

    df['M_0-10'] = df['Sex_male'] * df['Age_0-10']

    df['M_10-60'] = df['Sex_male'] * df['Age_10-60']

    df['M_60-100'] = df['Sex_male'] * df['Age_60-100']



    df['F_C1'] = df['Sex_female'] * df['Pclass_1']

    df['F_C2'] = df['Sex_female'] * df['Pclass_2']

    df['F_C3'] = df['Sex_female'] * df['Pclass_3']

    

    df['M_C1'] = df['Sex_male'] * df['Pclass_1']

    df['M_C2'] = df['Sex_male'] * df['Pclass_2']

    df['M_C3'] = df['Sex_male'] * df['Pclass_3']

    

    df['Family_Size'] = df['Parch'] + df['SibSp']

    return df





def RunRandomForest(X_train, Y_train, X_test):

## Random Forests

    

    forest = RandomForestClassifier(max_features='sqrt')

       

    # Tune Hyperparameters through a grid search

    parameter_grid = {'n_estimators': [5,10,15,25,50,100,200,300],

                     'criterion': ['gini','entropy']}



    cross_validation = StratifiedKFold(Y_train, n_folds=5)



    grid_search = GridSearchCV(forest,

                               param_grid=parameter_grid,

                               cv=cross_validation)



    grid_search.fit(X_train, Y_train)



    #print('Best score: {}'.format(grid_search.best_score_))

    #print('Best parameters: {}'.format(grid_search.best_params_))

    

    forest = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'], criterion=grid_search.best_params_['criterion'])

    

    # Build a forest of trees from the training set (X, y)

    forest.fit(X_train, Y_train)

    # Predict class for X

    Y_pred = forest.predict(X_test)    

    # Returns the mean accuracy on the given test data and labels.

    score = forest.score(X_train, Y_train)

    print('Random Forest score:', score)

    # Plot Feature Importance

    PlotFeatureImportance(X_train, forest)

    

    return Y_pred, score, forest



def PlotFeatureImportance(X, random_forest):

    

    importances = random_forest.feature_importances_

    std = np.std([tree.feature_importances_ for tree in random_forest.estimators_],

             axis=0)

    indices = np.argsort(importances)[::-1]

    features = np.array(list(X))

    # Print the feature ranking

    print("Feature ranking:")



    for f in range(X.shape[1]):

        print("%d. %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))



    # Plot the feature importances of the forest

    plt.figure()

    plt.title("Feature importances")

    plt.bar(range(X.shape[1]), importances[indices],

           color="r", yerr=std[indices], align="center")

    plt.xticks(range(X.shape[1]), features[indices])

    plt.xlim([-1, X.shape[1]])

    plt.show()





def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



def ConfusionMatrix(Y_train, Y_pred, train):

    cnf_matrix = confusion_matrix(Y_train, Y_pred)

    np.set_printoptions(precision=2)



    # Plot non-normalized confusion matrix

    plt.figure()

    plot_confusion_matrix(cnf_matrix, classes=list(train),

                          title='Confusion matrix, without normalization')



    # Plot normalized confusion matrix

    plt.figure()

    plot_confusion_matrix(cnf_matrix, classes=list(train), normalize=True,

                          title='Normalized confusion matrix')



    plt.show()



def benchmark(clf):

## Benchmark classifiers

## Modified after: http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#sphx-glr-auto-examples-text-document-classification-20newsgroups-py

    print('_' * 80)

    print("Training: ")

    print(clf)

    t0 = time()

    clf.fit(X_train, Y_train)

    train_time = time() - t0

    print("train time: %0.3fs" % train_time)



    t0 = time()

    Y_pred = clf.predict(X_test)

    test_time = time() - t0

    print("test time:  %0.3fs" % test_time)



    score = clf.score(X_train, Y_train)

    print("Training accuracy:   %0.3f" % score)



    if hasattr(clf, 'coef_'):

        print("dimensionality: %d" % clf.coef_.shape[1])

        print("density: %f" % density(clf.coef_))



    print("classification report:")

    print(metrics.classification_report(Y_test, Y_pred, target_names=target_names))



    print("confusion matrix:")

    print(metrics.confusion_matrix(Y_test, Y_pred))



    print()

    clf_descr = str(clf).split('(')[0]

    return clf_descr, score, train_time, test_time





def PlotClf(results):

## make some plots



    indices = np.arange(len(results))



    results = [[x[i] for x in results] for i in range(4)]



    clf_names, score, training_time, test_time = results

    training_time = np.array(training_time) / np.max(training_time)

    test_time = np.array(test_time) / np.max(test_time)



    plt.figure(figsize=(12, 8))

    plt.title("Score")

    plt.barh(indices, score, .2, label="score", color='navy')

    plt.barh(indices + .3, training_time, .2, label="training time",

             color='c')

    plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')

    plt.yticks(())

    plt.legend(loc='best')

    plt.subplots_adjust(left=.25)

    plt.subplots_adjust(top=.95)

    plt.subplots_adjust(bottom=.05)



    for i, c in zip(indices, clf_names):

        plt.text(-.3, i, c)



    plt.show()
###################

# Begin Execution #

###################



# Read Data

test, train = ReadData()



# Prepare Data

train_prep = PrepareData(train.copy())

test_prep = PrepareData(test.copy())

target_names = train_prep.columns



# Split traning set

x = train_prep.drop("Survived",axis=1)

y = train_prep["Survived"]

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=42)



# FSurvival Stats

#SurvivalStats(train_prep)



results = []

for clf, name in (

        (RidgeClassifier(alpha=5.0, tol=1e-2, solver="lsqr"), "Ridge Classifier"),

        (Perceptron(n_iter=50), "Perceptron"),

        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),

        (KNeighborsClassifier(n_neighbors=10), "kNN"),

        (RandomForestClassifier(n_estimators=100), "Random forest")):

    print('=' * 80)

    print(name)

    results.append(benchmark(clf))



for penalty in ["l2", "l1"]:

    print('=' * 80)

    print("%s penalty" % penalty.upper())

    # Train Liblinear model

    results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,

                                            dual=False, tol=1e-3)))



    # Train SGD model

    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,

                                           penalty=penalty)))



# Train SGD with Elastic Net penalty

print('=' * 80)

print("Elastic-Net penalty")

results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,

                                       penalty="elasticnet")))



# Train NearestCentroid without threshold

print('=' * 80)

print("NearestCentroid (aka Rocchio classifier)")

results.append(benchmark(NearestCentroid()))



# Train sparse Naive Bayes classifiers

print('=' * 80)

print("Naive Bayes")

results.append(benchmark(MultinomialNB(alpha=.01)))

results.append(benchmark(BernoulliNB(alpha=.01)))



print('=' * 80)

print("LinearSVC with L1-based feature selection")

# The smaller C, the stronger the regularization.

# The more regularization, the more sparsity.

results.append(benchmark(Pipeline([

  ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),

  ('classification', LinearSVC())

])))



PlotClf(results)



# Define training and testing sets

#X_train = train_prep.drop("Survived",axis=1)

#Y_train = train_prep["Survived"]

X_test  = test_prep





# Logistic Regression

#Y_pred, score, coeff_df = RunLogReg(X_train, Y_train, X_test)



# Ridge Regression

#Y_pred, score, coeff_df = RunRidgeReg(X_train, Y_train, X_test)







# Compute confusion matrix

#ConfusionMatrix(Y_train, Y_pred, train)



# Random Forest

Y_pred, rf_score, random_forest = RunRandomForest(X_train, Y_train, X_test)



# Write predictions

test['Survived'] = Y_pred

test.to_csv('output.csv',index=True, columns=['Survived'], header=True)