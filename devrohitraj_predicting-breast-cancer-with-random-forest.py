import numpy as np

import pandas as pd



from time import time



%matplotlib inline 

import matplotlib.pyplot as plt

from matplotlib import pyplot

import seaborn as sns



from sklearn.metrics import f1_score

from sklearn import metrics



# For splitting dataset

from sklearn.cross_validation import ShuffleSplit, train_test_split



import sklearn.learning_curve as curves

from sklearn.learning_curve import validation_curve



# k-fold cross validation

from sklearn.cross_validation import KFold, cross_val_score



# Import sklearn models

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier
all_data = pd.read_csv("../input/data.csv")

all_data.head()
all_data['diagnosis'].unique()
all_data = all_data.drop(['id', 'Unnamed: 32'], axis = 1)

all_data['diagnosis'] = all_data['diagnosis'].map({'M':1,'B':0})

all_data.head()
target = all_data['diagnosis']

features = all_data.drop('diagnosis', axis = 1)

all_features = list(features.columns[0:11])

features.head()
target.unique()
features.describe()
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size= .3,random_state=0)
Kfold = KFold(len(all_data),n_folds = 10,shuffle = False)
def train_and_predict_model(model, model_name, X_train, X_test, y_train, y_test, selected_cols):

    t0 = time()

    model.fit(X_train[selected_cols], y_train)

    train_time = time() - t0

    

    t1 = time()

    pred = model.predict(X_test[selected_cols])

    predict_time = time() - t1

    

    score = f1_score(y_test, pred)

    

    print ("f1_score of {} is {}".format(model_name, score))

    print ("Accuracy of {} is {}".format(model_name, metrics.accuracy_score(y_test, pred)))

    print ("cross_val_score of {} is {}".format(model_name, cross_val_score(model, features[selected_cols], target , cv = 10).mean()))

    

    print ("Time taken to train {} is {}".format(model_name, train_time))

    print ("Time taken to predict {} is {}".format(model_name, predict_time))
def ModelLearning(X, y):

    """ Calculates the performance of several models with varying sizes of training data.

        The learning and testing scores for each model are then plotted. """

    

    # Create 10 cross-validation sets for training and testing

    cv = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.2, random_state = 0)



    # Generate the training set sizes increasing by 50

    train_sizes = np.rint(np.linspace(1, X.shape[0]*0.8 - 1, 9)).astype(int)



    # Create the figure window

    fig = plt.figure(figsize=(10,7))



    # Create three different models based on max_depth

    for k, depth in enumerate([1,3,6,10]):

        

        # Create a Random Forest at max_depth = depth

        classifier = RandomForestClassifier(max_depth = depth)



        # Calculate the training and testing scores

        sizes, train_scores, test_scores = curves.learning_curve(classifier, X, y, \

            cv = cv, train_sizes = train_sizes, scoring = 'accuracy')

        

        # Find the mean and standard deviation for smoothing

        train_std = np.std(train_scores, axis = 1)

        train_mean = np.mean(train_scores, axis = 1)

        test_std = np.std(test_scores, axis = 1)

        test_mean = np.mean(test_scores, axis = 1)



        # Subplot the learning curve 

        ax = fig.add_subplot(2, 2, k+1)

        ax.plot(sizes, train_mean, 'o-', color = 'r', label = 'Training Score')

        ax.plot(sizes, test_mean, 'o-', color = 'g', label = 'Testing Score')

        ax.fill_between(sizes, train_mean - train_std, \

            train_mean + train_std, alpha = 0.15, color = 'r')

        ax.fill_between(sizes, test_mean - test_std, \

            test_mean + test_std, alpha = 0.15, color = 'g')

        

        # Labels

        ax.set_title('max_depth = %s'%(depth))

        ax.set_xlabel('Number of Training Points')

        ax.set_ylabel('Score')

        ax.set_xlim([0, X.shape[0]*0.8])

        ax.set_ylim([-0.05, 1.05])

    

    # Visual aesthetics

    ax.legend(bbox_to_anchor=(1.05, 2.05), loc='lower left', borderaxespad = 0.)

    fig.suptitle('Random Forest Learning Performances', fontsize = 16, y = 1.03)

    fig.tight_layout()

    fig.show()





def ModelComplexity(X, y):

    """ Calculates the performance of the model as model complexity increases.

        The learning and testing errors rates are then plotted. """

    

    # Create 10 cross-validation sets for training and testing

    cv = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.2, random_state = 0)



    # Vary the max_depth parameter from 1 to 10

    max_depth = np.arange(1,11)



    # Calculate the training and testing scores

    train_scores, test_scores = curves.validation_curve(RandomForestClassifier(), X, y, \

        param_name = "max_depth", param_range = max_depth, cv = cv, scoring = 'accuracy')



    # Find the mean and standard deviation for smoothing

    train_mean = np.mean(train_scores, axis=1)

    train_std = np.std(train_scores, axis=1)

    test_mean = np.mean(test_scores, axis=1)

    test_std = np.std(test_scores, axis=1)



    # Plot the validation curve

    plt.figure(figsize=(7, 5))

    plt.title('Random Forest Complexity Performance')

    plt.plot(max_depth, train_mean, 'o-', color = 'r', label = 'Training Score')

    plt.plot(max_depth, test_mean, 'o-', color = 'g', label = 'Validation Score')

    plt.fill_between(max_depth, train_mean - train_std, \

        train_mean + train_std, alpha = 0.15, color = 'r')

    plt.fill_between(max_depth, test_mean - test_std, \

        test_mean + test_std, alpha = 0.15, color = 'g')

    

    # Visual aesthetics

    plt.legend(loc = 'lower right')

    plt.xlabel('Maximum Depth')

    plt.ylabel('Score')

    plt.ylim([-0.05,1.05])

    plt.show()

    

    

def ModelValidation(X, y):

    param_range = np.arange(1,11)

    train_scores, test_scores = validation_curve(

        RandomForestClassifier(), X, y, param_name="max_depth", param_range=param_range,

        cv=10, scoring="accuracy", n_jobs=1)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)



    plt.title("Validation Curve with Random Forest")

    plt.xlabel("$\max_depth$")

    plt.ylabel("Score")

    plt.ylim(0.0, 1.1)

    lw = 2

    plt.semilogx(param_range, train_scores_mean, label="Training score",

                 color="darkorange", lw=lw)

    plt.fill_between(param_range, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.2,

                     color="darkorange", lw=lw)

    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",

                 color="navy", lw=lw)

    plt.fill_between(param_range, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.2,

                     color="navy", lw=lw)

    plt.legend(loc="best")

    plt.show()
ModelLearning(features[all_features], target)

ModelComplexity(features[all_features], target)

ModelValidation(features[all_features], target)
total_features = list(features.columns)

Forest = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)

train_and_predict_model(Forest, 'Random Forest', X_train, X_test, y_train, y_test, all_features)
dt = DecisionTreeClassifier(max_depth=5)

lr = LogisticRegression()

rfc = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)

sv = SVC(kernel='linear',C=1)

boost = AdaBoostClassifier()
for clf, name in [(dt, 'DecisionTree'),

                  (lr, 'Logistic'),

                  (rfc, 'Random Forest'),

                  (sv, 'Support Vector Classification'),

                  (boost, 'AdaBoost Classifier')]:

    print ("---------------------------------------------------------------------")

    train_and_predict_model(clf, name, X_train, X_test, y_train, y_test, all_features)

    
importance = Forest.feature_importances_

names = list(features)

pyplot.bar(range(len(importance)), importance)

print (sorted(zip(map(lambda x: round(x, 4), importance), names), 

             reverse=True))

pyplot.show()
selected_features = ['radius_mean', 'concavity_mean', 'compactness_mean', 'smoothness_mean', 'concave points_mean', 'perimeter_mean', 'radius_se']
sns.set(style="whitegrid", color_codes=True)

fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(ncols=7, sharey=True)

tups = (ax1, ax2, ax3, ax4, ax5, ax6, ax7)

for i in range(len(selected_features)):

    sns.stripplot(x='diagnosis', y=selected_features[i], data=all_data, ax = tups[i]);
sns.violinplot(x="diagnosis", y="radius_mean", data=all_data, inner=None)

sns.swarmplot(x="diagnosis", y="radius_mean", data=all_data, color="w", alpha=.5);
for clf, name in [(dt, 'DecisionTree'),

                  (lr, 'Logistic'),

                  (rfc, 'Random Forest'),

                  (sv, 'Support Vector Classification'),

                  (boost, 'AdaBoost Classifier')]:

    print ("---------------------------------------------------------------------")

    train_and_predict_model(clf, name, X_train, X_test, y_train, y_test, selected_features)
ModelLearning(features[selected_features], target)

ModelComplexity(features[selected_features], target)

ModelValidation(features[selected_features], target)