from subprocess import check_output

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import itertools

import graphviz

import random

import math

import matplotlib.pyplot as plt # plotting

from scipy import stats 

from scipy.stats import chi2_contingency

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn import tree

from sklearn.utils import resample



%matplotlib inline



print(check_output(["ls", "../input"]).decode("utf8"))

raw_data = pd.read_csv('../input/mushrooms.csv')

print(raw_data.describe())
raw_data.head()
# Helper functions



def replace_binary(field_value, possible_values):

    """ Replace an arbitray two label field with a binary value

    possible_values: list of values to check against

    """

    if field_value == possible_values[0]:

        return 0

    elif field_value == possible_values[1]:

        return 1

    else:

        raise KeyError



def plot_ConfusionMatrix(y_test,y_pred, classes,

                          normalize=False, size=(8,6),

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    # Compute confusion matrix

    cm = confusion_matrix(y_test, y_pred)

    np.set_printoptions(precision=2)



    # Plot non-normalized confusion matrix

    plt.figure(figsize=size)

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



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

    plt.show()
two_label_columns = { 'class': ['e','p'], # Note: This will make edible=0, poisonous=

                      'bruises': ['f','t'],

                      'gill-attachment': ['f','a'],

                      'gill-spacing': ['c','w'],

                      'gill-size': ['b','n'],

                      'stalk-shape': ['t','e'],

                      'veil-type': ['p','n'] }

multi_label_columns = ['cap-shape','cap-surface', 'cap-color', 'odor', 'gill-color',

                       'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',

                       'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-color',

                       'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat' ]



# Expand each multi-label column into n columns, where the column has n possible values.

data = pd.get_dummies(raw_data, columns=multi_label_columns)



# Would could also use get_dummies for the two-label columns, but this is a good introduction to

# using a callable to transform or create new features.

for column, values in two_label_columns.items():

    data[column] = raw_data[column].apply(replace_binary, possible_values=values)



# Show the data now

data.head()
# Create a DataFrame that will house 

chisquare_tests = pd.DataFrame(columns=['Feature','Test Statistic','P-Value'])



# A key assumption of the Chi Square test is that the expected frequency of each

# category will be great than 5. We perform a test to highlight which categories

# violate this.

print('Variable Frequency Check:')

for col in data.columns.drop('class'):

    ct = pd.crosstab(data[col],data['class'],margins=True)

    if (ct['All']/2<5).any():

        print(col,' is too small')

    else:

        chi2, p, dof, expted = chi2_contingency(ct)

        chisquare_tests = chisquare_tests.append(

            {'Feature':col,

             'Test Statistic':chi2,

             'P-Value':p,

             'DOF':dof }, 

            ignore_index=True )



# Refactor the dataframe

chisquare_tests = chisquare_tests.set_index('Feature')

chisquare_tests = chisquare_tests.sort_values(by='P-Value')



# Plot the results in a bar plot

# "Confidence Level" is just 1-pvalue. This is probably made up, but it's easier 

# for non-statistical types to understand

max_stat = chisquare_tests['Test Statistic'].max()

fig = plt.figure(figsize=(8,24))

ax = chisquare_tests['Test Statistic'].plot(kind='barh',color='C1')

ax.set_title('Chi Square Test Results')

ax.set_xlabel('Test Statistic')

ax.set_xlim((0,max_stat+3))

ax.set_ylim((-1,chisquare_tests.shape[0]+0.25))

plt.text(max_stat+0.5, chisquare_tests.shape[0]-0.5,'Confidence Level')

for i in np.arange(chisquare_tests.shape[0]):

    confidence = (1-chisquare_tests.iloc[i]['P-Value'])

    plt.text(max_stat+2, i,'{0:0.1%}'.format(confidence))
X_train, X_test, y_train, y_test = train_test_split(data.drop('class',axis=1), 

                                                    data['class'], 

                                                    test_size=0.33, random_state=25)



RFclf = RandomForestClassifier(n_estimators=100,

                             max_depth=4, 

                             random_state=25)



RFclf.fit(X_train, y_train)



print('Random Forest Classifier Score on Train Set: {:0.3f}'.format(RFclf.score(X_train, y_train)))

print('Random Forest Classifier Score on Test Set: {:0.3f}'.format(RFclf.score(X_test, y_test)))
DTclf = DecisionTreeClassifier(max_depth=3,random_state=43)

DTclf.fit(X_train, y_train)



print('Decision Tree Classifier Score on Train Set: {:0.3f}'.format(DTclf.score(X_train, y_train)))

print('Decision Tree Classifier Score on Test Set: {:0.3f}'.format(DTclf.score(X_test, y_test)))



feature_names = data.columns.drop('class')

class_names = ['Edible','Poisonous']



dot_data = tree.export_graphviz(DTclf, out_file=None, 

                                    feature_names=feature_names,

                                    class_names=class_names, 

                                    impurity=False,proportion=True)

graph = graphviz.Source(dot_data)



DT_y_pred = DTclf.predict(X_test)
graph
plot_ConfusionMatrix(y_test, DT_y_pred, 

                     class_names,

                     normalize=True, 

                     title='Mushroom Identification Confusion matrix')
LRclf = LogisticRegression(C=0.1)

LRclf.fit(X_train, y_train)



print('Logistic Regression Score on Train Set: {:0.3f}'.format(LRclf.score(X_train, y_train)))

print('Logistic Regression Score on Test Set: {:0.3f}'.format(LRclf.score(X_test, y_test)))
def bootstrap_predict(classifiers, X):

    """

    Perform a prediction across a population of classifiers

    Arguments:

    classifiers - list of classifiers

    X - DataFrame of samples to predict

    Returns:

    List of predictions

    """

    preds = np.zeros(len(classifiers))

    for clf,i in zip(classifiers, np.arange(len(classifiers))):

        # Predict_proba returns probabilities for both classes (sum is 1)

        # Return the first value, referring to probability of the mushroom being edible

        pred = clf.predict_proba(X).ravel()[0]

        preds[i]= pred



    # confidence intervals

    alpha = 0.95

    p = ((1.0-alpha)/2.0) * 100

    lower = max(0.0, np.percentile(preds, p))

    p = (alpha+((1.0-alpha)/2.0)) * 100

    upper = min(1.0, np.percentile(preds, p))

    return preds.mean(), (lower,upper)





n_iterations = 1000

classifiers = []

scores = np.zeros(n_iterations)

for i in range(n_iterations):

    # Each iteration selects a new sample set from the data

    X, y = resample(data.drop('class',axis=1), data['class'], random_state=i)

    # For model scoring purposes, we still need training and testing sets

    bsX_train, bsX_test, bsy_train, bsy_test = train_test_split(X, y, test_size=0.33, random_state=i)

    # Train and test the classifier

    clf = LogisticRegression(C=0.1)

    clf.fit(bsX_train, bsy_train)

    classifiers.append(clf)

    scores[i] = clf.score(bsX_train, bsy_train)

    

print('Classifiers Average Score: {:0.3f} '.format(scores.mean()))

print('Classifiers Score Std Dev: {:0.3f} '.format(np.std(scores)))



# Here's an example of how to predict a value, and provide 

pred, CI = bootstrap_predict(classifiers, X.iloc[1].values.reshape(1, -1))



print('Prediction Expected Value: {:0.3f} '.format(pred))

print('Prediction 95% Confidence Interval: ({0:0.3f},{1:0.3f})'.format(CI[0],CI[1]))