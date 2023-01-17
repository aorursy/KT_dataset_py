import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from scipy import stats

from scipy.stats import randint



# prep

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.datasets import make_classification

from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler



# models

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier



# Validation libraries

from sklearn import metrics

from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_curve

from sklearn.model_selection import cross_val_score



#Neural Network

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import RandomizedSearchCV



#Bagging

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier



#Naive bayes

from sklearn.naive_bayes import GaussianNB 



#Stacking

from mlxtend.classifier import StackingClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



#reading in CSV's from a file path

train_df = pd.read_csv('../input/mental-health-of-youths/survey.csv')





#Pandas: whats the data row count?

print(train_df.shape)

    

#Pandas: whats the distribution of the data?

print(train_df.describe())

    

#Pandas: What types of data do i have?

print(train_df.info())
#dealing with missing data

#Let’s get rid of the variables "Timestamp",“comments”, “state” just to make our lives easier.

train_df = train_df.drop(['comments'], axis= 1)

train_df = train_df.drop(['state'], axis= 1)

train_df = train_df.drop(['Timestamp'], axis= 1)



train_df.isnull().sum().max() #just checking that there's no missing data missing...

train_df.head(5)
# Assign default values for each data type

defaultInt = 0

defaultString = 'NaN'

defaultFloat = 0.0



# Create lists by data tpe

intFeatures = ['Age']

stringFeatures = ['Gender', 'Country', 'breakup_loss', 'family_history', 'treatment', 'abuse',

                 'no_students', 'online_offline', 'coed_noncoed', 'anonymity', 'confidence_selfesteem', 'mental_health_consequence',

                 'phys_health_consequence', 'classmates', 'counsellor', 'mental_health_interview', 'phys_health_interview',

                 'mental_vs_physical', 'obs_consequence', 'food', 'sleep', 'wellness_program',

                 'seek_help']

floatFeatures = []



# Clean the NaN's

for feature in train_df:

    if feature in intFeatures:

        train_df[feature] = train_df[feature].fillna(defaultInt)

    elif feature in stringFeatures:

        train_df[feature] = train_df[feature].fillna(defaultString)

    elif feature in floatFeatures:

        train_df[feature] = train_df[feature].fillna(defaultFloat)

    else:

        print('Error: Feature %s not recognized.' % feature)

train_df.head(5) 
#clean 'Gender'

#Slower case all columm's elements

gender = train_df['Gender'].str.lower()

#print(gender)



#Select unique elements

gender = train_df['Gender'].unique()



#Made gender groups

male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "Cis Male", "cis male"]

trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]           

female_str = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]



for (row, col) in train_df.iterrows():



    if str.lower(col.Gender) in male_str:

        train_df['Gender'].replace(to_replace=col.Gender, value='male', inplace=True)



    if str.lower(col.Gender) in female_str:

        train_df['Gender'].replace(to_replace=col.Gender, value='female', inplace=True)



    if str.lower(col.Gender) in trans_str:

        train_df['Gender'].replace(to_replace=col.Gender, value='trans', inplace=True)



#Get rid of bullshit

stk_list = ['A little about you', 'p']

train_df = train_df[~train_df['Gender'].isin(stk_list)]



print(train_df['Gender'].unique())
#complete missing age with mean

train_df['Age'].fillna(train_df['Age'].median(), inplace = True)



# Fill with media() values < 18 and > 120

s = pd.Series(train_df['Age'])

s[s<18] = train_df['Age'].median()

train_df['Age'] = s

s = pd.Series(train_df['Age'])

s[s>120] = train_df['Age'].median()

train_df['Age'] = s



#Ranges of Age

train_df['age_range'] = pd.cut(train_df['Age'], [0,20,30,65,100], labels=["0-20", "21-30", "31-65", "66-100"], include_lowest=True)
#There are only 0.014% of them who had a break up or loss in the family so let's change NaN to NOT self_employed

#Replace "NaN" string from defaultString

train_df['breakup_loss'] = train_df['breakup_loss'].replace([defaultString], 'No')

print(train_df['breakup_loss'].unique())
#There are only 0.20% who got abused so let's change NaN to "Don't know"

#Replace "NaN" string from defaultString



train_df['abuse'] = train_df['abuse'].replace([defaultString], 'Don\'t know' )

print(train_df['abuse'].unique())
#Encoding data

labelDict = {}

for feature in train_df:

    le = preprocessing.LabelEncoder()

    le.fit(train_df[feature])

    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    train_df[feature] = le.transform(train_df[feature])

    # Get labels

    labelKey = 'label_' + feature

    labelValue = [*le_name_mapping]

    labelDict[labelKey] =labelValue

    

for key, value in labelDict.items():     

    print(key, value)



#Get rid of 'Country'

train_df = train_df.drop(['Country'], axis= 1)

train_df.head()
#missing data

total = train_df.isnull().sum().sort_values(ascending=False)

percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)

print(missing_data)
#correlation matrix

corrmat = train_df.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);

plt.show()



#treatment correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'treatment')['treatment'].index

cm = np.corrcoef(train_df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
# Distribiution and density by Age

plt.figure(figsize=(12,8))

sns.distplot(train_df["Age"], bins=24)

plt.title("Distribuition and density by Age")

plt.xlabel("Age")
# Separate by treatment or not



g = sns.FacetGrid(train_df, col='treatment', height=5)

g = g.map(sns.distplot, "Age")
# Let see how many people has been treated

plt.figure(figsize=(12,8))

labels = labelDict['label_Gender']

g = sns.countplot(x="treatment", data=train_df)

g.set_xticklabels(labels)



plt.title('Total Distribuition by treated or not')
o = labelDict['label_age_range']



g = sns.catplot(x="age_range", y="treatment", hue="Gender", data=train_df, kind="bar",  ci=None, height=5, aspect=2, legend_out = True)

g.set_xticklabels(o)



plt.title('Probability of mental health condition')

plt.ylabel('Probability x 100')

plt.xlabel('Age')

# replace legend labels



new_labels = labelDict['label_Gender']

for t, l in zip(g._legend.texts, new_labels): t.set_text(l)



# Positioning the legend

g.fig.subplots_adjust(top=0.9,right=0.8)



plt.show()

o = labelDict['label_family_history']

g = sns.catplot(x="family_history", y="treatment", hue="Gender", data=train_df, kind="bar", ci=None, height=5, aspect=2, legend_out = True)

g.set_xticklabels(o)

plt.title('Probability of mental health condition')

plt.ylabel('Probability x 100')

plt.xlabel('Family History')



# replace legend labels

new_labels = labelDict['label_Gender']

for t, l in zip(g._legend.texts, new_labels): t.set_text(l)



# Positioning the legend

g.fig.subplots_adjust(top=0.9,right=0.8)



plt.show()
o = labelDict['label_sleep']

g = sns.catplot(x="sleep", y="treatment", hue="Gender", data=train_df, kind="bar", ci=None, height=5, aspect=2, legend_out = True)

g.set_xticklabels(o)

plt.title('Probability of mental health condition')

plt.ylabel('Probability x 100')

plt.xlabel('Sleep')



# replace legend labels

new_labels = labelDict['label_Gender']

for t, l in zip(g._legend.texts, new_labels): t.set_text(l)



# Positioning the legend

g.fig.subplots_adjust(top=0.9,right=0.8)

plt.show()
o = labelDict['label_food']

g = sns.catplot(x="food", y="treatment", hue="Gender", data=train_df, kind="bar", ci=None, height=5, aspect=2, legend_out = True)

g.set_xticklabels(o)

plt.title('Probability of mental health condition')

plt.ylabel('Probability x 100')

plt.xlabel('Food')



# replace legend labels

new_labels = labelDict['label_Gender']

for t, l in zip(g._legend.texts, new_labels): t.set_text(l)



# Positioning the legend

g.fig.subplots_adjust(top=0.9,right=0.8)

plt.show()
o = labelDict['label_abuse']

g = sns.catplot(x="abuse", y="treatment", hue="Gender", data=train_df, kind="bar", ci=None, height=5, aspect=2, legend_out = True)

g.set_xticklabels(o)

plt.title('Probability of mental health condition')

plt.ylabel('Probability x 100')

plt.xlabel('Abuse')



# replace legend labels

new_labels = labelDict['label_Gender']

for t, l in zip(g._legend.texts, new_labels): t.set_text(l)



# Positioning the legend

g.fig.subplots_adjust(top=0.9,right=0.8)

plt.show()
# Scaling Age

scaler = MinMaxScaler()

train_df['Age'] = scaler.fit_transform(train_df[['Age']])

train_df.head()
# define X and y

feature_cols = ['Age', 'Gender', 'family_history', 'sleep', 'food', 'anonymity', 'confidence_selfesteem', 'abuse']

X = train_df[feature_cols]

y = train_df.treatment



# split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)



# Create dictionaries for final graph

# Use: methodDict['Stacking'] = accuracy_score

methodDict = {}

rmseDict = ()
# Build a forest and compute the feature importances

forest = ExtraTreesClassifier(n_estimators=250,

                              random_state=0)



forest.fit(X, y)

importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



labels = []

for f in range(X.shape[1]):

    labels.append(feature_cols[f])      

    

# Plot the feature importances of the forest

plt.figure(figsize=(12,8))

plt.title("Feature importance")

plt.bar(range(X.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(X.shape[1]), labels, rotation='vertical')

plt.xlim([-1, X.shape[1]])

plt.show()
def evalClassModel(model, y_test, y_pred_class, plot=False):

    #Classification accuracy: percentage of correct predictions

    # calculate accuracy

    print('Accuracy:', metrics.accuracy_score(y_test, y_pred_class))

    

    #Null accuracy: accuracy that could be achieved by always predicting the most frequent class

    # examine the class distribution of the testing set (using a Pandas Series method)

    print('Null accuracy:\n', y_test.value_counts())

    

    # calculate the percentage of ones

    print('Percentage of ones:', y_test.mean())

    

    # calculate the percentage of zeros

    print('Percentage of zeros:',1 - y_test.mean())

    

    #Comparing the true and predicted response values

    print('True:', y_test.values[0:25])

    print('Pred:', y_pred_class[0:25])

    

    #Conclusion:

    #Classification accuracy is the easiest classification metric to understand

    #But, it does not tell you the underlying distribution of response values

    #And, it does not tell you what "types" of errors your classifier is making

    

    #Confusion matrix

    # save confusion matrix and slice into four pieces

    confusion = metrics.confusion_matrix(y_test, y_pred_class)

    #[row, column]

    TP = confusion[1, 1]

    TN = confusion[0, 0]

    FP = confusion[0, 1]

    FN = confusion[1, 0]

    

    # visualize Confusion Matrix

    sns.heatmap(confusion,annot=True,fmt="d") 

    plt.title('Confusion Matrix')

    plt.xlabel('Predicted')

    plt.ylabel('Actual')

    plt.show()

    

    #Metrics computed from a confusion matrix

    #Classification Accuracy: Overall, how often is the classifier correct?

    accuracy = metrics.accuracy_score(y_test, y_pred_class)

    print('Classification Accuracy:', accuracy)

    

    #Classification Error: Overall, how often is the classifier incorrect?

    print('Classification Error:', 1 - metrics.accuracy_score(y_test, y_pred_class))

    

    #False Positive Rate: When the actual value is negative, how often is the prediction incorrect?

    false_positive_rate = FP / float(TN + FP)

    print('False Positive Rate:', false_positive_rate)

    

    #Precision: When a positive value is predicted, how often is the prediction correct?

    print('Precision:', metrics.precision_score(y_test, y_pred_class))

    

    

    # IMPORTANT: first argument is true values, second argument is predicted probabilities

    print('AUC Score:', metrics.roc_auc_score(y_test, y_pred_class))

    

    # calculate cross-validated AUC

    print('Cross-validated AUC:', cross_val_score(model, X, y, cv=10, scoring='roc_auc').mean())

    

    ##########################################

    #Adjusting the classification threshold

    ##########################################

    # print the first 10 predicted responses

    # 1D array (vector) of binary values (0, 1)

    print('First 10 predicted responses:\n', model.predict(X_test)[0:10])



    # print the first 10 predicted probabilities of class membership

    print('First 10 predicted probabilities of class members:\n', model.predict_proba(X_test)[0:10])



    # print the first 10 predicted probabilities for class 1

    model.predict_proba(X_test)[0:10, 1]

    

    # store the predicted probabilities for class 1

    y_pred_prob = model.predict_proba(X_test)[:, 1]

    

    if plot == True:

        # histogram of predicted probabilities

        # adjust the font size 

        plt.rcParams['font.size'] = 12

        # 8 bins

        plt.hist(y_pred_prob, bins=8)

        

        # x-axis limit from 0 to 1

        plt.xlim(0,1)

        plt.title('Histogram of predicted probabilities')

        plt.xlabel('Predicted probability of treatment')

        plt.ylabel('Frequency')

    

    

    # predict treatment if the predicted probability is greater than 0.3

    # it will return 1 for all values above 0.3 and 0 otherwise

    # results are 2D so we slice out the first column

    y_pred_prob = y_pred_prob.reshape(-1,1) 

    y_pred_class = binarize(y_pred_prob, 0.3)[0]

    

    # print the first 10 predicted probabilities

    print('First 10 predicted probabilities:\n', y_pred_prob[0:10])

    

    ##########################################

    #ROC Curves and Area Under the Curve (AUC)

    ##########################################

    

    #Question: Wouldn't it be nice if we could see how sensitivity and specificity are affected by various thresholds, without actually changing the threshold?

    #Answer: Plot the ROC curve!

    

    

    #AUC is the percentage of the ROC plot that is underneath the curve

    #Higher value = better classifier

    roc_auc = metrics.roc_auc_score(y_test, y_pred_prob)

    

    



    # IMPORTANT: first argument is true values, second argument is predicted probabilities

    # we pass y_test and y_pred_prob

    # we do not use y_pred_class, because it will give incorrect results without generating an error

    # roc_curve returns 3 objects fpr, tpr, thresholds

    # fpr: false positive rate

    # tpr: true positive rate

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)

    if plot == True:

        plt.figure()

        

        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)

        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

        plt.xlim([0.0, 1.0])

        plt.ylim([0.0, 1.0])

        plt.rcParams['font.size'] = 12

        plt.title('ROC curve for treatment classifier')

        plt.xlabel('False Positive Rate (1 - Specificity)')

        plt.ylabel('True Positive Rate (Sensitivity)')

        plt.legend(loc="lower right")

        plt.show()

    

    # define a function that accepts a threshold and prints sensitivity and specificity

    def evaluate_threshold(threshold):

        #Sensitivity: When the actual value is positive, how often is the prediction correct?

        #Specificity: When the actual value is negative, how often is the prediction correct?print('Sensitivity for ' + str(threshold) + ' :', tpr[thresholds > threshold][-1])

        print('Specificity for ' + str(threshold) + ' :', 1 - fpr[thresholds > threshold][-1])



    # One way of setting threshold

    predict_mine = np.where(y_pred_prob > 0.50, 1, 0)

    confusion = metrics.confusion_matrix(y_test, predict_mine)

    print(confusion)

    

    

    

    return accuracy
##########################################

# Tuning with cross validation score

##########################################

def tuningCV(knn):

    

    # search for an optimal value of K for KNN

    k_range = list(range(1, 31))

    k_scores = []

    for k in k_range:

        knn = KNeighborsClassifier(n_neighbors=k)

        scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')

        k_scores.append(scores.mean())

    print(k_scores)

    # plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)

    plt.plot(k_range, k_scores)

    plt.xlabel('Value of K for KNN')

    plt.ylabel('Cross-Validated Accuracy')

    plt.show()
def tuningGridSerach(knn):

    #More efficient parameter tuning using GridSearchCV

    # define the parameter values that should be searched

    k_range = list(range(1, 31))

    print(k_range)

    

    # create a parameter grid: map the parameter names to the values that should be searched

    param_grid = dict(n_neighbors=k_range)

    print(param_grid)

    

    # instantiate the grid

    grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')



    # fit the grid with data

    grid.fit(X, y)

    

    # view the complete results (list of named tuples)

    grid.grid_scores_

    

    # examine the first tuple

    print(grid.grid_scores_[0].parameters)

    print(grid.grid_scores_[0].cv_validation_scores)

    print(grid.grid_scores_[0].mean_validation_score)

    

    # create a list of the mean scores only

    grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]

    print(grid_mean_scores)

    

    # plot the results

    plt.plot(k_range, grid_mean_scores)

    plt.xlabel('Value of K for KNN')

    plt.ylabel('Cross-Validated Accuracy')

    plt.show()

    

    # examine the best model

    print('GridSearch best score', grid.best_score_)

    print('GridSearch best params', grid.best_params_)

    print('GridSearch best estimator', grid.best_estimator_)
def tuningRandomizedSearchCV(model, param_dist):

    #Searching multiple parameters simultaneously

    # n_iter controls the number of searches

    rand = RandomizedSearchCV(model, param_dist, cv=10, scoring='accuracy', n_iter=10, random_state=5)

    rand.fit(X, y)

    rand.cv_results_

    

    # examine the best model

    print('Rand. Best Score: ', rand.best_score_)

    print('Rand. Best Params: ', rand.best_params_)

    

    # run RandomizedSearchCV 20 times (with n_iter=10) and record the best score

    best_scores = []

    for _ in range(20):

        rand = RandomizedSearchCV(model, param_dist, cv=10, scoring='accuracy', n_iter=10)

        rand.fit(X, y)

        best_scores.append(round(rand.best_score_, 3))

    print(best_scores)
def tuningMultParam(knn):

    

    #Searching multiple parameters simultaneously

    # define the parameter values that should be searched

    k_range = list(range(1, 31))

    weight_options = ['uniform', 'distance']

        

    # create a parameter grid: map the parameter names to the values that should be searched

    param_grid = dict(n_neighbors=k_range, weights=weight_options)

    print(param_grid) 

    

    # instantiate and fit the grid

    grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')

    grid.fit(X, y) 

    

    # view the complete results

    print(grid.grid_scores_)

    

    # examine the best model

    print('Multiparam. Best Score: ', grid.best_score_)

    print('Multiparam. Best Params: ', grid.best_params_)
def logisticRegression():

    # train a logistic regression model on the training set

    logreg = LogisticRegression()

    logreg.fit(X_train, y_train)

    

    # make class predictions for the testing set

    y_pred_class = logreg.predict(X_test)

    

    print('########### Logistic Regression ###############')

    

    accuracy_score = evalClassModel(logreg, y_test, y_pred_class, True)

    

    #Data for final graph

    methodDict['Log. Regres.'] = accuracy_score * 100
logisticRegression()
def Knn():

    # Calculating the best parameters

    knn = KNeighborsClassifier(n_neighbors=5)

    

    # From https://github.com/justmarkham/scikit-learn-videos/blob/master/08_grid_search.ipynb

    #tuningCV(knn)

    #tuningGridSerach(knn)

    #tuningMultParam(knn)

    

    # define the parameter values that should be searched

    k_range = list(range(1, 31))

    weight_options = ['uniform', 'distance']

    

    # specify "parameter distributions" rather than a "parameter grid"

    param_dist = dict(n_neighbors=k_range, weights=weight_options)

    tuningRandomizedSearchCV(knn, param_dist)

    

    # train a KNeighborsClassifier model on the training set

    knn = KNeighborsClassifier(n_neighbors=27, weights='uniform')

    knn.fit(X_train, y_train)

    

    # make class predictions for the testing set

    y_pred_class = knn.predict(X_test)

    

    print('########### KNeighborsClassifier ###############')

    

    accuracy_score = evalClassModel(knn, y_test, y_pred_class, True)



    #Data for final graph

    methodDict['KNN'] = accuracy_score * 100

Knn()
def treeClassifier():

    # Calculating the best parameters

    tree = DecisionTreeClassifier()

    featuresSize = feature_cols.__len__()

    param_dist = {"max_depth": [3, None],

              "max_features": randint(1, featuresSize),

              "min_samples_split": randint(2, 9),

              "min_samples_leaf": randint(1, 9),

              "criterion": ["gini", "entropy"]}

    tuningRandomizedSearchCV(tree, param_dist)

    

    # train a decision tree model on the training set

    tree = DecisionTreeClassifier(max_depth=3, min_samples_split=8, max_features=6, criterion='entropy', min_samples_leaf=7)

    tree.fit(X_train, y_train)

    

    # make class predictions for the testing set

    y_pred_class = tree.predict(X_test)

    

    print('########### Tree classifier ###############')

    

    accuracy_score = evalClassModel(tree, y_test, y_pred_class, True)



    #Data for final graph

    methodDict['Tree clas.'] = accuracy_score * 100
treeClassifier()

def randomForest():

    # Calculating the best parameters

    forest = RandomForestClassifier(n_estimators = 20)



    featuresSize = feature_cols.__len__()

    param_dist = {"max_depth": [3, None],

              "max_features": randint(1, featuresSize),

              "min_samples_split": randint(2, 9),

              "min_samples_leaf": randint(1, 9),

              "criterion": ["gini", "entropy"]}

    tuningRandomizedSearchCV(forest, param_dist)

    

    # Building and fitting my_forest

    forest = RandomForestClassifier(max_depth = None, min_samples_leaf=8, min_samples_split=2, n_estimators = 20, random_state = 1)

    my_forest = forest.fit(X_train, y_train)

    

    # make class predictions for the testing set

    y_pred_class = my_forest.predict(X_test)

    

    print('########### Random Forests ###############')

    

    accuracy_score = evalClassModel(my_forest, y_test, y_pred_class, True)



    #Data for final graph

    methodDict['R. Forest'] = accuracy_score * 100
randomForest()
def bagging():

    # Building and fitting 

    bag = BaggingClassifier(DecisionTreeClassifier(), max_samples=1.0, max_features=1.0, bootstrap_features=False)

    bag.fit(X_train, y_train)

    

    # make class predictions for the testing set

    y_pred_class = bag.predict(X_test)

    

    print('########### Bagging ###############')

    

    accuracy_score = evalClassModel(bag, y_test, y_pred_class, True)



    #Data for final graph

    methodDict['Bagging'] = accuracy_score * 100
bagging()
def boosting():

    # Building and fitting 

    clf = DecisionTreeClassifier(criterion='entropy', max_depth=1)

    boost = AdaBoostClassifier(base_estimator=clf, n_estimators=500)

    boost.fit(X_train, y_train)

    

    # make class predictions for the testing set

    y_pred_class = boost.predict(X_test)

    

    print('########### Boosting ###############')

    

    accuracy_score = evalClassModel(boost, y_test, y_pred_class, True)



    #Data for final graph

    methodDict['Boosting'] = accuracy_score * 100
boosting()
def stacking():

    # Building and fitting 

    clf1 = KNeighborsClassifier(n_neighbors=1)

    clf2 = RandomForestClassifier(random_state=1)

    clf3 = GaussianNB()

    lr = LogisticRegression()

    stack = StackingClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr)

    stack.fit(X_train, y_train)

    

    # make class predictions for the testing set

    y_pred_class = stack.predict(X_test)

    

    print('########### Stacking ###############')

    

    accuracy_score = evalClassModel(stack, y_test, y_pred_class, True)



    #Data for final graph

    methodDict['Stacking'] = accuracy_score * 100
stacking()
import tensorflow as tf

import argparse





batch_size = 100

train_steps = 1000



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)



def train_input_fn(features, labels, batch_size):

    """An input function for training"""

    # Convert the inputs to a Dataset.

    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))



    # Shuffle, repeat, and batch the examples.

    return dataset.shuffle(1000).repeat().batch(batch_size)



def eval_input_fn(features, labels, batch_size):

    """An input function for evaluation or prediction"""

    features=dict(features)

    if labels is None:

        # No labels, use only features.

        inputs = features

    else:

        inputs = (features, labels)



    # Convert the inputs to a Dataset.

    dataset = tf.data.Dataset.from_tensor_slices(inputs)



    # Batch the examples

    assert batch_size is not None, "batch_size must not be None"

    dataset = dataset.batch(batch_size)



    # Return the dataset.

    return dataset
# Define Tensorflow feature columns

age = tf.feature_column.numeric_column("Age")

gender = tf.feature_column.numeric_column("Gender")

family_history = tf.feature_column.numeric_column("family_history")

food = tf.feature_column.numeric_column("food")

sleep = tf.feature_column.numeric_column("sleep")

anonymity = tf.feature_column.numeric_column("anonymity")

confidence_selfesteem = tf.feature_column.numeric_column("confidence_selfesteem")

abuse = tf.feature_column.numeric_column("abuse")

feature_columns = [age, gender, family_history, food, sleep, anonymity, confidence_selfesteem, abuse]
# Build a DNN with 2 hidden layers and 10 nodes in each hidden layer.

model = tf.estimator.DNNClassifier(feature_columns=feature_columns,

                                    hidden_units=[10, 10],

                                    optimizer=tf.keras.optimizers.Adam())
model.train(input_fn=lambda:train_input_fn(X_train, y_train, batch_size), steps=train_steps)
eval_result = model.evaluate(

    input_fn=lambda:eval_input_fn(X_test, y_test, batch_size))



print('\nTest set accuracy: {accuracy:0.2f}\n'.format(**eval_result))



#Data for final graph

accuracy = eval_result['accuracy'] * 100

methodDict['NN DNNClasif.'] = accuracy
predictions = list(model.predict(input_fn=lambda:eval_input_fn(X_train, y_train, batch_size=batch_size)))
# Generate predictions from the model

template = ('\nIndex: "{}", Prediction is "{}" ({:.1f}%), expected "{}"')



# Dictionary for predictions

col1 = []

col2 = []

col3 = []





for idx, input, p in zip(X_train.index, y_train, predictions):

    v  = p["class_ids"][0] 

    class_id = p['class_ids'][0]

    probability = p['probabilities'][class_id] # Probability

    

    # Adding to dataframe

    c1=col1.append(idx) # Index

    col2.append(v) # Prediction

    col3.append(input) # Expected

    

   

    #print(template.format(idx, v, 100 * probability, input))





results = pd.DataFrame({'index':col1, 'prediction':col2, 'expected':col3})

results.tail()
def plotSuccess():

    s = pd.Series(methodDict)

    s = s.sort_values(ascending=False)

    plt.figure(figsize=(12,8))

    #Colors

    ax = s.plot(kind='bar') 

    for p in ax.patches:

        ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.005, p.get_height() * 1.005))

    plt.ylim([70.0, 90.0])

    plt.xlabel('Method')

    plt.ylabel('Percentage')

    plt.title('Success of methods')

     

    plt.show()
plotSuccess()
# Generate predictions with the best method

clf = AdaBoostClassifier()

clf.fit(X, y)

dfTestPredictions = clf.predict(X_test)



# Write predictions to csv file

# We don't have any significative field so we save the index

results = pd.DataFrame({'Index': X_test.index, 'Treatment': dfTestPredictions})

# Save to file

# This file will be visible after publishing in the output section

results.to_csv('results.csv', index=False)

results.tail(1)
# Write predictions to csv file

# We don't have any significative field so we save the index

results = pd.DataFrame({'Index': X_test.index, 'Treatment': dfTestPredictions})

# Save to file

# This file will be visible after publishing in the output section

results.to_csv('submission.csv', index=False)