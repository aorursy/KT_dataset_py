# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # For creating plots

import matplotlib.ticker as mtick # For specifying the axes tick format 

import matplotlib.pyplot as plt

import sklearn

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# loading into Panda dataframe

df = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# looking at the features we have

df.columns.values
df.head()
# Look at the unique data in each variable

for item in df.columns:

    print(item)

    print (df[item].unique())
# converting yes/no variables to 1/0 

# for simplicity we combine "no" and "no internet service" to 0

df = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")

for i in ["Partner", 

          "Dependents", 

          "PhoneService", 

          "OnlineSecurity",

          "DeviceProtection",

          "OnlineBackup",

          "TechSupport", 

          "StreamingTV", 

          "StreamingMovies", 

          "PaperlessBilling",

          "Churn"]:

    df[i].replace(to_replace='Yes', value=1, inplace=True)

    df[i].replace(to_replace='No', value=0, inplace=True)

    df[i].replace(to_replace='No internet service', value=0, inplace=True)

    

df["gender"].replace(to_replace='Female', value=1, inplace=True)

df["gender"].replace(to_replace='Male', value=0, inplace=True)    

    

# creating dummy variables for categorial features; categorial features automatically dropped

dummy = ["MultipleLines", "InternetService", "Contract", "PaymentMethod"]

df = pd.get_dummies(df,prefix=dummy, columns=dummy)

df.head()
# View unique data in each column after converting to 1/0

for item in df.columns:

    print(item)

    print (df[item].unique())
df.dtypes
# Need to convert TotalCharges to a float, cannot be "object"

df["TotalCharges"] = pd.to_numeric(df.TotalCharges, errors='coerce')

df.dtypes
# checking for empty values

df.isnull().sum(axis = 0)
# since its only 11 cases where TotalCharges is missing we simply drop those rows. 

# we could have replaced with a 0, but that's not correct.

df = df.dropna()

df.isnull().sum(axis = 0)
# We can look at statistics for the variables

df.describe()
# We plot the distribution of each variable

df_plot = df.drop(columns=["customerID"])

item = "TotalCharges"

for item in df_plot.columns:

    df[item].plot(kind="hist",bins=10, figsize=(12,6), title = item)

    plt.show()
# Calculate and display the correlation matrix

# Note that the target "Churn" is one of the variables in the data set, and we see that it is not especially strongly correlated with a single feature

# We see natural relationships in the data such as "Streaming Movies" being correlated with "TotalCharges"

corr = df.corr()



fig = plt.figure(figsize = (20,10))

# Plot the heatmap

sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns,

           center=0)
# Ranking features with highest correlation with Churn.

# (Naive) idea is to remove columns with low correlation to simplify our feature set and reduce model training time.

# By only looking at correlations between single variables we miss higher order effects and may discard variables that can help our model. 

# The variables with highest correlation to the target variable might not be the same as the most important features in our model.

# The sign of the correlation depends on how the feature is formulated.

# Since we do not have a lot of data and features, we do not remove any in this case.

df.corr()["Churn"].sort_values(axis=0,ascending=False)
# We are happy with our data set and ready to train a classifier.

# Split into test and training set, random selection of customers.



from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.3)

train_x = train.drop(columns=["Churn","customerID"])

train_y = train[["Churn"]]

test_x = test.drop(columns=["Churn","customerID"])

test_y = test[["Churn"]]
# Decision tree

# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

# Have to set a max depth to avoid overtraining!

from sklearn.ensemble import RandomForestClassifier

from sklearn import tree

clf = tree.DecisionTreeClassifier(max_depth=5)

#clf = RandomForestClassifier(n_estimators=40)

clf = clf.fit(train_x, train_y)

pred = clf.predict(test_x)

# The probability of each class is predicted, which is the fraction of training samples of the same class in a leaf

prob = clf.predict_proba(test_x)
# Calculating ROC AUC and plotting the ROC curve, 

# Calculate the fpr and tpr for all thresholds of the classification

from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, threshold = roc_curve(test_y, prob[:,1])

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.legend([round(roc_auc_score(test_y, prob[:,1]),2)]);

plt.show()
# The best way to think about hyperparameters is like the settings of an algorithm that can be adjusted to optimize performance.

# Adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py



# Utility function to report best scores

def report(results, n_top=3):

    for i in range(1, n_top + 1):

        candidates = np.flatnonzero(results['rank_test_score'] == i)

        for candidate in candidates:

            print("Model with rank: {0}".format(i))

            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(

                  results['mean_test_score'][candidate],

                  results['std_test_score'][candidate]))

            print("Parameters: {0}".format(results['params'][candidate]))

            print("")



            

# Specify parameters and distributions to sample from. 

# For now, I select only parameters that are related to the design of the tree. Ideally all parameters are included in the search space.

# One should perform the search in several  iterations, starting from a very rough grid narrowing down each iteration.

# I restrict the search space for max_depth to a rather shallow tree on purpose, so that we end up with a model which can be easily be interpreted in the end.

# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

from scipy.stats import randint as sp_randint

param_dist = {"max_depth": np.arange(1,8),

              "min_samples_split": np.arange(2,10),

              "min_samples_leaf": np.arange(1,10),

              "max_features": np.arange(1,10)

             }



# Run randomized search, print out results, and store the best parameters

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from time import time

n_iter_search = 1000

random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search, cv=5)

start = time()

random_search.fit(train_x, train_y)

print("RandomizedSearchCV took %.2f seconds for %d candidates"

      " parameter settings." % ((time() - start), n_iter_search))

report(random_search.cv_results_)
# We train the Decision Tree again with the best parameters.

clf = tree.DecisionTreeClassifier(min_samples_split = 8,

                                  min_samples_leaf = 1,

                                  max_depth = 6,

                                  max_features = 9

                                 )

clf = clf.fit(train_x, train_y)

pred = clf.predict(test_x)

# The probability of each class is predicted, which is the fraction of training samples of the same class in a leaf

prob = clf.predict_proba(test_x)
# Calculating ROC AUC and plotting the ROC curve, 

# Calculate the fpr and tpr for all thresholds of the classification

# Ironically we get a slightly worse ROC AUC after Hyperparameter tuning.

from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, threshold = roc_curve(test_y, prob[:,1])

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.legend([round(roc_auc_score(test_y, prob[:,1]),2)]);

plt.show()
# We start by looking at our tree model to see if it makes intuitive sense; which features are used in decisions.

import graphviz 

dot_data = tree.export_graphviz(clf, out_file=None,

                                feature_names=train_x.columns.values[0:],

                                class_names=["Non-Churn","Churn"],

                                filled=True, rounded=True,

                                special_characters=True)  

graph = graphviz.Source(dot_data)

graph
# We can also look at the importance of each feature in the classifier.

# The deeper our tree, the more features are used.

plt.figure(figsize=(10,10))

plt.barh(train_x.columns,clf.feature_importances_)

plt.xticks(rotation=90)

plt.show()
# We can also look at the "confusion matrix", which shows us the labeling made by our model for all the test customers.

# The "Confusion matrix" is the source of the business case we build for operationalizing the model, using it to contact customers.

# We also print out standard evaluation metrics for our model.

# from sklearn documentation



from sklearn.metrics import confusion_matrix

import itertools





def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

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



    

cnf_matrix = confusion_matrix(test_y, pred)

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure()

class_names = ['Not churned','churned']



plot_confusion_matrix(cnf_matrix, classes=class_names,

                      title='Confusion matrix, without normalization')



# Plot normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,

                      title='Normalized confusion matrix')



plt.show()



from sklearn.metrics import classification_report

eval_metrics = classification_report(test_y, pred, target_names=class_names)

print(eval_metrics)
# We compute the the gain curve which tells us what %y of Churners we correctly identify after inspecting %x of customers.

results = pd.DataFrame({'y': test_y["Churn"], 'y_proba': prob[:,1]})

results = results.sort_values(by='y_proba', ascending=False).reset_index(drop=True)

results.index = results.index + 1

results.index = results.index / len(results.index) * 100



sns.set_style('darkgrid')

results['Gains Curve'] = results.y.cumsum() / results.y.sum() * 100

results['Baseline'] = results.index

base_rate = test_y["Churn"].sum() / len(test_y) * 100

results[['Gains Curve', 'Baseline']].plot(style=['-', '--', '--'])

pd.Series(data=[0, 100, 100], index=[0, base_rate, 100]).plot(style='--')

plt.title('Cumulative Gains')

plt.xlabel('% of Customers inspected')

plt.ylabel("% of Churners identified")

plt.legend(['Gains Curve', 'Baseline', 'Ideal']);
# We can also plot the lift itself.

# We see that we don't manage much more than 2.5, which is rather poor.

# Some strange behavior around 0, not sure why... Otherwise looks good.

sns.set_style('darkgrid')

results['Lift Curve']=(results.y.cumsum() / results.y.sum() * 100)/results.index



results['Gains Curve'] = results.y.cumsum() / results.y.sum() * 100

results['Baseline'] = results.index



results[['Lift Curve']].plot(style=['-', '--', '--'])

plt.title('Lift chart')

plt.xlabel('% of Customers inspected')

plt.ylabel("Lift")

plt.legend(['Lift Curve']);
# Finally, getting ready for using the prediction in the real world. 

# We want to perform an action towards the customers that our model predicts as the most likely to churn.

# If we select for example the top 10% of Churners that our model predicts we get a new confusion matrix for the business case for operationalization.



# Creating a list [predicted churn probability, predicted churn 1/0, actual churn] which we then sort by predicted churn probability

list = []

for i in range(len(test_y)):

    list.append([prob[i][1],pred[i],test_y.iloc[i,0]])

list.sort(reverse=True)



# Keeping the top x customers, setting predicted churn to 0 by hand for the remaining customers

# E.g. we create a threshold for "label = true" by picking top x%

# Note that this % must give a number larger than the number of guesses that have probability 0.5



x = 0.10

top = []

for i in range(len(list)):

    if i < len(test_y)*x:

        top.append(list[i])

    else:

        top.append([list[i][0],0,list[i][2]]) #setting predicted churn to 0

        

# Creating our input lists for the function to plot confusion matrix

test_y_top = []

pred_top  = []

for i in range(len(top)):

    test_y_top.append(top[i][2])

    pred_top.append((top[i][1]))



# Number of customers we guess churn and thus will contact

sum(pred_top)
# In the confusion matrix, we see that when we pick the top 10% of customers to contact, we guess 70% right, vs 25% by guessing randomly.

cnf_matrix = confusion_matrix(test_y_top, pred_top)

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure()

class_names = ['Not churned','churned']



plot_confusion_matrix(cnf_matrix, classes=class_names,

                      title='Confusion matrix, without normalization')



# Plot normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,

                      title='Normalized confusion matrix')



plt.show()



from sklearn.metrics import classification_report

eval_metrics = classification_report(test_y_top, pred_top, target_names=class_names)

print(eval_metrics)