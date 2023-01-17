### This project is for educational purposes, as a student project. Do not use this code to make production decisions! :)





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/software-defect-prediction/jm1.csv')
df.dtypes
df.describe()
###something is wrong, those last five that start with uniq_Op are objects - 

df['uniq_Op'] = pd.to_numeric(df['uniq_Op'], errors='coerce') ## convert to number, make NaNs from ?s 

df['uniq_Opnd'] = pd.to_numeric(df['uniq_Opnd'], errors='coerce') ## convert to number, make NaNs from ?s 

df['total_Op'] = pd.to_numeric(df['total_Op'], errors='coerce') ## convert to number, make NaNs from ?s 

df['total_Opnd'] = pd.to_numeric(df['total_Opnd'], errors='coerce') ## convert to number, make NaNs from ?s 

df['branchCount'] = pd.to_numeric(df['branchCount'], errors='coerce') ## convert to number, make NaNs from ?s 

df['defects'] = df['defects'].astype(int)



df = df.dropna()

df = df.reset_index(drop=True)



df.describe()
### Okay - now we can do some work.

# import sklearn model_selection code

from sklearn import model_selection



# Split-out validation dataset

array = df.values

X = array[:,0:21]

Y = array[:,21]



# Get Training and Validation sets

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.2, random_state=7)
### USE MACHINE LEARNING ALGORITHM ###



from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, Y_train)

predictions = model.predict(X_validation)



print("1. Accuracy: {}".format(accuracy_score(Y_validation, predictions)))

print("2. Confusion Matrix:n{}".format(pd.crosstab(Y_validation, predictions, rownames=['True'], colnames=['Predicted'])))
#Ah, interesting - we have a warning:

###/opt/conda/lib/python3.6/site-packages/sklearn/discriminant_analysis.py:388: UserWarning: Variables are collinear.

### warnings.warn("Variables are collinear.")



# Collinear variables are going to mess with this model. We need to remove them - but how?

## Can we use some kind of cross-checking random forest model?



# Create correlation matrix

corr_matrix = df.corr().abs()



# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]



# Drop features 

df_modified = df.drop(df[to_drop], axis=1)



df_modified.describe()
array.shape[1]
### USE MACHINE LEARNING ALGORITHM ###

### Again. ###



# Split-out validation dataset

array = df_modified.values



cols = array.shape[1]



X = array[:,0:cols-1]

Y = array[:,cols-1]



# Get Training and Validation sets

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.2, random_state=7)



from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, Y_train)

predictions = model.predict(X_validation)



print("1. Accuracy: {}".format(accuracy_score(Y_validation, predictions)))

print("2. Confusion Matrix:\n{}".format(pd.crosstab(Y_validation, predictions, rownames=['True'], colnames=['Predicted'])))
### Weird. Let's try a feature importance exercise.



from sklearn.datasets import make_classification

from sklearn.ensemble import ExtraTreesClassifier



import matplotlib.pyplot as plt





# Build a forest and compute the feature importances

forest = ExtraTreesClassifier(n_estimators=250,

                              random_state=0)



forest.fit(X_train, Y_train)

importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(X.shape[1]):

    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



# Plot the feature importances of the forest

plt.figure()

plt.title("Feature importances")

plt.bar(range(X.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(X.shape[1]), indices)

plt.xlim([-1, X.shape[1]])

plt.show()
### Let's grab the top five features.

X_modified = array[:,indices[0:5]]



# Get Training and Validation sets

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X_modified, Y, test_size=0.2, random_state=7)



from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, Y_train)

predictions = model.predict(X_validation)



print("1. Accuracy: {}".format(accuracy_score(Y_validation, predictions)))

print("2. Confusion Matrix:\n{}".format(pd.crosstab(Y_validation, predictions, rownames=['True'], colnames=['Predicted'])))
### So far the best we've done is 82.7% accuracy. Not great. Maybe we can try Kfold or CV

### With LogisticRegression, it looks like there's a LogisticRegressionCV package. 

from sklearn.linear_model import LogisticRegressionCV

clf = LogisticRegressionCV(cv=5, random_state=0, multi_class='multinomial').fit(X_train, Y_train)

predictions = clf.predict(X_validation)

shaped = clf.predict_proba(X_validation).shape



print("1. Accuracy: {}".format(accuracy_score(Y_validation, predictions)))
### 82.67%. No better. 



### What if we tried it with the full range of features from the beginning?

# Get Training and Validation sets

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.2, random_state=7)

clf = LogisticRegressionCV(cv=5, random_state=0, multi_class='multinomial', max_iter=1000).fit(X_train, Y_train)

predictions = clf.predict(X_validation)

shaped = clf.predict_proba(X_validation).shape



print("1. Accuracy: {}".format(accuracy_score(Y_validation, predictions)))

#Result: 82.17%. Worse!

## Finally, let's function out a way to get accuracy based on how many of the features we include, and see how many we really need to be accurate.

def accuracies_for_selectedtopfeatures(df, num_top_features=1):

    # Split-out validation dataset

    array = df_modified.values

    cols = array.shape[1]

    

    X = array[:,0:cols-1]

    Y = array[:,cols-1]



    # Get Training and Validation sets

    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.2, random_state=7)

    

    # Build a forest and compute the feature importances

    forest = ExtraTreesClassifier(n_estimators=250,

                              random_state=0)



    forest.fit(X_train, Y_train)

    importances = forest.feature_importances_

    std = np.std([tree.feature_importances_ for tree in forest.estimators_],

             axis=0)

    indices = np.argsort(importances)[::-1]

    

    ### Let's grab the top features.

    X_modified = array[:,indices[0:num_top_features]]

    

    # Get Training and Validation sets

    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X_modified, Y, test_size=0.2, random_state=7)

    

    from sklearn.linear_model import LogisticRegressionCV

    clf = LogisticRegressionCV(cv=5, random_state=0, multi_class='multinomial').fit(X_train, Y_train)

    predictions = clf.predict(X_validation)

    return accuracy_score(Y_validation, predictions)
top_five = accuracies_for_selectedtopfeatures(df, 5)

print(top_five)



top_four = accuracies_for_selectedtopfeatures(df, 4)

print(top_four)



top_three = accuracies_for_selectedtopfeatures(df, 3)

print(top_three)
## It looks like 4 is the magic number of features -- which are they?

print(df.columns[indices[0:4]], importances[indices[0:4]])
###Findings:

### The four most critical predictive features of defects in code are:



### loc - Lines of Code (importance 0.14621159)

### v(g) - Cyclomatic Complexity (importance 0.0849397)

### iv(g) - Design Complexity (importance 0.07449589)

### d - Difficulty (importance 0.07433222)



## A cross-validated logistic regression can achieve a predictive accuracy of 82.72% using these four features.

## Including more or fewer features from the data set yields reduced accuracy.