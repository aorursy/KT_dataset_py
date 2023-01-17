# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Let's explore the Titanic dataset in a semi-rigorous manner that gives a true measure

# of accuracy.  The emphasis here will be on proper cross-validation technique and the

# avoidance of data leakage, which is a very common mis-step for beginners.



df_train = pd.read_csv('../input/train.csv', header = 0) # Read in the training set

df_test = pd.read_csv('../input/test.csv', header = 0)   # Read in the test set



print("There are %d samples and %d features in the training set" %(df_train.shape[0], df_train.shape[1]))

print("There are %d Survived samples and %d Perished samples" % 

      (sum(df_train["Survived"]==0), sum(df_train["Survived"]==1)))

print("\nHere are the null counts for each feature")

print(df_train.isnull().sum(axis=0))



df_train.head()
from sklearn import preprocessing



# Let's sort the columns properly

y = df_train["Survived"].as_matrix()  # Our ground truth labels



# These variables we will assume are good as is and the magnitude has meaning (although we are not sure

# exactly the scale of the axis of each of these)

X_num = df_train[["Age", "SibSp", "Parch", "Fare", "Pclass"]]



# What about the categorical features?  Which are we going to keep?

X_cat = df_train.select_dtypes(include=[object])



# Let's just drop the troublesome columns for now - Cabin contains a lot of NaN and possible

# clerical errors.  Embarked contains only 2 NaN so we can keep that.  Let's assume

# Name and Ticket are not important for now.

X_cat = X_cat.drop("Cabin", 1)

#X_cat = X_cat.drop("Embarked", 1)

X_cat = X_cat.drop("Name", 1)

X_cat = X_cat.drop("Ticket", 1)



# Because Embarked only has two missing values.

null_cols = X_cat[X_cat.isnull().any(axis=1)].index.values



# Columns 61 and 829 contain null Embarked columns

print(df_train.iloc[null_cols])



# Temporarily fill these in - Because there are only 2 missing values, we will accept a tiny bit of data

# leakage here and just fill in a value of 'C' randomly.  This makes the LabelEncoding and OneHotEncoding

# easier.  TODO: Add these step to our pipelines.

X_cat = X_cat.fillna('C')



le = preprocessing.LabelEncoder()



X_le = X_cat.apply(le.fit_transform)



enc = preprocessing.OneHotEncoder()



enc.fit(X_le)



X_enc = enc.transform(X_le)



# Append sex to the other variables

X = np.c_[X_num.as_matrix(), X_enc.toarray()]



# Repeat for test data

Xt_num = df_test[["Age", "SibSp", "Parch", "Fare", "Pclass"]]

Xt_cat = df_test.select_dtypes(include=[object])



# Let's just drop the troublesome columns for now - Cabin contains a lot of NaN and possible

# clerical errors.  Embarked contains only 2 NaN so let's come back to that one.  Let's assume

# Name and Ticket are not important for now.

Xt_cat = Xt_cat.drop("Cabin", 1)

#Xt_cat = Xt_cat.drop("Embarked", 1)

Xt_cat = Xt_cat.drop("Name", 1)

Xt_cat = Xt_cat.drop("Ticket", 1)



Xt_le = Xt_cat.apply(le.fit_transform)

enc.fit(Xt_le)



Xt_enc = enc.transform(Xt_le)



# Append sex to the other variables

Xt = np.c_[Xt_num.as_matrix(), Xt_enc.toarray()]



print(X.shape)
from sklearn.feature_extraction.text import CountVectorizer



X_name = df_train["Name"]



cv = CountVectorizer();



X_dict = cv.fit_transform(X_name).toarray()



# Looks like 1509 unique words in the "Name" feature.  We will have to be more intelligent about this.

#cv.get_feature_names()

X_counts=X_dict.sum(axis=0)

X_freq=np.argsort(X_counts)[::-1]

top = X_freq[:100]



tokes = np.asarray(cv.get_feature_names())



# Looking at the top 100, we can pick out some interesting common features that may be important

tokes[top]
# Pull out columsn for titles. mr, miss, mrs, master, jr, and dr.



named_cols = ['mr', 'miss', 'mrs', 'master', 'jr', 'dr', 'rev']



title_cols = [cv.vocabulary_[x] for x in named_cols]



X_dict = X_dict[:, title_cols]



# Append name to the other variables

X = np.c_[X, X_dict]
# 6 new variables added

X.shape
# Do the same thing for the test set

X_name = df_test["Name"]



cv = CountVectorizer();



X_dict = cv.fit_transform(X_name).toarray()



# Looks like 1509 unique words in the "Name" feature.  We will have to be more intelligent about this.

#cv.get_feature_names()

X_counts=X_dict.sum(axis=0)

X_freq=np.argsort(X_counts)[::-1]

top = X_freq[:100]



tokes = np.asarray(cv.get_feature_names())



title_cols = [cv.vocabulary_[x] for x in named_cols]



X_dict = X_dict[:, title_cols]



# Append name to the other variables

Xt = np.c_[Xt, X_dict]



Xt.shape
# Data visualization - I'm not going to explore the distributions of the data as this is covered

# extensively in other kernels.  But, let's reduce the dimensionality of the data and check for

# natural distance in the data we have created.

#

# This will require some pre-processing: Impute NaN values and normalize.  PCA expects equal

# weighting of features (TODO: I am not sure about TSNE requirements on features)

#

# TSNE will be the default dimensionality reducer. 

# TSNE generally gives nice results.  Perplexity can be manipulated (between 5-50) for

# better visualization, but generally does not affect results.

# Note that it can be computationally expensive so PCA can be used instead, although the results

# generally will not be as nice.



from sklearn import manifold

from sklearn.decomposition import PCA



perplexity=10

n_components=2



imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)

Xi = imp.fit(X).transform(X)



pca = PCA(n_components=2)



scalar = preprocessing.StandardScaler()

scalar.fit(Xi)



X_t= scalar.transform(Xi)



tsne = manifold.TSNE(n_components=n_components, init='random',

                         random_state=0, perplexity=perplexity)





#Y = pca.fit_transform(X_t)

Y = tsne.fit_transform(X_t)
# Plot our visualization results

# Plot the data in 3 dimensions

import matplotlib.pyplot as plt



classes = ['Survived', 'Perished']



fig = plt.figure(figsize=(10,10))



plt.scatter(Y[y==0, 0], Y[y==0, 1], c="b", s=100, alpha = 0.5)

plt.scatter(Y[y==1, 0], Y[y==1, 1], c="orange", alpha = 0.5, s=100, edgecolors = 'black')



plt.legend(classes)



plt.show()
# Set up a nested cross-validation pipeline on our cleaned Titanic dataset.  Note that I was

# careful to say cleaned, and not pre-processed.  We have done nothing to the data other than

# inspect it, throw away clearly clerical or overly sparse data, and one-hot encode categorical 

# data.  Nothing about the structure/distribution of the data has been used to help clean the

# data thus far.  We did run some pre-processing for visualization processes, but then threw away

# that analysis.  X contains our clean feature data.  y contains our target data.

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB



from sklearn.ensemble import RandomForestClassifier, VotingClassifier



# Logistic Regression

from sklearn.linear_model import LogisticRegression 



# MLP

from sklearn.neural_network import MLPClassifier



from sklearn.pipeline import Pipeline as Pipeline_SK

from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict, StratifiedKFold

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



# Set up a stratified CV for both outer and inner CVs.  Stratified because we want to test on

# the real distribution of our skewed data - Just going to use 3 for speed.  Increase this

# for more realistic results

inner_cv = StratifiedKFold(n_splits=5, shuffle=True)

outer_cv = StratifiedKFold(n_splits=5, shuffle=True)



# Standard svm with a rbf kernel and balanced weights to account for the skewed data.  We pick

# a regularization constant of 9, but we will tune this.

svm = SVC(kernel='rbf', class_weight='balanced', C=9, random_state=1, probability=True)

lr = LogisticRegression(class_weight='balanced', C=9, random_state=1)

gnb = GaussianNB()

mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, alpha=1e-4,

                    solver='sgd', tol=1e-4, random_state=1,

                    learning_rate_init=.1)



imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)



# Add this lda to the pipe if you like.

lda = LDA(solver='eigen', shrinkage='auto')



p_grid_svm = {"svm__C": [0.05, 0.1, 0.2, 0.4, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]}

p_grid_lr = {"lr__C": [0.05, 0.1, 0.2, 0.4, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]}

p_grid_gnb = {}

p_grid_mlp = {"mlp__hidden_layer_sizes": [(100,), (50,), (100, 50,), (50, 20,)],

             "mlp__alpha": [1e-5, 1e-4, 1e-3, 1e-2]}



# We are setting up separate pipes and grids for each classifier, but you can do this on the Voting

# Classifier as well and just index over one grid as v_svm__svm__C, v_lr__lr__C, etc...



# Parameter tuning - inner CV - SVM

pipe = Pipeline_SK([('imp', imp), ('standardscalar', preprocessing.StandardScaler()), ('svm', svm)])

clf_svm = GridSearchCV(estimator=pipe, param_grid=p_grid_svm, cv=inner_cv, scoring='accuracy')



# Parameter tuning - inner CV - LR

pipe = Pipeline_SK([('imp', imp), ('standardscalar', preprocessing.StandardScaler()), ('lr', lr)])

clf_lr = GridSearchCV(estimator=pipe, param_grid=p_grid_lr, cv=inner_cv, scoring='accuracy')



# Parameter tuning - inner CV - MLP

pipe = Pipeline_SK([('imp', imp), ('standardscalar', preprocessing.StandardScaler()), ('mlp', mlp)])

clf_mlp = GridSearchCV(estimator=pipe, param_grid=p_grid_mlp, cv=inner_cv, scoring='accuracy')



# Parameter tuning - inner CV - GB - No Parameters to tune so we won't use the inner loop

pipe = Pipeline_SK([('imp', imp), ('standardscalar', preprocessing.StandardScaler()), ('lda', lda), ('gnb', gnb)])

clf_gnb=pipe



# Set up a voting classifier on the probability outputs of our 4 sample classifiers.  This should 

# generally give us a few more percent.  A decision tree would be a good addition to this.

eclf2 = VotingClassifier(estimators=[('v_svm', clf_svm), ('v_lr', clf_lr), ('v_mlp', clf_mlp), ('v_gnb', clf_gnb)], voting='soft')



# Measure scores - outer CV

ns_vote = cross_val_score(eclf2, X=X, y=y, cv=outer_cv, scoring='accuracy')

ypred_vote = cross_val_predict(eclf2, X, y, cv=outer_cv)
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score)



nested_score=ns_vote

ypred=ypred_vote



print(nested_score)



print("Accuracy score to expect from an unseen dataset: %0.03f (+/- %0.03f)" % (nested_score.mean(), nested_score.std()))



print("Accuracy score to expect from an unseen dataset: %0.03f %%" % (100*accuracy_score(y, ypred)))





cm = confusion_matrix(y, ypred)



print(classification_report(y, ypred, target_names=classes))
clf_svm.fit(X, y)
# And predict

ypred_test = clf_svm.predict(Xt)



ypred_test
# And save in the proper format

# One column each of PassengerId	Survived

data = pd.DataFrame(df_test['PassengerId'])



data['Survived'] = ypred_test



data.to_csv('submission_svm.csv', index=False)