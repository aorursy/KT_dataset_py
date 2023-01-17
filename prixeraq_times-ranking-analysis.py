# Import all the libraries

import pandas as pd

import numpy as np

from numpy import set_printoptions
# import dataset

times = pd.read_csv("../input/world-university-rankings/timesData.csv")

times.head()
print('dtypes of times dataset:')

times.dtypes



# mix between float, string and integer.
print('number of NaNs per column:')

times.isna().sum()



# 4 columns have missing values. Female male ratio has the most with 233. 
# drop university name and country because they're strings

# drop female male ratio because 233 rows are missing

# drop total score because it's too similar to world rank

times.drop(columns=['university_name', 'country', 'female_male_ratio', 'total_score'], inplace=True)
times.head()
# converting string values to numeric

# world rank values are numeric from 1 to 100 afterwards they're string like 100-150

# convert world rank to numeric and rest is converted to NaN

# we're only interested in top 100, top 50, top 10 so we only care about the first 100 for the binarizer



times['world_rank'] = pd.to_numeric(times['world_rank'], errors='coerce')



# fill with 101 so it's below the binarize threshold of 100

times['world_rank'].fillna(101, inplace=True)



# binarizer converts value to 1 if it's above the threshold

# so we need to invert world rank i.e. make negative

times['world_rank'] = (times['world_rank'] * -1)



# prepare object or string columns for numeric conversion

# Few columns had "-" for missing value, replace with 0

# num students has ",", replace with nothing ""

# international students has "%", replace with nothing ""

str_cols = times.select_dtypes(['object']).columns

times[str_cols] = times[str_cols].replace('-', 0)

times['num_students'] = times['num_students'].str.replace(',', '')

times['international_students'] = times['international_students'].str.replace('%', '')



# convert object or string columns to numeric

times[str_cols] = times[str_cols].apply(pd.to_numeric, errors='coerce', axis=1)



# convert international students percentage to decimal

times['international_students'] = times['international_students'] / 100
# determine number of NaNs

times.isna().sum()
# drop remaining NaNs 

times.dropna(inplace=True)
# check dataframe, dtypes and NaNs

print(times.dtypes)

print(times.isna().sum())

times.head()
# convert times dataframe to array

times_array = times.values

X = times_array[:,1:]

y_ = times_array[:,[0]]
set_printoptions(precision=3, suppress=True)

X[:5]
y_[:5]
# drop world_rank, not needed

times.drop(columns='world_rank', inplace=True)
# create binary variable

from sklearn.preprocessing import Binarizer



top_n = -50 + (-1)



binarizer=Binarizer(threshold=top_n).fit(y_)

y_binary=binarizer.transform(y_)



y_binary[:5]



y_reshaped = np.ravel(y_binary)

y_reshaped
# reshape using ravel() so that it works with LogisticRegression

y_reshaped = np.ravel(y_binary)

y_reshaped
times.head()
# Univariate selection using Chi-squared 

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2 



# feature selection (we select the 3 best)

test = SelectKBest(score_func=chi2, k=3)

fit = test.fit(X,y_reshaped)

print("Scores")



print(fit.scores_)



print("The 3 attributes with the highest scores are: teaching, research and num_students ")

print()

print('teaching: university score for teaching')

print('reserach: university score for research (volume, income and reputation)')

print('num_students: number of students at the university')



features=fit.transform(X)

features[0:5,:]
# Recursive Feature Elimiantion

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression



#Logistic regression

model = LogisticRegression(solver='liblinear')



rfe = RFE(model, 3) #  we want to find the 3 top features

fit = rfe.fit(X, y_reshaped)



print(f'Number of features {fit.n_features_:d}')

print(f'Selected features {fit.support_}')

print(f'Ranking of features {fit.ranking_}')

print()

print("Top features seem to be teaching, research and citations")
from sklearn.ensemble import ExtraTreesClassifier



model = ExtraTreesClassifier(n_estimators=100)

model.fit(X,y_reshaped)



print(model.feature_importances_)

print()

print("Top features seem to be citations, research and teaching")
top_unis = [10, 50, 100]

univariate = []

rfe_ranking = []

etc_features = []



for n in top_unis:



    top_n = (n + 1) * (-1)



    binarizer=Binarizer(threshold=top_n).fit(y_)

    y_binary=binarizer.transform(y_)



    y_reshaped = np.ravel(y_binary)



    print('*************************************************************')

    print('Univariate Selection using Chi-Squared: top', n)



    #set_printoptions(precision=3, suppress)



    # feature selection (we select the 3 best)

    test = SelectKBest(score_func=chi2, k=3)

    fit = test.fit(X,y_reshaped)

    print("Scores")



    univariate.append(fit.scores_)



    features=fit.transform(X)

    print(features[0:5,:])



    print('*************************************************************')

    print('Recursive Feature Elimination: top', n)

    print()



    model = LogisticRegression(solver='liblinear')



    rfe = RFE(model, 3) #  we want to find the 3 top features

    fit = rfe.fit(X, y_reshaped)



    print(f'Number of features {fit.n_features_:d}')

    print(f'Selected features {fit.support_}')

    print(f'Ranking of features {fit.ranking_}')



    rfe_ranking.append(fit.ranking_)

    print()



    print('*************************************************************')

    print('ExtraTreeClassifier: top', n)



    model = ExtraTreesClassifier(n_estimators=100, random_state=7)

    model.fit(X,y_reshaped)



    print(model.feature_importances_)

    etc_features.append(model.feature_importances_)



print(times.head())

    

print('top unis:', top_unis)

print(univariate)

print(rfe_ranking)

print(etc_features)
times.head()
# Answer for Univariate Selection. 

# First row is top 10, then top 50, then top 100

univariate
# Answer for Recursive Feature Selection.

# First row is top 10, then top 50, then top 100

rfe_ranking
# Answer for ExtraTreeClassifier

# First row is top 10, then top 50, then top 100

etc_features
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
top_unis = [10, 50, 100]

train_test_split_accuracy = []

k_fold_accuracy = []



for n in top_unis:



    top_n = (n + 1) * (-1)



    binarizer=Binarizer(threshold=top_n).fit(y_)

    y_binary=binarizer.transform(y_)



    y_reshaped = np.ravel(y_binary)

    

    print('*************************************************************')

    print('train-test-split: top', n)

    

    # we need to make it reproducible, so we use a seed for the pseudo-random

    test_size = 0.3

    seed = 7



    # the actual split

    X_train, X_test, y_train, y_test = train_test_split(X, y_reshaped, test_size=test_size, random_state=seed)



    # Let's do the log regresssion

    model = LogisticRegression(solver='liblinear')

    model.fit(X_train,y_train)



    # Now let's find the accurary with the test split

    result = model.score(X_test, y_test)

    train_test_split_accuracy.append(result)



    print(f'Accuracy {result*100:5.3f}')

    print()

    

    print('*************************************************************')

    print('k-fold-10 validation: top', n)

    print()

    

    # KFold

    splits = 10

    kfold = KFold(n_splits=splits, random_state=seed)



    #Logistic regression

    model = LogisticRegression(solver='liblinear')



    # Obtain the performance measure - accuracy

    results = cross_val_score(model, X, y_reshaped, cv=kfold)

    k_fold_accuracy.append(results.mean())

    

    print(f'Logistic regression, k-fold {splits:d} - Accuracy {results.mean()*100:5.3f}% ({results.std()*100:5.3f}%)')

    print()

    

    

train_test_accuracy = [ '%.3f' % elem for elem in train_test_split_accuracy]

kfold_accuracy = [ '%.3f' % elem for elem in k_fold_accuracy]



print('Top unis: ', top_unis)

print(train_test_accuracy)

print(kfold_accuracy)

print('Accuracy decreases as the number of universities to be classified increases')
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report
top_unis = [10, 50, 100]

scoring = ['accuracy', 'neg_log_loss', 'roc_auc']

k_fold_accuracy = []



for n in top_unis:



    top_n = (n + 1) * (-1)



    binarizer=Binarizer(threshold=top_n).fit(y_)

    y_binary=binarizer.transform(y_)



    y_reshaped = np.ravel(y_binary)



    print('*************************************************************')

    

    for score in scoring:

        

        print('*************************************************************')

        print(score, ', top', n)



        # StratifiedKFold because top10 with kfold causes an error (bug)

        splits = 10

        skfold = StratifiedKFold(n_splits=splits, random_state=7)





        #Logistic regression

        model = LogisticRegression(solver='liblinear')



        # Obtain the performance measure - accuracy

        results = cross_val_score(model, X, y_reshaped, scoring=score, cv=skfold)



        print(score, f': {results.mean():.3f}')

        print()



    print('*************************************************************')

    print('Confusion Matrix, top', n)

    

    test_size=0.3

    seed=7



    X_train, X_test, Y_train, Y_test = train_test_split(X, y_reshaped, test_size=test_size, random_state=seed)



    model = LogisticRegression(solver='liblinear')

    log_reg = model.fit(X_train, Y_train)



    Y_predicted = log_reg.predict(X_test)



    c_matrix=confusion_matrix(Y_test, Y_predicted)



    print(c_matrix)



    print()

    print(f'Accuracy {model.score(X_test, Y_test)*100:.3f}')

    print(f'Accuracy check with conf. matrix {(c_matrix[0,0]+c_matrix[1,1])/c_matrix.sum()*100:.3f}')

    print()

    

    print('*************************************************************')

    print('Classification Report, top', n)    

    

    report = classification_report(Y_test, Y_predicted, digits=3)

    

    print(f'Accuracy {model.score(X_test, Y_test)*100:.3f}')

    print()

    print(report)



print('All the scores decrease as the number of universities in the group to predict increases')